"""Verify paged local CUDA inference with a published Windows x64 Monolith and TinyStories."""
from contextlib import closing
import hashlib
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import time


helpers = runpy.run_path(str(Path(__file__).with_name("native-aot-stream-embeddings.py")))
native_image = helpers["native_image"]
wait_for_port = helpers["wait_for_port"]
audit_entries = helpers["audit_entries"]
events = helpers["events"]

KEY = "native-paged-llama-test-only-key"
PROMPT = "Once upon a time"
BLOCK_SIZE = 4096
EXPECTED_ASSETS = {
    "stories260K.bin": "b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696",
    "tok512.bin": "037cb335abb25d1fa9e8ecae30ed2a3a8ace9302862ebcdc05d51a6bbb10c312",
}


def smoke(exe, assets, output, ordinal):
    with tempfile.TemporaryDirectory(prefix="local-cuda-", dir=output) as temp:
        work = Path(temp)
        roots = {name: work / name for name in ("config", "model", "vfs", "capability")}
        for directory in roots.values():
            directory.mkdir()
        for name, digest in EXPECTED_ASSETS.items():
            source = assets / name
            assert hashlib.sha256(source.read_bytes()).hexdigest() == digest, name
            shutil.copyfile(source, roots["model"] / name)
        (roots["model"] / "broken.bin").write_bytes(b"invalid")

        def registration(model_id, ram, vram, checkpoint="stories260K.bin", ethos="approve"):
            return {
                "id": model_id, "provider": "providers.paged-llama", "kind": "chat",
                "local": {"checkpoint": checkpoint, "tokenizer": "tok512.bin", "device": ordinal,
                          "blockSize": BLOCK_SIZE, "ramBlocks": ram, "vramBlocks": vram,
                          "contextTokens": 64, "maxNewTokens": 8},
                "policy": {"logos": "approve", "ethos": ethos, "pathos": "approve"},
            }

        registrations = [registration("paged", 4, 2), registration("roomy", 4, 512),
                         registration("broken", 4, 2, "broken.bin"),
                         registration("denied", 4, 2, "missing.bin", "reject")]
        (roots["model"] / "models.json").write_text(json.dumps(registrations), encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if k.lower() not in {"http_proxy", "https_proxy", "all_proxy"}}
        env.update({"ASPNETCORE_ENVIRONMENT": "Production", "DOTNET_ENVIRONMENT": "Production",
                    "AIKERNEL_API_KEY": KEY, "AIKERNEL_API_REQUEST_LIMIT": "1000",
                    "AIKERNEL_API_WINDOW_SECONDS": "60",
                    "Logging__LogLevel__Microsoft.Hosting.Lifetime": "Information"})
        log_path = output / f"gpu-{ordinal}-monolith.log"
        audit = roots["config"] / "inference-audit.jsonl"
        with log_path.open("wb") as log:
            process = subprocess.Popen([str(exe), "--urls=http://127.0.0.1:0",
                                        *[f"--{key}-root={value}" for key, value in roots.items()]],
                                       cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                port = wait_for_port(process, log_path)
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}

                def request(route, payload, status):
                    with closing(HTTPConnection("127.0.0.1", port, timeout=120)) as connection:
                        method = "POST" if payload is not None else "GET"
                        connection.request(method, route, json.dumps(payload) if payload is not None else None, headers)
                        response = connection.getresponse()
                        raw = response.read()
                        assert response.status == status, (status, response.status, raw)
                        return json.loads(raw)

                models = request("/v1/models", None, 200)["data"]
                assert {model["id"] for model in models if model["owned_by"] == "providers.paged-llama"} == {
                    "paged", "roomy", "broken", "denied",
                }
                for model_id, status, code in [("denied", 403, "policy_denied"), ("broken", 502, "provider_error")]:
                    error = request("/v1/completions", {"model": model_id, "prompt": PROMPT}, status)
                    assert error["error"]["code"] == code

                timings = {}
                completions = {}
                for model_id in ("roomy", "paged"):
                    started = time.monotonic()
                    completions[model_id] = request("/v1/completions", {"model": model_id, "prompt": PROMPT}, 200)
                    timings[model_id] = round(time.monotonic() - started, 3)
                baseline = completions["roomy"]["choices"][0]["text"]
                paged = completions["paged"]["choices"][0]
                assert baseline.startswith(", there was a little"), baseline
                assert paged["text"] == baseline and paged["finish_reason"] == "length", paged
                assert list((roots["config"] / "model-swap").glob(".aikernel-swap-*"))

                body = {"model": "paged", "messages": [{"role": "user", "content": PROMPT}], "stream": True}
                with closing(HTTPConnection("127.0.0.1", port, timeout=120)) as connection:
                    connection.request("POST", "/v1/chat/completions", json.dumps(body), headers)
                    response = connection.getresponse()
                    assert response.status == 200
                    assert response.getheader("Content-Type").startswith("text/event-stream")
                    frames = list(events(response))
                assert frames[-1] == "[DONE]"
                chunks = [json.loads(frame) for frame in frames[:-1]]
                assert len(chunks) > 2 and chunks[-1]["choices"][0]["finish_reason"] == "length"
                assert "".join(chunk["choices"][0]["delta"].get("content") or "" for chunk in chunks) == baseline

                overlong = request("/v1/completions", {"model": "paged", "prompt": "word " * 200}, 400)
                assert overlong["error"]["code"] == "context_length_exceeded"

                def counters(completion_id):
                    row = next(row for row in audit_entries(audit)
                               if row["EventType"] == "inference.execution" and row["Description"] == "succeeded"
                               and row["Metadata"]["completion_id"] == completion_id)
                    return {key: int(value) for key, value in row["Metadata"].items()
                            if key.startswith(("weight_", "cuda_")) or key == "checkpoint_bytes"}

                baseline_counts = counters(completions["roomy"]["id"])
                assert baseline_counts["weight_swap_blocks"] == baseline_counts["weight_swap_reads"] == 0
                paged_counts = counters(completions["paged"]["id"])
                stream_counts = counters(chunks[0]["id"])
                for counts in (paged_counts, stream_counts):
                    assert counts["checkpoint_bytes"] > BLOCK_SIZE * 2
                    assert counts["weight_blocks"] > 4 + 2
                    assert 0 < counts["weight_ram_blocks"] <= 4
                    assert 0 < counts["weight_vram_blocks"] <= 2
                    assert counts["weight_swap_blocks"] > 0 and counts["weight_swap_reads"] > 0
                    assert counts["weight_vram_page_ins"] > 0 and counts["cuda_matvec_blocks"] > 0

                return {"device_ordinal": ordinal, "weight_vram_budget_bytes": BLOCK_SIZE * 2,
                        "completion": baseline, "seconds": timings,
                        "paged": paged_counts, "roomy": baseline_counts, "stream": stream_counts}
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=15)
                if audit.exists():
                    shutil.copyfile(audit, output / f"gpu-{ordinal}-audit.jsonl")


if __name__ == "__main__":
    exe = Path(sys.argv[1]).resolve(strict=True)
    assets = Path(sys.argv[2]).resolve(strict=True)
    output = Path(sys.argv[3]).resolve()
    ordinals = [int(value) for value in sys.argv[4:]] or [0]
    output.mkdir(parents=True, exist_ok=True)
    report = {"machine": "win-x64", "sha256": native_image(exe),
              "results": [smoke(exe, assets, output, ordinal) for ordinal in ordinals]}
    (output / "paged-llama-result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
