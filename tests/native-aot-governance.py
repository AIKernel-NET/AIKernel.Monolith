"""Verify the resource gate, thermal policy, and GPU QoS on published Monolith."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import runpy
import subprocess
import tempfile
from urllib.parse import urlencode


standalone = runpy.run_path(str(Path(__file__).with_name("native-aot-standalone.py")))
helpers = standalone["helpers"]
key = standalone["KEY"]
prompt = standalone["PROMPT"]
expected = standalone["EXPECTED"]


def test(exe, assets, output, expect_nvml):
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="governance-", dir=output) as directory:
        work = Path(directory)
        mounts = {name: work / name for name in ("config", "model", "vfs", "capability")}
        for mount in mounts.values():
            mount.mkdir()
        standalone["fixtures"](assets, mounts["model"])
        (mounts["model"] / "broken-tokenizer.json").unlink()

        def model(name, **policy):
            return {"id": name, "provider": "providers.paged-llama",
                    "policy": {"logos": "approve", "ethos": "approve", "pathos": "approve"},
                    "local": {"checkpoint": "stories.gguf", "device": 0, "blockSize": 4096,
                              "ramBlocks": 4, "vramBlocks": 2, "kvRamBlocks": 1, "kvVramBlocks": 1,
                              "contextTokens": 64, "maxNewTokens": 8, **policy}}

        registrations = [
            model("batch", priority="backgroundBatch", encryptSwap=True, zeroWipe=True),
            model("urgent", priority="emergency"),
            model("zero-pcie", maxPcieSwapBytes=0),
            model("thermal", maxTemperatureCelsius=1),
            model("health", maxTemperatureCelsius=120, thermalDeviceLimits=[
                {"device": 0, "maxTemperatureCelsius": 120, "maxMigrationTargetTemperatureCelsius": 1},
                {"device": 1, "maxTemperatureCelsius": 119, "maxMigrationTargetTemperatureCelsius": 118},
            ]),
            model("vetoed") | {"policy": {"logos": "approve", "ethos": "reject", "pathos": "approve"}},
        ]
        (mounts["model"] / "models.json").write_text(json.dumps(registrations), encoding="utf-8")
        env = dict(os.environ, AIKERNEL_API_KEY=key, AIKERNEL_API_REQUEST_LIMIT="1000",
                   Logging__LogLevel__Microsoft__Hosting__Lifetime="Information")
        log_path = output / "monolith.log"
        audit_path = mounts["config"] / "inference-audit.jsonl"
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                [str(exe), "--urls=http://127.0.0.1:0",
                 *[f"--{name}-root={mount}" for name, mount in mounts.items()]],
                cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                port = helpers["wait_for_port"](process, log_path)
                headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

                def admission(model_id, status=200, *, authenticated=True):
                    path = "/v1/models/admission?" + urlencode({"model": model_id} if model_id else {})
                    with closing(HTTPConnection("127.0.0.1", port, timeout=150)) as connection:
                        connection.request("GET", path, headers=headers if authenticated else {})
                        result = connection.getresponse()
                        body = result.read()
                        assert result.status == status, (model_id, result.status, body)
                        assert result.getheader("Cache-Control") == "no-store"
                        return json.loads(body)

                assert admission("health", 401, authenticated=False)["error"]["code"] == "unauthorized"
                assert admission("missing", 404)["error"]["code"] == "model_not_found"
                assert admission(None, 400)["error"]["code"] == "invalid_request"
                assert admission("vetoed")["reason"] == "policy_denied"
                admission_devices = []
                if expect_nvml:
                    ready = admission("health")
                    assert ready["admissible"] and ready["admission_device"] == ready["device"] == 0, ready
                    assert ready["backend"] == "cuda" and ready["reason"] is None, ready
                    admission_devices = ready["devices"]
                    assert admission_devices and admission_devices[0]["device"] == 0, ready
                    for gpu in admission_devices:
                        ordinal = gpu["device"]
                        assert gpu["temperature_celsius"] > 2 and gpu["pci_bus_id"] and gpu["telemetry_error"] is None
                        assert gpu["max_temperature_celsius"] == (119 if ordinal == 1 else 120)
                        assert gpu["max_migration_target_temperature_celsius"] == (
                            1 if ordinal == 0 else 118 if ordinal == 1 else 120)
                        assert gpu["migration_allowed"]
                        assert gpu["migration_target_eligible"] == (ordinal != 0), gpu
                    hot = admission("thermal")
                    assert not hot["admissible"] and hot["admission_device"] is None, hot
                    assert hot["reason"] == "device_temperature_exceeded" and hot["retry_after_seconds"] == 5, hot
                assert not audit_path.exists()
                assert not (mounts["config"] / "model-swap").exists()

                def complete(name, status=200, *, stream=False, retry_after=None):
                    with closing(HTTPConnection("127.0.0.1", port, timeout=150)) as connection:
                        connection.request("POST", "/v1/completions", json.dumps(
                            {"model": name, "prompt": prompt, "stream": stream}), headers)
                        result = connection.getresponse()
                        body = result.read()
                        assert result.status == status, (name, result.status, body)
                        assert result.getheader("Retry-After") == (
                            str(retry_after) if retry_after is not None else None)
                        payload = json.loads(body)
                        if status == 503:
                            if retry_after is None:
                                assert "retry_after_seconds" not in payload["error"]
                            else:
                                assert payload["error"]["retry_after_seconds"] == retry_after
                        return payload

                def metadata(completion_id):
                    return next(entry["Metadata"] for entry in helpers["audit_entries"](audit_path)
                                if entry["EventType"] == "inference.execution" and
                                entry["Metadata"].get("completion_id") == completion_id)

                for name in ("batch", "urgent"):
                    warm = complete(name)
                    assert warm["choices"][0]["text"] == expected
                    assert metadata(warm["id"])["backend"] == "cuda"
                assert complete("zero-pcie", 503)["error"]["code"] == "resource_pcie_budget_exceeded"
                health = metadata(complete("health")["id"])
                if expect_nvml:
                    assert health["backend"] == "cuda" and int(health["device_temperature_celsius"]) > 1
                    loaded = admission("health")
                    assert loaded["admissible"] and loaded["admission_device"] == loaded["device"] == 0, loaded
                    assert loaded["devices"][0]["temperature_celsius"] > 2, loaded
                if health["backend"] == "cuda":
                    for stream in (False, True):
                        rejected = complete("thermal", 503, stream=stream, retry_after=5)
                        assert rejected["error"]["code"] == "device_temperature_exceeded"

                with ThreadPoolExecutor(max_workers=1) as pool, closing(
                        HTTPConnection("127.0.0.1", port, timeout=150)) as connection:
                    connection.request("POST", "/v1/completions", json.dumps(
                        {"model": "batch", "prompt": prompt, "stream": True}), headers)
                    result = connection.getresponse()
                    assert result.status == 200
                    fragments, batch_id, urgent = [], None, None
                    while True:
                        raw = result.readline()
                        assert raw, "batch stream ended before [DONE]"
                        if not raw.startswith(b"data: "):
                            continue
                        payload = raw[6:].strip()
                        if payload == b"[DONE]":
                            break
                        chunk = json.loads(payload)
                        batch_id = chunk["id"]
                        text = chunk["choices"][0].get("text", "")
                        if text:
                            fragments.append(text)
                            if urgent is None:
                                urgent = pool.submit(complete, "urgent")
                    assert "".join(fragments) == expected
                    assert urgent is not None
                    urgent_result = urgent.result(timeout=150)
                    assert urgent_result["choices"][0]["text"] == expected

                batch = metadata(batch_id)
                emergency = metadata(urgent_result["id"])
                assert int(batch["qos_preemptions"]) > 0, batch
                assert batch["qos_priority"] == "BackgroundBatch" and emergency["qos_priority"] == "Emergency"
                assert int(batch["kv_swap_reads"]) > 0
                records = [row["Metadata"].get("completion_id") for row in helpers["audit_entries"](audit_path)
                           if row["EventType"] == "inference.execution"]
                assert records.index(urgent_result["id"]) < records.index(batch_id)
                report = {"pcie_rejected_before_inference": True, "batch_preemptions": int(batch["qos_preemptions"]),
                          "emergency_completed_first": True, "outputs_match": True,
                          "encrypted_kv_swap_reads": int(batch["kv_swap_reads"]),
                          "telemetry_route": health["backend"],
                          "temperature_celsius": health.get("device_temperature_celsius"),
                          "admission_devices": admission_devices}
                encoded = json.dumps(report, indent=2)
                (output / "governance-result.json").write_text(encoded, encoding="utf-8")
                print(encoded)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe", type=Path)
    parser.add_argument("assets", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expect-nvml", action="store_true")
    args = parser.parse_args()
    test(args.exe.resolve(), args.assets.resolve(), args.output.resolve(), args.expect_nvml)
