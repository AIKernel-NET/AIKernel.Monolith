"""Exercise a published Windows x64 Native AOT Monolith against a loopback provider."""
from contextlib import closing
import hashlib
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time


TEST_API_KEY = "native-stream-test-only-key"


def events(response):
    """Read data-only SSE events, one at a time."""
    while line := response.readline():
        if line.startswith(b"data: "):
            data = line[6:].decode("utf-8").rstrip("\r\n")
            assert response.readline() in (b"\n", b"\r\n"), data
            yield data


def audit_entries(file):
    if not file.exists():
        return []
    return [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines()]


def native_image(exe):
    image = exe.read_bytes()
    assert image[:2] == b"MZ"
    pe = struct.unpack_from("<I", image, 0x3C)[0]
    assert image[pe:pe + 4] == b"PE\0\0"
    assert struct.unpack_from("<H", image, pe + 4)[0] == 0x8664
    optional = pe + 24
    assert struct.unpack_from("<H", image, optional)[0] == 0x20B
    assert struct.unpack_from("<II", image, optional + 112 + 14 * 8) == (0, 0)
    return hashlib.sha256(image).hexdigest()


def wait_for_port(process, log):
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline and process.poll() is None:
        match = re.search(r"Now listening on: http://127\.0\.0\.1:(\d+)",
                          log.read_text(encoding="utf-8", errors="replace"))
        if match:
            return int(match[1])
        time.sleep(0.1)
    raise RuntimeError(log.read_text(encoding="utf-8", errors="replace"))


def request(port, route, payload):
    connection = HTTPConnection("127.0.0.1", port, timeout=15)
    connection.request("POST", route, json.dumps(payload), {"Content-Type": "application/json", "Authorization": f"Bearer {TEST_API_KEY}"})
    return connection


def read_request(handler):
    if handler.headers.get("Transfer-Encoding", "").lower() != "chunked":
        return json.loads(handler.rfile.read(int(handler.headers["Content-Length"])))
    chunks = []
    while size := int(handler.rfile.readline().split(b";", 1)[0], 16):
        chunks.append(handler.rfile.read(size))
        assert handler.rfile.read(2) == b"\r\n"
    while handler.rfile.readline() not in (b"\r\n", b""):
        pass
    return json.loads(b"".join(chunks))


def smoke(exe, output, capability_exe=None):
    digest = native_image(exe)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="native-inference-", dir=output) as temp:
        work = Path(temp)
        binary = work / exe.name
        shutil.copy2(exe, binary)
        roots = {name: work / name for name in ("config", "model", "vfs", "capability")}
        for directory in roots.values():
            directory.mkdir()

        audit = roots["config"] / "inference-audit.jsonl"
        log_path = work / "monolith.log"
        calls = []
        capability_calls = []
        releases = {mode: threading.Event() for mode in ("chat", "text")}
        released_before_finish = {}
        chat_route = "/v1/chat/completions"
        text_route = "/v1/completions"

        class Upstream(BaseHTTPRequestHandler):
            def do_POST(self):
                body = read_request(self)
                if self.path == "/capability/v1/embeddings":
                    assert body == {"model": "upstream-embedding", "input": ["capability-native-check"],
                                    "encoding_format": "float"}
                    capability_calls.append(body)
                    self.send_json(200, {"data": [{"index": 0, "embedding": [0.125, -0.25]}]})
                    return

                calls.append((self.path, body, audit_entries(audit)[-1]))
                if self.path == "/embed/v1/embeddings":
                    assert body["model"] == "upstream-embedding"
                    assert body["encoding_format"] == "float"
                    texts = body["input"]
                    data = [{"index": index, "embedding": [0.125 + index, -0.25 - index]}
                            for index in range(len(texts))]
                    if texts[0] == "bad-index":
                        data[1]["index"] = 0
                    self.send_json(200, {"data": list(reversed(data))})
                    return

                assert self.path == "/chat/v1/chat/completions"
                assert body["model"] == "upstream-chat"
                mode = body["messages"][0]["content"]
                if not body.get("stream"):
                    self.send_json(200, {"choices": [{"message": {"content": "buffered answer"}}]})
                    return
                assert self.headers["Accept"] == "text/event-stream"
                if mode == "early-error":
                    self.send_json(503, {"error": {"message": "upstream unavailable"}})
                    return

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b": keepalive\r\n\r\n")
                self.frame({"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
                self.frame({"choices": [{"index": 0, "delta": {"content": "first"}, "finish_reason": None}]})
                if mode == "truncated":
                    return

                released_before_finish[mode] = releases[mode].wait(10)
                self.frame({"choices": [{"index": 0, "delta": {"content": " second"}, "finish_reason": None}]})
                self.frame({"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

            def frame(self, payload):
                self.wfile.write(("data: " + json.dumps(payload) + "\n\n").encode())
                self.wfile.flush()

            def send_json(self, status, payload):
                encoded = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *_):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Upstream) as upstream:
            upstream_port = upstream.server_address[1]
            thread = threading.Thread(target=upstream.serve_forever, daemon=True)
            thread.start()

            def model(alias, kind, endpoint, upstream_model, ethos="approve"):
                return {"id": alias, "provider": "providers.openai", "kind": kind,
                        "endpoint": f"http://127.0.0.1:{upstream_port}/{endpoint}/v1",
                        "model": upstream_model,
                        "policy": {"logos": "approve", "ethos": ethos, "pathos": "approve"}}

            registrations = [
                model("chat-test", "chat", "chat", "upstream-chat"),
                model("embed-test", "embedding", "embed", "upstream-embedding"),
                model("embed-denied", "embedding", "embed", "upstream-embedding", "reject"),
            ]
            (roots["model"] / "models.json").write_text(json.dumps(registrations), encoding="utf-8")

            env = {key: value for key, value in os.environ.items()
                   if key.lower() not in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}}
            env.update({"ASPNETCORE_ENVIRONMENT": "Production", "DOTNET_ENVIRONMENT": "Production",
                        "AIKERNEL_API_KEY": TEST_API_KEY, "AIKERNEL_API_REQUEST_LIMIT": "1000",
                        "AIKERNEL_API_WINDOW_SECONDS": "60",
                        "NO_PROXY": "127.0.0.1,localhost", "Logging__LogLevel__Microsoft.Hosting.Lifetime": "Information"})
            with log_path.open("wb") as log:
                process = subprocess.Popen([str(binary), "--urls=http://127.0.0.1:0",
                                            *[f"--{key}-root={value}" for key, value in roots.items()]],
                                           cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           creationflags=subprocess.CREATE_NO_WINDOW)
                try:
                    port = wait_for_port(process, log_path)

                    def json_request(route, payload, status):
                        with closing(request(port, route, payload)) as connection:
                            response = connection.getresponse()
                            data = json.loads(response.read())
                            assert response.status == status, (status, response.status, data)
                            return data

                    def chat_payload(text, stream=True, model="chat-test"):
                        return {"model": model, "stream": stream, "messages": [{"role": "user", "content": text}]}

                    buffered = json_request(chat_route, chat_payload("buffered", stream=False), 200)
                    assert buffered["choices"][0]["message"]["content"] == "buffered answer"

                    for mode in ("chat", "text"):
                        chat = mode == "chat"
                        route = chat_route if chat else text_route
                        payload = chat_payload(mode) if chat else {"model": "chat-test", "prompt": mode, "stream": True}
                        with closing(request(port, route, payload)) as connection:
                            response = connection.getresponse()
                            assert response.status == 200
                            assert response.getheader("Content-Type").startswith("text/event-stream")
                            stream = events(response)
                            first = json.loads(next(stream))
                            assert first["object"] == ("chat.completion.chunk" if chat else "text_completion")
                            choice = first["choices"][0]
                            assert choice["finish_reason"] is None
                            assert (choice["delta"]["content"] if chat else choice["text"]) == "first"
                            assert first["model"] == "chat-test"
                            if chat:
                                assert choice["delta"]["role"] == "assistant"
                            releases[mode].set()
                            second = json.loads(next(stream))
                            tail = json.loads(next(stream))
                            assert first["id"] == second["id"] == tail["id"]
                            choice = second["choices"][0]
                            assert (choice["delta"]["content"] if chat else choice["text"]) == " second"
                            assert tail["choices"][0]["finish_reason"] == "length"
                            assert list(stream) == ["[DONE]"]
                            assert released_before_finish[mode]

                    error = json_request(chat_route, chat_payload("early-error"), 502)
                    assert error["error"]["code"] == "provider_error"
                    with closing(request(port, chat_route, chat_payload("truncated"))) as connection:
                        response = connection.getresponse()
                        assert response.status == 200
                        frames = list(events(response))
                        assert len(frames) == 2
                        assert json.loads(frames[0])["choices"][0]["delta"]["content"] == "first"
                        assert json.loads(frames[1])["error"]["code"] == "provider_error"

                    vectors = json_request("/v1/embeddings", {"model": "embed-test", "input": ["alpha", "beta"]}, 200)
                    assert vectors == {"object": "list", "model": "embed-test", "data": [
                        {"object": "embedding", "index": 0, "embedding": [0.125, -0.25]},
                        {"object": "embedding", "index": 1, "embedding": [1.125, -1.25]},
                    ]}
                    single = json_request("/v1/embeddings", {"model": "embed-test", "input": "alpha"}, 200)
                    assert single["data"] == vectors["data"][:1]
                    assert calls[-2][1]["input"] == ["alpha", "beta"]
                    assert calls[-1][1]["input"] == ["alpha"]
                    bad = json_request("/v1/embeddings", {"model": "embed-test", "input": ["bad-index", "beta"]}, 502)
                    assert bad["error"]["code"] == "provider_error"

                    count = len(calls)
                    assert json_request(chat_route, chat_payload("unused", model="missing"), 404)["error"]["code"] == "model_not_found"
                    for model_id, value, status, code in [
                        ("missing", "alpha", 404, "model_not_found"),
                        ("chat-test", "alpha", 400, "unsupported_operation"),
                        ("embed-test", [123], 400, "invalid_request"),
                        ("embed-denied", "alpha", 403, "policy_denied"),
                    ]:
                        error = json_request("/v1/embeddings", {"model": model_id, "input": value}, status)
                        assert error["error"]["code"] == code
                    assert len(calls) == count == 8
                    for _, _, before in calls:
                        assert before["EventType"] == "inference.policy" and before["Description"] == "allowed"
                    entries = audit_entries(audit)
                    assert sum(row["Description"] == "succeeded" for row in entries) == 5
                    assert sum(row["Description"] == "provider_error" for row in entries) == 3
                    assert sum(row["Description"] == "denied" for row in entries) == 1
                    assert all("upstream unavailable" not in row["Description"] for row in entries)

                    result = {"sha256": digest, "machine": "win-x64", "streamed_before_upstream_finish": released_before_finish,
                              "truncated_stream_rejected": True, "embedding_vectors": vectors["data"],
                              "upstream_calls": count, "audit_records": len(entries)}
                    if capability_exe is not None:
                        result["capability_native_sha256"] = native_image(capability_exe)
                        capability_copy = work / capability_exe.name
                        shutil.copy2(capability_exe, capability_copy)
                        probe = subprocess.run([str(capability_copy), f"http://127.0.0.1:{upstream_port}/capability/v1"],
                                               env=env, cwd=work, capture_output=True, text=True, timeout=15,
                                               creationflags=subprocess.CREATE_NO_WINDOW, check=True)
                        vector_json, vector_hash = probe.stdout.splitlines()
                        assert json.loads(vector_json) == [0.125, -0.25]
                        assert vector_hash == "sha256:" + hashlib.sha256(vector_json.encode("utf-8")).hexdigest()
                        assert len(capability_calls) == 1
                        result["capability_embedding_output_hash"] = vector_hash
                    (output / "smoke-result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                    print(json.dumps(result, indent=2))
                finally:
                    for release in releases.values():
                        release.set()
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=10)
                    if audit.exists():
                        shutil.copy2(audit, output / "smoke-audit.jsonl")
                    shutil.copy2(log_path, output / "smoke-monolith.log")
                    upstream.shutdown()
                    thread.join(timeout=5)


if __name__ == "__main__":
    capability = Path(sys.argv[3]).resolve(strict=True) if len(sys.argv) > 3 else None
    smoke(Path(sys.argv[1]).resolve(strict=True), Path(sys.argv[2]).resolve(), capability)
