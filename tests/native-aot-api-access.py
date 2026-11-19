"""Verify access control on a published Windows or Linux x64 Native AOT Monolith."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import threading
import time

helpers = runpy.run_path(str(Path(__file__).with_name("native-aot-stream-embeddings.py")))
KEY = "native-access-test-only-key"
WINDOW_SECONDS = 6


def request(port, method, route, token=None, body=None):
    headers = {}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if body is not None:
        headers["Content-Type"] = "application/json"
    with closing(HTTPConnection("127.0.0.1", port, timeout=10)) as connection:
        connection.request(method, route, json.dumps(body) if body is not None else None, headers)
        response = connection.getresponse()
        return response.status, response.headers, json.loads(response.read())


def check_error(result, status, code):
    actual, headers, body = result
    assert actual == status and body["error"]["code"] == code, (actual, body)
    assert headers["Cache-Control"] == "no-store"
    return headers


def smoke(exe, output):
    digest = helpers["native_image"](exe)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="native-access-", dir=output) as temp:
        work = Path(temp)
        binary = work / exe.name
        shutil.copy2(exe, binary)
        roots = {name: work / name for name in ("config", "model", "vfs", "capability")}
        for directory in roots.values():
            directory.mkdir()

        calls = []
        audit = roots["config"] / "inference-audit.jsonl"
        log = work / "monolith.log"
        env = {key: value for key, value in os.environ.items()
               if key.lower() not in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}}
        env.update({"AIKERNEL_API_KEY": KEY, "AIKERNEL_API_REQUEST_LIMIT": "2",
                    "AIKERNEL_API_WINDOW_SECONDS": str(WINDOW_SECONDS),
                    "ASPNETCORE_ENVIRONMENT": "Production", "DOTNET_ENVIRONMENT": "Production",
                    "NO_PROXY": "127.0.0.1,localhost", "Logging__LogLevel__Microsoft.Hosting.Lifetime": "Information"})
        command = [str(binary), "--urls=http://127.0.0.1:0",
                   *[f"--{name}-root={directory}" for name, directory in roots.items()]]

        for name, value in [("AIKERNEL_API_KEY", None), ("AIKERNEL_API_REQUEST_LIMIT", "0"),
                            ("AIKERNEL_API_WINDOW_SECONDS", "invalid")]:
            invalid = dict(env)
            if value is None:
                invalid.pop(name)
            else:
                invalid[name] = value
            result = subprocess.run(command, cwd=work, env=invalid, capture_output=True, text=True,
                                    timeout=15, creationflags=helpers["PROCESS_FLAGS"])
            assert result.returncode == 2 and name in result.stderr, (name, result.stderr)

        class Upstream(BaseHTTPRequestHandler):
            def do_POST(self):
                assert self.path == "/v1/chat/completions"
                body = helpers["read_request"](self)
                calls.append(body)
                assert body["model"] == "upstream-chat"
                assert self.headers.get("Authorization") is None
                encoded = b'{"choices":[{"message":{"content":"authorized answer"}}]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *_):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Upstream) as upstream:
            model = {"id": "access-chat", "kind": "chat", "provider": "providers.openai",
                     "model": "upstream-chat", "endpoint": f"http://127.0.0.1:{upstream.server_address[1]}/v1",
                     "policy": {"logos": "approve", "ethos": "approve", "pathos": "approve"}}
            (roots["model"] / "access.json").write_text(json.dumps(model), encoding="utf-8")
            thread = threading.Thread(target=upstream.serve_forever, daemon=True)
            thread.start()
            with log.open("wb") as output_stream:
                process = subprocess.Popen(command, cwd=work, env=env, stdout=output_stream,
                                           stderr=subprocess.STDOUT, creationflags=helpers["PROCESS_FLAGS"])
                try:
                    port = helpers["wait_for_port"](process, log)
                    chat = "/v1/chat/completions"
                    payload = {"model": "access-chat", "messages": [{"role": "user", "content": "access smoke test"}]}

                    for key in (None, "incorrect-token"):
                        headers = check_error(request(port, "POST", chat, key, payload), 401, "unauthorized")
                        assert headers["WWW-Authenticate"].startswith("Bearer")
                    check_error(request(port, "GET", "/health"), 401, "unauthorized")
                    assert not calls and not audit.exists()

                    status, _, models = request(port, "GET", "/v1/models", KEY)
                    assert status == 200 and models["data"][0]["id"] == "access-chat"
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        outcomes = list(pool.map(lambda _: request(port, "GET", "/health", KEY), range(4)))
                    assert sorted(result[0] for result in outcomes) == [200, 429, 429, 429]
                    for result in outcomes:
                        if result[0] == 429:
                            check_error(result, 429, "rate_limit_exceeded")

                    limited = request(port, "POST", chat, KEY, {**payload, "stream": True})
                    headers = check_error(limited, 429, "rate_limit_exceeded")
                    assert headers["Content-Type"].startswith("application/json")
                    retry_after = int(headers["Retry-After"])
                    assert 1 <= retry_after <= WINDOW_SECONDS
                    check_error(request(port, "POST", chat, "incorrect-token", payload), 401, "unauthorized")
                    assert not calls and not audit.exists()

                    time.sleep(retry_after + 0.15)
                    status, _, response = request(port, "POST", chat, KEY, payload)
                    assert status == 200 and response["choices"][0]["message"]["content"] == "authorized answer"
                    assert len(calls) == 1
                    records = helpers["audit_entries"](audit)
                    assert [record["Description"] for record in records] == ["allowed", "succeeded"]
                    assert KEY not in log.read_text(encoding="utf-8") and KEY not in audit.read_text(encoding="utf-8")
                    evidence = {"sha256": digest, "missing_and_wrong_key_status": 401,
                                "concurrent_authenticated_statuses": sorted(result[0] for result in outcomes),
                                "over_limit_inference_status": limited[0], "retry_after_seconds": retry_after,
                                "after_window_status": status, "upstream_calls": len(calls),
                                "audit_records": len(records), "invalid_settings_exit": 2}
                    (output / "api-access-result.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
                    print(json.dumps(evidence, indent=2))
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=10)
                    if audit.exists():
                        shutil.copy2(audit, output / "api-access-audit.jsonl")
                    shutil.copy2(log, output / "api-access-monolith.log")
                    upstream.shutdown()
                    thread.join(timeout=5)


if __name__ == "__main__":
    smoke(Path(sys.argv[1]).resolve(strict=True), Path(sys.argv[2]).resolve())
