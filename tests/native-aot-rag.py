"""Exercise document indexing, search, and grounded inference in a published Native AOT Monolith."""
from contextlib import closing, contextmanager
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


helpers = runpy.run_path(str(Path(__file__).with_name("native-aot-stream-embeddings.py")))
native_image = helpers["native_image"]
wait_for_port = helpers["wait_for_port"]
read_request = helpers["read_request"]
audit_entries = helpers["audit_entries"]

KEY = "native-rag-test-only-key"
INDEX = "/v1/rag/indexes/handbook"
QUESTION = "What is the orbital period?"
FACT = "The orbital period of the example planet is 37 days."
OTHER = "The accounting office closes at noon."
ANSWER = "The orbital period is 37 days [1]."
PREFIX = "A" * 998 + "  "
FAILED_CHUNK = "B" * 1000


def smoke(exe, output):
    digest = native_image(exe)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="native-rag-", dir=output) as temp:
        work = Path(temp)
        binary = work / exe.name
        shutil.copy2(exe, binary)
        roots = {name: work / name for name in ("config", "model", "vfs", "capability")}
        for directory in roots.values():
            directory.mkdir()
        audit = roots["config"] / "inference-audit.jsonl"
        calls = []

        class Upstream(BaseHTTPRequestHandler):
            def do_POST(self):
                body = read_request(self)
                decision = audit_entries(audit)[-1]
                assert decision["EventType"] == "inference.policy" and decision["Description"] == "allowed"
                calls.append((self.path, body))
                if self.path == "/embed/v1/embeddings":
                    assert body["model"] == "upstream-embedding"
                    assert body["encoding_format"] == "float"
                    texts = body["input"]
                    if texts == ["fail-replacement"]:
                        self.send_json(503, {"error": "upstream unavailable"})
                        return
                    vectors = {PREFIX.strip(): [1, 1, 0], FACT: [3, 0, 0], OTHER: [-2, 0, 0],
                               QUESTION: [2, 0, 0], "No matches": [0, 0, 1], "zero vector": [0, 0, 0],
                               FAILED_CHUNK: [0, 1, 0]}
                    self.send_json(200, {"data": [{"index": i, "embedding": vectors[texts[i]]}
                                                 for i in reversed(range(len(texts)))]})
                    return

                assert self.path == "/chat/v1/chat/completions"
                assert body["model"] == "upstream-chat"
                system, user = body["messages"]
                assert system["role"] == "system" and "citation labels" in system["content"]
                assert user["role"] == "user" and user["content"].endswith("\n\nQuestion:\n" + QUESTION)
                evidence = json.loads(user["content"].split("\n", 1)[1].split("\n\nQuestion:", 1)[0])
                assert evidence == expected_search["hits"][:1]
                self.send_json(200, {"choices": [{"message": {"content": ANSWER}}]})

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
            thread = threading.Thread(target=upstream.serve_forever, daemon=True)
            thread.start()

            def model(alias, kind, upstream_model, ethos="approve"):
                endpoint = "embed" if kind == "embedding" else "chat"
                return {"id": alias, "provider": "providers.openai", "kind": kind,
                        "endpoint": f"http://127.0.0.1:{upstream.server_port}/{endpoint}/v1",
                        "model": upstream_model,
                        "policy": {"logos": "approve", "ethos": ethos, "pathos": "approve"}}

            def write_models(embedding_model="upstream-embedding"):
                registrations = [model("embed", "embedding", embedding_model),
                                 model("denied", "embedding", embedding_model, "reject"),
                                 model("chat", "chat", "upstream-chat")]
                (roots["model"] / "models.json").write_text(json.dumps(registrations), encoding="utf-8")

            env = {key: value for key, value in os.environ.items()
                   if key.lower() not in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}}
            env.update({"ASPNETCORE_ENVIRONMENT": "Production", "DOTNET_ENVIRONMENT": "Production",
                        "AIKERNEL_API_KEY": KEY, "AIKERNEL_API_REQUEST_LIMIT": "1000",
                        "AIKERNEL_API_WINDOW_SECONDS": "60", "NO_PROXY": "127.0.0.1,localhost",
                        "Logging__LogLevel__Microsoft.Hosting.Lifetime": "Information"})

            @contextmanager
            def start(name):
                log_path = output / f"rag-{name}.log"
                with log_path.open("wb") as log:
                    process = subprocess.Popen([str(binary), "--urls=http://127.0.0.1:0",
                                                *[f"--{key}-root={value}" for key, value in roots.items()]],
                                               cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT,
                                               creationflags=subprocess.CREATE_NO_WINDOW)
                    try:
                        yield wait_for_port(process, log_path)
                    finally:
                        process.terminate()
                        process.wait(timeout=10)

            def request(port, route, payload, status, method="POST", key=KEY):
                headers = {"Content-Type": "application/json"}
                if key is not None:
                    headers["Authorization"] = f"Bearer {key}"
                with closing(HTTPConnection("127.0.0.1", port, timeout=15)) as connection:
                    connection.request(method, route, json.dumps(payload), headers)
                    response = connection.getresponse()
                    data = response.read()
                    assert response.status == status, (method, route, status, response.status, data)
                    return json.loads(data)

            def put(port, model_id, docs, status=200, key=KEY):
                return request(port, INDEX, {"embeddingModel": model_id, "documents": docs}, status, "PUT", key)

            query = {"query": QUESTION}
            question = {"model": "chat", "question": QUESTION, "topK": 1}
            docs = [{"id": "planet", "title": "Planet handbook", "source": "https://example.test/planet",
                     "text": PREFIX + FACT}, {"id": "accounting", "text": OTHER}]
            replacements = [{"id": "other", "text": FAILED_CHUNK * 32 + "fail-replacement"}]
            write_models()
            try:
                with start("index") as port:
                    assert request(port, INDEX + "/search", query, 404)["error"]["code"] == "index_not_found"
                    assert put(port, "embed", docs, 401, key=None)["error"]["code"] == "unauthorized"
                    assert put(port, "embed", [docs[0], docs[0]], 400)["error"]["code"] == "invalid_documents"
                    assert put(port, "missing", docs, 404)["error"]["code"] == "model_not_found"
                    assert put(port, "denied", docs, 403)["error"]["code"] == "policy_denied"
                    assert not calls
                    assert put(port, "embed", docs) == {
                        "indexId": "handbook", "embeddingModel": "embed", "documentCount": 2, "chunkCount": 3,
                    }
                    assert calls[-1][1]["input"] == [PREFIX.strip(), FACT, OTHER]
                    expected_search = request(port, INDEX + "/search", query, 200)
                    best, second = expected_search["hits"]
                    assert best == {"citation": "[1]", "score": 1, "text": FACT,
                                    "source": {"documentId": "planet", "chunkNumber": 2,
                                               "title": "Planet handbook", "source": "https://example.test/planet"}}
                    assert second["citation"] == "[2]" and second["source"]["chunkNumber"] == 1
                    assert 0.70 < second["score"] < 0.71
                    assert put(port, "embed", replacements, 502)["error"]["code"] == "provider_error"
                    assert [call[1]["input"] for call in calls[-2:]] == [
                        [FAILED_CHUNK] * 32, ["fail-replacement"],
                    ]
                    zero = put(port, "embed", [{"id": "zero", "text": "zero vector"}], 502)
                    assert zero["error"]["code"] == "invalid_embedding_vectors"

                with start("restart") as port:
                    assert request(port, INDEX + "/search", query, 200) == expected_search
                    no_evidence = request(port, INDEX + "/answer", {**question, "question": "No matches"}, 422)
                    assert no_evidence["error"]["code"] == "no_context"
                    assert all(route != "/chat/v1/chat/completions" for route, _ in calls)
                    answer = request(port, INDEX + "/answer", question, 200)
                    assert answer["answer"] == ANSWER and answer["model"] == "chat"
                    assert answer["id"].startswith("cmpl-") and answer["sources"] == [best]

                write_models("different-upstream-model")
                with start("model-changed") as port:
                    count = len(calls)
                    changed = request(port, INDEX + "/search", query, 409)
                    assert changed["error"]["code"] == "embedding_model_changed"
                    assert len(calls) == count

                entries = audit_entries(audit)
                execution = [row for row in entries if row["EventType"] == "inference.execution"]
                retrieval, completion = execution[-2:]
                assert retrieval["Description"] == completion["Description"] == "succeeded"
                assert retrieval["Metadata"]["operation"] == "rag.answer.retrieve"
                assert completion["Metadata"]["operation"] == "rag.answer"
                assert retrieval["Metadata"]["request_id"] == completion["Metadata"]["request_id"]
                assert completion["Metadata"]["completion_id"] == answer["id"]

                citations = [{key: hit[key] for key in ("citation", "source", "score")}
                             for hit in expected_search["hits"]]
                result = {"sha256": digest, "machine": "win-x64", "ranked_citations": citations,
                          "answer": answer["answer"], "survived_restart_and_failed_replacement": True,
                          "changed_embedding_model_rejected": True, "no_context_rejected": True,
                          "upstream_calls": len(calls), "audit_records": len(entries)}
                (output / "rag-result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                print(json.dumps(result, indent=2))
            finally:
                if audit.exists():
                    shutil.copy2(audit, output / "rag-audit.jsonl")
                upstream.shutdown()
                thread.join(timeout=5)


if __name__ == "__main__":
    smoke(Path(sys.argv[1]).resolve(strict=True), Path(sys.argv[2]).resolve())
