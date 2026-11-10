"""Exercise document indexing, search, and grounded inference in a published Native AOT Monolith."""
from concurrent.futures import ThreadPoolExecutor
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
from urllib.parse import urlencode


helpers = runpy.run_path(str(Path(__file__).with_name("native-aot-stream-embeddings.py")))
native_image = helpers["native_image"]
wait_for_port = helpers["wait_for_port"]
read_request = helpers["read_request"]
audit_entries = helpers["audit_entries"]

KEY = "native-rag-test-only-key"
INDEX = "/v1/rag/indexes/handbook"
DOCUMENTS = INDEX + "/documents"
QUESTION = "What is the orbital period?"
FACT = "The orbital period of the example planet is 37 days."
OTHER = "The accounting office closes at noon."
ANSWER = "The orbital period is 37 days [1]."
UPDATED_FACT = "The orbital period of the example planet is 41 days."
BACKUP_FACT = "The calendar records an orbital period of 53 days."
EXTRA_FACT = "The astronomical almanac records the example planet."
BACKUP_ID = "reference/暦"
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
        concurrent_adds = threading.Barrier(2, timeout=10)

        class Upstream(BaseHTTPRequestHandler):
            def do_POST(self):
                body = read_request(self)
                if body.get("input") in ([BACKUP_FACT], [EXTRA_FACT]):
                    concurrent_adds.wait()
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
                               FAILED_CHUNK: [0, 1, 0], UPDATED_FACT: [4, 0, 0],
                               BACKUP_FACT: [1, 1, 0], EXTRA_FACT: [1, 2, 0]}
                    self.send_json(200, {"data": [{"index": i, "embedding": vectors[texts[i]]}
                                                 for i in reversed(range(len(texts)))]})
                    return

                assert self.path == "/chat/v1/chat/completions"
                assert body["model"] == "upstream-chat"
                system, user = body["messages"]
                assert system["role"] == "system" and "citation labels" in system["content"]
                assert user["role"] == "user" and user["content"].endswith("\n\nQuestion:\n" + QUESTION)
                evidence = json.loads(user["content"].split("\n", 1)[1].split("\n\nQuestion:", 1)[0])
                assert evidence == expected_evidence
                self.send_json(200, {"choices": [{"message": {"content": expected_answer}}]})

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
                    connection.request(method, route, json.dumps(payload) if payload is not None else None, headers)
                    response = connection.getresponse()
                    data = response.read()
                    assert response.status == status, (method, route, status, response.status, data)
                    return json.loads(data)

            def put(port, model_id, docs, status=200, key=KEY):
                return request(port, INDEX, {"embeddingModel": model_id, "documents": docs}, status, "PUT", key)

            def upsert(port, document, status=200, key=KEY):
                return request(port, DOCUMENTS, document, status, "PUT", key)

            def delete(port, document_id, status=200, key=KEY):
                route = DOCUMENTS + "?" + urlencode({"documentId": document_id})
                return request(port, route, None, status, "DELETE", key)

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
                    assert upsert(port, docs[0], 404)["error"]["code"] == "index_not_found"
                    assert delete(port, "planet", 404)["error"]["code"] == "index_not_found"
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
                    expected_evidence, expected_answer = [best], ANSWER
                    answer = request(port, INDEX + "/answer", question, 200)
                    assert answer["answer"] == ANSWER and answer["model"] == "chat"
                    assert answer["id"].startswith("cmpl-") and answer["sources"] == [best]

                    count = len(calls)
                    assert upsert(port, docs[0], 401, key=None)["error"]["code"] == "unauthorized"
                    assert delete(port, "planet", 401, key=None)["error"]["code"] == "unauthorized"
                    assert upsert(port, {"id": "planet"}, 400)["error"]["code"] == "invalid_document"
                    assert len(calls) == count

                    backup = {"id": BACKUP_ID, "text": BACKUP_FACT, "source": "https://example.test/calendar"}
                    extra = {"id": "almanac", "text": EXTRA_FACT}
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        additions = [pool.submit(upsert, port, doc) for doc in (backup, extra)]
                        assert sorted(result.result()["documentCount"] for result in additions) == [3, 4]
                    assert sorted(call[1]["input"] for call in calls[count:]) == sorted([[BACKUP_FACT], [EXTRA_FACT]])

                    update = {"id": "planet", "text": UPDATED_FACT, "title": "Updated planet handbook",
                              "source": "https://example.test/updated-planet"}
                    count = len(calls)
                    assert upsert(port, update) == {
                        "indexId": "handbook", "embeddingModel": "embed", "documentCount": 4, "chunkCount": 4,
                    }
                    assert len(calls) == count + 1 and calls[-1][1]["input"] == [UPDATED_FACT]
                    updated = request(port, INDEX + "/search", query, 200)["hits"]
                    assert [hit["text"] for hit in updated] == [UPDATED_FACT, BACKUP_FACT, EXTRA_FACT]
                    assert updated[0]["source"] == {
                        "documentId": "planet", "chunkNumber": 1, "title": update["title"], "source": update["source"],
                    }
                    failed_update = upsert(port, {**update, "text": "fail-replacement"}, 502)
                    assert failed_update["error"]["code"] == "provider_error"
                    assert request(port, INDEX + "/search", query, 200)["hits"] == updated

                    count = len(calls)
                    assert delete(port, "planet")["documentCount"] == 3
                    assert delete(port, "planet", 404)["error"]["code"] == "document_not_found"
                    assert len(calls) == count

                with start("document-mutations") as port:
                    remaining = request(port, INDEX + "/search", query, 200)["hits"]
                    assert [hit["source"]["documentId"] for hit in remaining] == [BACKUP_ID, "almanac"]
                    assert [hit["text"] for hit in remaining] == [BACKUP_FACT, EXTRA_FACT]
                    expected_evidence = remaining
                    expected_answer = "The calendar records an orbital period of 53 days [1]."
                    after_delete = request(port, INDEX + "/answer", {**question, "topK": 2}, 200)
                    assert after_delete["sources"] == remaining and after_delete["answer"] == expected_answer

                write_models("different-upstream-model")
                with start("model-changed") as port:
                    count = len(calls)
                    changed = request(port, INDEX + "/search", query, 409)
                    assert changed["error"]["code"] == "embedding_model_changed"
                    assert upsert(port, docs[0], 409)["error"]["code"] == "embedding_model_changed"
                    assert delete(port, BACKUP_ID)["documentCount"] == 2
                    assert delete(port, "almanac")["documentCount"] == 1
                    assert delete(port, "accounting")["chunkCount"] == 0
                    assert request(port, INDEX + "/search", query, 200)["hits"] == []
                    assert request(port, INDEX + "/answer", question, 422)["error"]["code"] == "no_context"
                    assert len(calls) == count

                entries = audit_entries(audit)
                execution = [row for row in entries if row["EventType"] == "inference.execution"]
                completion = next(row for row in execution if row["Metadata"].get("completion_id") == after_delete["id"])
                retrieval = next(row for row in execution
                                 if row["Metadata"]["request_id"] == completion["Metadata"]["request_id"]
                                 and row["Metadata"]["operation"] == "rag.answer.retrieve")
                assert sum(row["Description"] == "succeeded" and row["Metadata"]["operation"] == "rag.document.upsert"
                           for row in execution) == 3
                assert retrieval["Description"] == completion["Description"] == "succeeded"
                assert retrieval["Metadata"]["operation"] == "rag.answer.retrieve"
                assert completion["Metadata"]["operation"] == "rag.answer"
                assert retrieval["Metadata"]["request_id"] == completion["Metadata"]["request_id"]
                assert completion["Metadata"]["completion_id"] == after_delete["id"]

                citations = [{key: hit[key] for key in ("citation", "source", "score")}
                             for hit in expected_search["hits"]]
                result = {"sha256": digest, "machine": "win-x64", "ranked_citations": citations,
                          "answer": answer["answer"], "survived_restart_and_failed_replacement": True,
                          "changed_embedding_model_rejected": True, "no_context_rejected": True,
                          "concurrent_additions_preserved": True, "updated_search_text": updated[0]["text"],
                          "failed_update_preserved_document": True, "empty_index_returns_no_hits": True,
                          "search_documents_after_delete": [hit["source"]["documentId"] for hit in remaining],
                          "answer_after_delete": after_delete["answer"],
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
