# AIKernel.Monolith

AIKernel.Monolith is the official AIOS distribution host built on the AIKernel SDK. It packages the Semantic Runtime, deterministic routing, external capability loading, model provider registry, virtual file system access, and an OpenAI-compatible API into one container-first runtime.

AIKernel.Monolith は、AIKernel SDK 上に構築される公式 AIOS ディストリビューションホストです。Semantic Runtime、決定論的 routing、外部 Capability loading、model provider registry、VFS access、OpenAI 互換 API を 1 つの container-first runtime として統合します。

## Repository Layout

```text
src/AIKernel.Monolith/       C# / .NET 10 Minimal API runtime
python/aikernel_monolith/    Python CLI wrapper package
tests/                       C# and Python tests
docker/                      Base, Linux, and Windows container stack
samples/                     Read-only sample configuration templates
docs/                        Architecture and operating guides
```

## Run the Monolith

The runtime requires four external mount roots. Inference audit events are appended to the external config mount.

```bash
dotnet run --project src/AIKernel.Monolith/AIKernel.Monolith.csproj -- \
  --config-root=/path/to/config \
  --model-root=/path/to/models \
  --vfs-root=/path/to/vfs \
  --capability-root=/path/to/capabilities
```

Equivalent environment variables are supported:

```bash
AIKERNEL_CONFIG_ROOT=/config
AIKERNEL_MODEL_ROOT=/models
AIKERNEL_VFS_ROOT=/vfs
AIKERNEL_CAPABILITY_ROOT=/capabilities
```

If any root is missing, AIKernel.Monolith fails closed with a clear error message.

## Print Sample Configuration

```bash
dotnet run --project src/AIKernel.Monolith/AIKernel.Monolith.csproj -- --print-sample-config
```

Container images copy the same templates into `/aikernel/samples/`.

## OpenAI-Compatible API

The Minimal API exposes:

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/embeddings`
- `GET /v1/models`
- `GET /health`

### API Access

Set `AIKERNEL_API_KEY` to a private, randomly generated bearer token before
starting Monolith. Use a separate `OPENAI_API_KEY` for the upstream Provider.
Monolith requires `Authorization: Bearer <token>` on requests, including
`/health`. Missing or incorrect tokens receive `401` and a
`WWW-Authenticate: Bearer` challenge. Use HTTPS for remote API clients.

The request budget is shared by authenticated callers on each Monolith
instance. It uses a fixed window and defaults to **60 requests per 60 seconds**.
Set `AIKERNEL_API_REQUEST_LIMIT` and `AIKERNEL_API_WINDOW_SECONDS` to positive
integers to change the budget. Each authenticated HTTP request consumes one
permit, including streaming requests and diagnostic calls. Once the limit is
reached, the host responds immediately with `429` and `Retry-After`. Failed
authentication does not consume a permit. Restart Monolith to apply changes to
the token or budget.

For example, with the token already configured in your shell:

```bash
curl -H "Authorization: Bearer $AIKERNEL_API_KEY" http://127.0.0.1:8080/v1/models
```

### Model Registration

Register a model by copying [the sample](samples/sample-models.json) under the
model mount and setting its upstream model, endpoint, and policy. Restart the
host to load changes to model registrations. The API uses
`id` for exact selection and `model` for the upstream model name. Set the
environment variable named by `apiKeyEnvironmentVariable` in the host or pass
it into the container. Omit the field for an endpoint that needs no key. The
runtime sends API keys only over HTTPS.

The Core CTG decision gate evaluates the operator-supplied `logos`, `ethos`,
and `pathos` votes before each completion or embedding request. Missing or unrecognized
votes deny execution. An Ethos rejection vetoes execution even when the other
two votes approve. Unknown model IDs return 404 and policy denials return 403.

The host appends decisions and execution outcomes to `/config/inference-audit.jsonl`.
The config mount must be writable by the host. Restrict writes to the model
mount to operators and use a separate config mount per instance. Audit events
include the request ID, model and provider IDs, and the outcome.

Set `stream: true` on chat or text completions to receive Server-Sent Events.
The host flushes each text delta as it arrives and sends `[DONE]` after a
successful upstream finish. If a stream fails after output has started, it ends
with an error event and no `[DONE]`.

Register an embedding model with `kind: "embedding"` and its upstream model ID.
`POST /v1/embeddings` accepts one string or an array of non-empty strings and
returns floating-point vectors in input order. A chat registration only accepts
completion requests, and an embedding registration only accepts embedding requests.

## Retrieval-Augmented Generation

RAG uses registered embedding and chat models. The same Monolith bearer token,
request budget, Core policy checks, and inference audit apply.

| Method and route | JSON body | Result |
| --- | --- | --- |
| `GET /v1/rag/indexes` | No body | `indexes` array of index IDs in ordinal order |
| `DELETE /v1/rag/indexes/{indexId}` | No body | Delete the entire index and return `204 No Content` |
| `PUT /v1/rag/indexes/{indexId}` | `{"embeddingModel":"embed","documents":[{"id":"handbook","title":"Handbook","source":"https://example.com/handbook","text":"The office closes at 18:00."}]}` | Index ID, embedding model ID, and document and chunk counts |
| `GET /v1/rag/indexes/{indexId}/documents` | No body | Index ID and document summaries with IDs, titles, sources, and chunk counts |
| `PUT /v1/rag/indexes/{indexId}/documents` | `{"id":"handbook","title":"Updated handbook","text":"The office closes at 19:00."}` | Add or replace one document and return the current index counts |
| `DELETE /v1/rag/indexes/{indexId}/documents?documentId=handbook` | No body. URL-encode the document ID in the query string. | Delete one document and return the current index counts |
| `POST /v1/rag/indexes/{indexId}/search` | `{"query":"When does the office close?","topK":3}` | Ranked hits with citation labels, source metadata, chunk text, and cosine scores |
| `POST /v1/rag/indexes/{indexId}/answer` | `{"model":"chat","question":"When does the office close?","topK":3}` | Completion ID, chat model ID, answer text, and the source chunks supplied to the model |

Use registered model IDs in place of `embed` and `chat`. Document IDs must
be unique within an index. The `title` and `source` fields are optional
metadata. Monolith indexes the supplied `text` directly. Each index holds up
to 64 documents and 256 chunks. Chunks are at most 1,000 UTF-16 code units
long and carry their document ID and one-based chunk number.

Index IDs use up to 64 lowercase ASCII letters, digits, hyphens, or underscores.
Indexes are saved under `rag/` in the writable config mount, including
their text and embeddings. A successful PUT to the index route replaces the
entire named index. Indexes survive a restart with the same config mount.
Rebuild an index when its embedding provider endpoint or upstream model changes.
Deleting an index removes its stored snapshot. Subsequent document, search,
and answer requests for that ID return `404 index_not_found`.

Document changes require an existing index. A document PUT replaces all
chunks and metadata for the matching ID, or adds the document if its ID is
new. Only the supplied document is embedded. DELETE removes all chunks for
the requested ID and returns `404 document_not_found` for an absent document.
Both operations atomically save the updated index snapshot and reuse the
vectors of other documents. An index remains available after its last
document is deleted and returns empty document and search lists. Document
summaries are sorted by document ID in ordinal order.

Search embeds the query using the model recorded in the index. It returns up
to `topK` chunks with positive cosine similarity, defaulting to five and
allowing up to ten. The answer endpoint passes the same ranked chunks to the
chat model as evidence and asks it to cite them using labels such as `[1]`.
The response's `sources` map these labels to the evidence supplied to the
model. An answer request with no matching chunks returns `422 no_context`.

## Container Stack

```bash
docker build --build-context aikernel-providers=../AIKernel.Providers -f docker/Dockerfile.base -t aikernel/monolith:base .
docker build -f docker/Dockerfile.linux -t aikernel/monolith:linux .
docker build -f docker/Dockerfile.windows -t aikernel/monolith:windows .
```

The base image contains the AOT-compiled monolith binary, entrypoint, and sample configuration. Linux and Windows images extend it with platform-specific native ABI layers.
The native build uses the sibling AIKernel.Providers checkout for the AOT-safe ChatOpenAI client. For a checkout without the Provider sources, set `ChatOpenAIProviderPackageVersion` to a package version that includes the streaming and embedding APIs.

Run with explicit mounts:

```bash
docker run --rm -p 8080:8080 \
  -v "$HOME/.aikernel/config:/config" \
  -v "$HOME/.aikernel/models:/models" \
  -v "$HOME/.aikernel/vfs:/vfs" \
  -v "$HOME/.aikernel/capabilities:/capabilities" \
  -e AIKERNEL_CONFIG_ROOT=/config \
  -e AIKERNEL_MODEL_ROOT=/models \
  -e AIKERNEL_VFS_ROOT=/vfs \
  -e AIKERNEL_CAPABILITY_ROOT=/capabilities \
  -e AIKERNEL_API_KEY \
  -e AIKERNEL_API_REQUEST_LIMIT \
  -e AIKERNEL_API_WINDOW_SECONDS \
  -e OPENAI_API_KEY \
  aikernel/monolith:linux
```

To verify a published Windows x64 Native AOT executable against a loopback Provider:

```powershell
python tests/native-aot-stream-embeddings.py <published-exe> <report-directory>
python tests/native-aot-api-access.py <published-exe> <report-directory>
python tests/native-aot-rag.py <published-exe> <report-directory>
```

## Python CLI

The `aikernel-monolith` package provides the `aikernel` command.

```bash
pip install ./python
aikernel init
aikernel up
aikernel status
aikernel stop
```

`aikernel up` forwards `AIKERNEL_API_KEY`, the two request budget settings, and
`OPENAI_API_KEY` from the host. It creates and mounts:

- `~/.aikernel/config`
- `~/.aikernel/models`
- `~/.aikernel/vfs`
- `~/.aikernel/capabilities`

## Determinism Rules

- Capability scanning uses ordinal ordering.
- Model metadata is listed by stable id/provider order.
- Persistent data belongs only in external mounts.
- Missing mount roots fail closed.

## Documentation

- [Architecture](docs/architecture.md)
- [Capability Loader](docs/capability-loader.md)
- [Routing](docs/routing.md)
- [VFS](docs/vfs.md)
- [Python CLI](docs/python-cli.md)
- [Container Stack](docs/container-stack.md)
