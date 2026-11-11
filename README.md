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

### Local paged CUDA inference

Register a `providers.paged-llama` chat model under the model mount. This
provider accepts the legacy llama2.c little-endian FP32 checkpoint format
and the matching binary tokenizer. It runs the Transformer in the host with
CUDA matrix-vector multiplication over resident weight blocks. A 64-bit
Windows or Linux host with an NVIDIA driver provides the device backend.

```json
{
  "id": "local-stories",
  "provider": "providers.paged-llama",
  "kind": "chat",
  "local": {
    "checkpoint": "stories260K.bin",
    "tokenizer": "tok512.bin",
    "device": 0,
    "blockSize": 4096,
    "ramBlocks": 4,
    "vramBlocks": 2,
    "contextTokens": 64,
    "maxNewTokens": 8
  },
  "policy": { "logos": "approve", "ethos": "approve", "pathos": "approve" }
}
```

Asset paths are relative to the model mount. Model weights load on the first
authorized request and stay cached until the host exits. Place the config
mount on a private writable SSD directory for the swap files under
`model-swap/`. Allow disk capacity for the model's nonresident blocks.
`blockSize` is a power of two from 4 KiB to 32 MiB and must fit a matrix row.
The defaults are 2 MiB per block, 32 RAM blocks, 32 VRAM blocks, CUDA device
ordinal 0, a 512-token context, and at most 32 generated tokens. The weight
budgets cover resident model blocks. The runtime also uses host KV cache
and staging memory, a CUDA context, and device input/output buffers.

The local model works with `/v1/completions` and `/v1/chat/completions`,
including SSE. A single user message is used as a raw prompt. Multiple
messages are formatted as role-prefixed lines. Tokens are selected greedily.
The audit log records resident weight counts, swap reads, VRAM page-ins,
and CUDA matrix blocks processed for each successful request.

To exercise the published Windows x64 executable with the
[TinyStories 260K model and tokenizer](https://huggingface.co/karpathy/tinyllamas/tree/0bd21da/stories260K),
put `stories260K.bin` and `tok512.bin` in the model asset directory and run:

```powershell
python tests/native-aot-paged-llama.py <published-exe> <model-assets-directory> <report-directory> 0
```

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
| `POST /v1/rag/indexes/{indexId}/search` | `{"query":"When does the office close?","topK":3}` | Ranked hits with citation labels, source metadata, chunk text, and retrieval scores |
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

Search and answer accept `"searchMode": "hybrid"` to combine vector similarity
and exact keyword matches. The default `"vector"` mode ranks by positive
cosine similarity. Hybrid mode measures coverage of the distinct query terms
in each chunk's text, document title, and document ID. It compares whole
tokens without regard to case and keeps internal hyphens, underscores, periods,
and slashes in identifiers such as `ZX-42`.

The hybrid score combines positive cosine similarity with twice the keyword
coverage and divides by three. If the query contains a term with both letters
and digits, the upper half of the score range is reserved for chunks with an
exact match to such a term. This lets an exact model number rank above a
semantically similar prefix match. Chunks can qualify through either signal.
Hybrid mode uses the stored text and embeddings and applies the same filters
before ranking. It works with both buffered and streaming answers.

Both search and answer accept an optional `filter` object. For example, a
filtered search body can select a document ID and its registered source.

```json
{
  "query": "When does the office close?",
  "topK": 3,
  "filter": {
    "documentIds": ["handbook"],
    "sources": ["https://example.com/handbook"]
  }
}
```

The filter uses exact, case-sensitive matches against the stored document ID
and source string. Values within either list are alternatives. If both lists
are supplied, a document must match both. Each supplied list must have 1 to
64 entries. Document IDs use the same validation as indexed IDs, and source
strings may have up to 2,048 UTF-16 code units. Omitted or null lists impose
no restriction. Filtering happens before similarity ranking and `topK`.
When no document matches, search returns an empty `hits` array and answer
returns `422 no_context`.

Set `"stream": true` in the answer request body to receive Server-Sent
Events. The first data event contains `id`, `model`, and the same `sources`
that a buffered answer would use. It is followed by chat completion chunks
with the same ID. A successful stream sends the final `finish_reason`
and `[DONE]`. Errors before the first event use the usual HTTP status
and JSON error body. Errors after output has started send an error event
and end without `[DONE]`.

## Container Stack

```bash
docker build --build-context aikernel-providers=../AIKernel.Providers --build-context aikernel-core=../AIKernel.Core -f docker/Dockerfile.base -t aikernel/monolith:base .
docker build -f docker/Dockerfile.linux -t aikernel/monolith:linux .
docker build -f docker/Dockerfile.windows -t aikernel/monolith:windows .
```

The base image contains the AOT-compiled monolith binary, entrypoint, and sample configuration. Linux and Windows images extend it with platform-specific native ABI layers.
The native build uses the sibling AIKernel.Core and AIKernel.Providers checkouts.
Point `AIKernelKernelProject` and `ChatOpenAIProviderProject` at alternate source projects as needed.
Package-based builds can set `AIKernelKernelPackageVersion` and `ChatOpenAIProviderPackageVersion`
to versions with paged CUDA support and streaming/embedding support respectively.

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
