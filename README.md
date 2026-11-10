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
  -e OPENAI_API_KEY \
  aikernel/monolith:linux
```

To verify a published Windows x64 Native AOT executable against a loopback Provider:

```powershell
python tests/native-aot-stream-embeddings.py <published-exe> <report-directory>
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

`aikernel up` forwards `OPENAI_API_KEY` from the host and creates and mounts:

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
