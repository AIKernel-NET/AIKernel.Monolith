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

The runtime requires four external mount roots. It does not create or modify persistent user files inside the container.

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

The initial provider is `NullModelProvider`, a deterministic placeholder used until real model drivers are mounted.

## Container Stack

```bash
docker build -f docker/Dockerfile.base -t aikernel/monolith:base .
docker build -f docker/Dockerfile.linux -t aikernel/monolith:linux .
docker build -f docker/Dockerfile.windows -t aikernel/monolith:windows .
```

The base image contains the AOT-compiled monolith binary, entrypoint, and sample configuration. Linux and Windows images extend it with platform-specific native ABI layers.

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
  aikernel/monolith:linux
```

## Python CLI

The `aikernel-monolith` package provides the `aikernel` command.

```bash
pip install aikernel-monolith==0.1.1
aikernel init
aikernel up
aikernel status
aikernel stop
```

`aikernel up` creates and mounts:

- `~/.aikernel/config`
- `~/.aikernel/models`
- `~/.aikernel/vfs`
- `~/.aikernel/capabilities`

## Determinism Rules

- Capability scanning uses ordinal ordering.
- Model metadata is listed by stable id/provider order.
- User files are never auto-created by the container runtime.
- Persistent data belongs only in external mounts.
- Missing mount roots fail closed.

## Documentation

- [Architecture](docs/architecture.md)
- [Capability Loader](docs/capability-loader.md)
- [Routing](docs/routing.md)
- [VFS](docs/vfs.md)
- [Python CLI](docs/python-cli.md)
- [Container Stack](docs/container-stack.md)
