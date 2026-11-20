# AIKernel.Monolith Architecture

## Overview

AIKernel.Monolith is the official AIOS distribution host built on the AIKernel SDK. It combines the Semantic Runtime, deterministic routing, model provider registry, capability loader, VFS, and OpenAI-compatible API into one deployable process.

## Sequence Diagram

```text
client
  -> OpenAI-compatible API
  -> SemanticRuntime
  -> DeterministicRouter
  -> ModelProviderRegistry / CapabilityLoader / VFS
  -> response
```

## Extension Points

- Add model metadata under the external model root.
- Add capability manifests or binaries under the external capability root.
- Add VFS content under the external VFS root.
- Extend the route table in the host to change route resolution.

## Determinism Guarantees

- Capability scanning uses ordinal ordering.
- Model listing is stable by model id and provider id.
- The null model provider is deterministic and available before external drivers are mounted.

## External Mount Architecture

Persistent data must live outside the container:

- `/config`
- `/models`
- `/vfs`
- `/capabilities`

The monolith fails closed when any mount is missing.

## Container Stack Architecture

The Linux x64 image packages the AOT host, shared libraries and samples. Windows hosts can use a Linux container engine or run the native `win-x64` executable directly. See the [container guide](container-stack.md).

## Python CLI Workflow

```bash
pip install ./python
aikernel init
aikernel up
```
