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
- Add routing configuration under the external config root.

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

The base image contains the AOT-compiled monolith and sample configuration. Linux and Windows images extend the base image with platform-specific native ABI surfaces.

## Python CLI Workflow

```bash
pip install aikernel-monolith==0.1.1
aikernel init
aikernel up
```
