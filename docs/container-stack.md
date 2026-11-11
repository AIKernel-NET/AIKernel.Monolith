# Container Stack

## Overview

AIKernel.Monolith uses a base image plus platform-specific images.

## Sequence Diagram

```text
Dockerfile.base
  -> aikernel/monolith:base
  -> Dockerfile.linux / Dockerfile.windows
  -> runtime image
```

## Extension Points

- Add Linux `.so` libraries in the Linux image.
- Add Windows `.dll` libraries in the Windows image.
- Keep persistent state outside all images.

## Determinism Guarantees

The base image contains only the AOT binary, entrypoint, and read-only samples.

## External Mount Architecture

Runtime data must be mounted externally. Missing mounts cause startup failure. The runtime writes inference audit events to the external config mount.

## Container Stack Architecture

- `Dockerfile.base`: AOT binary and samples. Pass the sibling AIKernel.Providers and AIKernel.Core checkouts as the `aikernel-providers` and `aikernel-core` build contexts.
- `Dockerfile.linux`: Linux native ABI layer.
- `Dockerfile.windows`: Windows native ABI layer.

## Python CLI Workflow

The Python CLI selects the platform image and mounts host directories into the container.
