# Capability Loader

## Overview

The capability loader scans externally mounted capability artifacts and binds them into deterministic invoker handles.

## Sequence Diagram

```text
CapabilityRoot
  -> ICapabilityMetadataReader
  -> ICapabilityAssemblyLoader
  -> ICapabilityInvokerBinder
  -> CapabilityInvokerHandle
```

## Extension Points

- JSON manifests define stable metadata.
- Managed DLL files are recognized as managed capability artifacts.
- Native `.so` and `.dll` files are recognized as native ABI artifacts.

## Determinism Guarantees

- Files are discovered recursively and sorted ordinally.
- Descriptors are sorted by id and version.
- The loader never creates, rewrites, or deletes user files.

## External Mount Architecture

Mount capabilities at `/capabilities` and pass `--capability-root=/capabilities`.

## Container Stack Architecture

Base images contain the loader only. Platform images may add native ABI libraries.

## Python CLI Workflow

`aikernel up` mounts `~/.aikernel/capabilities` into `/capabilities`.
