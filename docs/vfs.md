# Virtual File System

## Overview

The monolith exposes a read-only VFS abstraction over external content.

## Sequence Diagram

```text
request path
  -> IVirtualFileSystem
  -> LocalVirtualFileSystem
  -> external /vfs mount
  -> content
```

## Extension Points

- `LocalVirtualFileSystem` reads from the external root.
- `MemoryVirtualFileSystem` supports deterministic tests and ephemeral mounts.

## Determinism Guarantees

- Directory listings are sorted ordinally.
- Path traversal outside the configured root is rejected.
- The monolith does not write into the VFS root.

## External Mount Architecture

Mount VFS content at `/vfs` and pass `--vfs-root=/vfs`.

## Container Stack Architecture

VFS support is part of the base image.

## Python CLI Workflow

`aikernel up` mounts `~/.aikernel/vfs` into `/vfs`.
