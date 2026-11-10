# Python CLI

## Overview

The `aikernel` Python command is the one-command local AIOS launcher for AIKernel.Monolith.

## Sequence Diagram

```text
aikernel up
  -> detect host OS
  -> select container image
  -> create ~/.aikernel mount roots
  -> docker run
  -> AIKernel.Monolith
```

## Extension Points

- Override image tags in future releases.
- Place custom configs, models, VFS content, and capabilities under `~/.aikernel`.

## Determinism Guarantees

The CLI uses fixed mount paths and fixed container environment variables.

## External Mount Architecture

The CLI creates:

- `~/.aikernel/config`
- `~/.aikernel/models`
- `~/.aikernel/vfs`
- `~/.aikernel/capabilities`

## Container Stack Architecture

Linux hosts use `aikernel/monolith:linux`; Windows hosts use `aikernel/monolith:windows`.

## Python CLI Workflow

Set a private `AIKERNEL_API_KEY` in the host environment before `aikernel up`.
The CLI also forwards optional `AIKERNEL_API_REQUEST_LIMIT` and
`AIKERNEL_API_WINDOW_SECONDS` settings. API requests use the key as a bearer token.

```bash
aikernel init
aikernel up
aikernel status
aikernel stop
```
