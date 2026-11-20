# Routing

## Overview

Routing decides whether an input should target a model, tool, or capability.

## Sequence Diagram

```text
RouteRequest
  -> SemanticRouteResolver
  -> ModelRouteBinder
  -> ToolRouteBinder
  -> CapabilityRouteBinder
  -> RouteDecision
```

## Extension Points

- Replace `IRouter` with a custom deterministic router.
- Extend `RouteTable` from external routing configuration.
- Bind tools and capabilities from mounted manifests.

## Determinism Guarantees

Explicit model requests take precedence over tool requests, which take precedence over generic capability routing.

## External Mount Architecture

The host uses its built-in route table. The sample routing YAML illustrates the structure for a custom configuration loader.

## Container Stack Architecture

Routing logic is in the base image and does not require native libraries.

## Python CLI Workflow

After building the image, run `aikernel init` to create the four mount directories. It saves the output of `--print-sample-config` to `~/.aikernel/config/sample-configs.txt`. To register a model, copy a JSON registration to `~/.aikernel/models/` and restart the host. [Registration example](../samples/cosmos-edge.json)
