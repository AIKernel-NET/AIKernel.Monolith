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

Routing configuration belongs under `/config`.

## Container Stack Architecture

Routing logic is in the base image and does not require native libraries.

## Python CLI Workflow

Use `aikernel init` to materialize sample routing files locally.
