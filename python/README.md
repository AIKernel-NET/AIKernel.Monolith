# aikernel-monolith

`aikernel-monolith` provides the `aikernel` command for launching the official AIKernel.Monolith container.

```bash
pip install aikernel-monolith==0.1.1
aikernel init
aikernel up
aikernel status
aikernel stop
```

The CLI creates local external mount directories under `~/.aikernel` and mounts them into the container as `/config`, `/models`, `/vfs`, and `/capabilities`.
