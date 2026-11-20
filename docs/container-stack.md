# Run Monolith in a container

Build the base and Linux images with the commands in the [host README](../README.md#container-stack). The base publishes for `linux-x64` and includes the runtime executable, shell entrypoint and sample files. The Linux image uses the `.so` native ABI.

| Environment | How to run Monolith |
| --- | --- |
| Linux x64 Docker engine | Build and run `aikernel/monolith:linux`. |
| Docker Desktop on Windows, Linux containers mode | Use the same Linux x64 image. The legacy `Dockerfile.windows` builds a compatibility tag for this runtime. |
| Docker Desktop on Windows, Windows containers mode | The project does not publish a native Windows container image. Switch Docker Desktop to Linux containers to use the CLI. |
| Native Windows x64 process | Run the published `win-x64` executable with the [four mount paths and loopback address](../README.md#run-the-monolith). |

The [Python CLI](python-cli.md) mounts the host's config, models, VFS and capabilities directories. The host appends inference audit records to the writable config directory. Keep these four directories outside the image so their contents survive a container restart.
