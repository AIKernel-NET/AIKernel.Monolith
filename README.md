# AIKernel.Monolith

AIKernel.Monolith is the official AIOS distribution host built on the AIKernel SDK. It packages the Semantic Runtime, deterministic routing, external capability loading, model provider registry, virtual file system access, and an OpenAI-compatible API into one container-first runtime.

AIKernel.Monolith は、AIKernel SDK 上に構築される公式 AIOS ディストリビューションホストです。Semantic Runtime、決定論的 routing、外部 Capability loading、model provider registry、VFS access、OpenAI 互換 API を 1 つの container-first runtime として統合します。

## Repository Layout

```text
src/AIKernel.Monolith/       C# / .NET 10 Minimal API runtime
python/aikernel_monolith/    Python CLI wrapper package
tests/                       C# and Python tests
docker/                      Base, Linux, and Windows container stack
samples/                     Read-only sample configuration templates
docs/                        Architecture and operating guides
```

## Run the Monolith

The runtime requires four external mount roots. Inference audit events are appended to the external config mount.

```bash
dotnet run --project src/AIKernel.Monolith/AIKernel.Monolith.csproj -- \
  --config-root=/path/to/config \
  --model-root=/path/to/models \
  --vfs-root=/path/to/vfs \
  --capability-root=/path/to/capabilities
```

Equivalent environment variables are supported:

```bash
AIKERNEL_CONFIG_ROOT=/config
AIKERNEL_MODEL_ROOT=/models
AIKERNEL_VFS_ROOT=/vfs
AIKERNEL_CAPABILITY_ROOT=/capabilities
```

If any root is missing, AIKernel.Monolith fails closed with a clear error message.

## Print Sample Configuration

```bash
dotnet run --project src/AIKernel.Monolith/AIKernel.Monolith.csproj -- --print-sample-config
```

Container images copy the same templates into `/aikernel/samples/`.

## OpenAI-Compatible API

The Minimal API exposes:

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/embeddings`
- `GET /v1/models`
- `GET /health`

### API Access

Set `AIKERNEL_API_KEY` to a private, randomly generated bearer token before
starting Monolith. Use a separate `OPENAI_API_KEY` for the upstream Provider.
Monolith requires `Authorization: Bearer <token>` on requests, including
`/health`. Missing or incorrect tokens receive `401` and a
`WWW-Authenticate: Bearer` challenge. Use HTTPS for remote API clients.

The request budget is shared by authenticated callers on each Monolith
instance. It uses a fixed window and defaults to **60 requests per 60 seconds**.
Set `AIKERNEL_API_REQUEST_LIMIT` and `AIKERNEL_API_WINDOW_SECONDS` to positive
integers to change the budget. Each authenticated HTTP request consumes one
permit, including streaming requests and diagnostic calls. Once the limit is
reached, the host responds immediately with `429` and `Retry-After`. Failed
authentication does not consume a permit. Restart Monolith to apply changes to
the token or budget.

For example, with the token already configured in your shell:

```bash
curl -H "Authorization: Bearer $AIKERNEL_API_KEY" http://127.0.0.1:8080/v1/models
```

### Model Registration

Register a model by copying [the sample](samples/sample-models.json) under the
model mount and setting its upstream model, endpoint, and policy. Restart the
host to load changes to model registrations. The API uses
`id` for exact selection and `model` for the upstream model name. Set the
environment variable named by `apiKeyEnvironmentVariable` in the host or pass
it into the container. Omit the field for an endpoint that needs no key. The
runtime sends API keys only over HTTPS.

The Core CTG decision gate evaluates the operator-supplied `logos`, `ethos`,
and `pathos` votes before each completion or embedding request. Missing or unrecognized
votes deny execution. An Ethos rejection vetoes execution even when the other
two votes approve. Unknown model IDs return 404 and policy denials return 403.

The host appends decisions and execution outcomes to `/config/inference-audit.jsonl`.
The config mount must be writable by the host. Restrict writes to the model
mount to operators and use a separate config mount per instance. Audit events
include the request ID, model and provider IDs, and the outcome.

Set `stream: true` on chat or text completions to receive Server-Sent Events.
The host flushes each text delta as it arrives and sends `[DONE]` after a
successful upstream finish. If a stream fails after output has started, it ends
with an error event and no `[DONE]`.

Register an embedding model with `kind: "embedding"` and its upstream model ID.
`POST /v1/embeddings` accepts one string or an array of non-empty strings and
returns floating-point vectors in input order. A chat registration only accepts
completion requests, and an embedding registration only accepts embedding requests.

### Local paged CUDA inference

For CPU inference, place the checkpoint and a registration under the model mount. The `providers.gguf-cpu` provider accepts `qwen3` (Bonsai), `gemma4-e2b`, and `gpt-oss-20b`. It accepts one user message with an optional preceding system or developer message. The streaming endpoint emits the completed answer as one content chunk. A single active model serves requests in sequence. Gemma and GPT-OSS use AIKernel's CPU weight paging. For GPT-OSS-20B, reserve 12.1 GB on disk for the MXFP4 GGUF. The host uses a 5 GB weight cache and checks process residency against a 7 GB budget. Evicted weights can be read again from the original GGUF.

On 2026-11-19 a CPU-only Linux host admitted a Bonsai 1.7B request with an empty GPU inventory and returned `4` for a local arithmetic question through `/v1/chat/completions`. Peak process RSS was 405,614,592 bytes with two CPUs in its affinity mask. The Windows x64 NativeAOT executable returned the same answer through authenticated HTTP.

```json
{
  "id": "bonsai-local", "provider": "providers.gguf-cpu", "model": "qwen3",
  "gguf": { "checkpoint": "Bonsai-1.7B-Q1_0.gguf", "cpuThreads": 4, "contextTokens": 512, "maxNewTokens": 96 },
  "policy": { "logos": "approve", "ethos": "approve", "pathos": "abstain" }
}
```

Use `GET /v1/models/admission?model=bonsai-local` to check the policy and checkpoint availability. Tensor validation takes place on first generation. The request and audit routes are shared with the other providers.

For a Cosmos 3 conversation, use `provider: "providers.cosmos-cpu"` and
`model: "cosmos3-edge"` or `"cosmos3-nano"`. In the registration's `cosmos`
object, set `directory` to a checkpoint directory inside the model mount and
`manifestSha256` to the trusted import manifest digest. Configure the same
governance policy as for other local models. The default `weightAccess` is
`"paged"`. A workstation with sufficient RAM can use `"mapped"`. Both modes
verify the checkpoint on load. Admission checks metadata and shard presence.

To reproduce the Server profile, add this object inside `cosmos`:

```json
"memory": {
  "maximumResidentBytes": 7000000000,
  "weightCacheBytes": 3221225472,
  "kvCacheBytes": 128000000
}
```

Each field is optional in paged mode. The defaults are 7,000,000,000 bytes
for process RSS, 5,000,000,000 for weights, 128,000,000 for KV and
1,000,000,000 for `runtimeReserveBytes`. The runtime adjusts cache capacity
within the cooperative RSS budget. Use an OS limit when a hard ceiling is
required. To exercise encrypted KV spill, set `kvCacheBytes` to 8,388,608.
When the text weights exceed the available cache, Cosmos processes the prompt
in tiles of up to 32 tokens. Each projection visits its weight pages once per
tile. Attention consumes tokens in order and restores earlier tiles' KV data.
The runtime reserves scratch space first and reduces the tile size under memory
pressure. Each generated token still requires a separate decode step.

The private inference audit records `model_load_ms` (including
`checkpoint_verify_ms`) and `generation_ms` (including `prefill_ms` and
`decode_ms`). Model load and verification are zero on reuse. Text prefill
starts after tokenization and vision encoding. `weight_backing_reads` counts
logical weight page misses during generation, `prefill_weight_page_misses`
counts the prefill subset, and `kv_swap_reads` counts reads from KV swap.
On Windows, cached weight bytes cover the encoded contents of open read-only
file views. Use the process peak working set to check the resident budget.

To reproduce the Windows first-request comparison, use Python 3.11+ with the
two published Native AOT executables and the original imported Edge directory.
The probe creates temporary hard links next to the import, waits for two
15-second CPU samples at or below 10%, and starts a fresh host for each run.
It uses four CPU workers, the 32-token arithmetic case, a 1 GiB weight cache
and a 7,000,000,000-byte process budget.

```powershell
py tests/cosmos-edge-cold-start.py --baseline old/AIKernel.Monolith.exe --candidate new/AIKernel.Monolith.exe --edge D:/models/Edge --output edge-comparison.jsonl --token-validator new-tools/AIKernel.Core.Tests.exe
```

The four runs alternate in A–B–B–A order and preserve the system file cache.
Each JSON line records the HTTP response time, audit counters, peak working
set, and process page faults. The probe checks the answer, finish reason,
input token count, and the policy and execution audit records. The optional
`--token-validator` runs the app published from Core's
`tests/AIKernel.Inference.Optimization.Validation` project after the timed
requests and compares its generated IDs with the original `[1052, 11]`. Missing timing
fields in an older executable appear as `null`. To identify physical storage
waits, capture WPR's General, Disk I/O and File I/O profiles in a separate
run. In WPA, correlate hard faults and disk activity with the shard paths,
request UTC interval and host PID from the JSON. Process page faults include
soft faults. [Windows Performance Recorder profiles](https://learn.microsoft.com/en-us/windows-hardware/test/wpt/recording-for-resource-based-analysis)

For a diagnostic run, set `"profilePrefill": true` in the model's `cosmos`
registration, or pass `--profile-prefill candidate` to the probe. Use
`--profile-prefill both` when both executables support it. The private audit
includes the UTC prefill start, the actual tile length and the BF16 dot kernel
selected by that executable. On x64 an AVX2-capable CPU can still run
`scalar-fp64` in a portable Native AOT build. To test the existing AVX2 path,
publish a second x64 executable from the same commit with
`-p:IlcInstructionSet=avx2` to a separate output directory and run it on
compatible hardware. Keep the portable executable for older CPUs.

| Prefill field | What the interval measures |
| --- | --- |
| `prefill_projection_page_setup_ms` | Time from entering the projection page cache to receiving the span, including any mapping and prefetch call. `prefill_projection_page_visits` includes cache hits. |
| `prefill_projection_rows_ms` | Wall time computing projection rows, including worker scheduling and faults incurred while touching the weights. |
| `prefill_attention_ms` | Head normalization, rotary embeddings, KV handling and causal attention. |
| `prefill_process_cpu_ms` | User and kernel CPU time of all host threads. Divide by `prefill_ms` for average cores used by the process over this interval. |

The first three wall intervals are disjoint. The residual includes embedding,
normalization outside attention, activations, and other coordination. To
investigate SSD stalls in WPA, select the host PID and the interval beginning
at `prefill_start_utc` and lasting `prefill_ms`. Inspect hard faults, file paths,
and disk service/wait times together. The probe's process fault counter spans
the full HTTP request and includes soft faults.

For a separate single-request trace, start
`wpr -start tests/cosmos-prefill.wprp!CosmosPrefill -filemode` before the
request and run `wpr -stop cosmos-prefill.etl` after it. The profile records
all page-fault types and the detailed hard-fault and disk I/O events. In the
same PID and prefill interval, count faults by type and use the detailed
hard-fault events to inspect waits. Check for lost ETW events before reporting
a hard-fault percentage. Keep traced timings separate from the A–B–B–A series.

Send alternating user and assistant messages to `/v1/chat/completions`,
optionally preceded by one system or developer message. User content accepts
plain text or an array of text, `image_rgb24`, and `video_rgb24` parts.
An image carries integer width and height (multiples of 32) and base64 RGB
in row-major order. A video carries an ordered frames array whose entries
have the same fields plus seconds. Include earlier attachments when sending a
follow-up. The host currently emits the completed reply in one content chunk.

Register a `providers.paged-llama` chat model under the model mount. This
provider accepts Llama GGUF v2/v3, Llama SafeTensors with a config.json, and
legacy llama2.c FP32 checkpoints. GGUF and SafeTensors tensors are mapped
read-only and faulted into the compute tier one block at a time. GGUF
supports F32, F16, BF16, Q4_0, and MXFP4 tensor types. SafeTensors supports
F32, F16, and BF16. The mapped formats support unscaled rotary embeddings.

The Native C ABI runs GEMV, normalization, RoPE, SwiGLU, Softmax, and paged
Attention on CUDA or the host. A CUDA context adds VRAM and page-locked RAM.
A host without an available CUDA device uses RAM and SSD with the same
inference pipeline. The CUDA backend needs the installed NVIDIA driver.

```json
{
  "id": "local-stories",
  "provider": "providers.paged-llama",
  "kind": "chat",
  "local": {
    "checkpoint": "stories260K.bin",
    "tokenizer": "tok512.bin",
    "device": 0,
    "blockSize": 4096,
    "ramBlocks": 4,
    "vramBlocks": 2,
    "contextTokens": 64,
    "maxNewTokens": 8
  },
  "policy": { "logos": "approve", "ethos": "approve", "pathos": "approve" }
}
```

Asset paths are relative to the model mount. GGUF can use embedded Llama
vocabulary and scores. Set tokenizer to a matching legacy binary, Llama
SentencePiece BPE model, or SentencePiece-style BPE tokenizer.json to load
it separately. For SafeTensors set config to the matching config.json.
Model state initializes on the first authorized request and stays cached
until the host exits. Place the config
mount on a private writable SSD directory for the swap files under
`model-swap/`. Allow disk capacity for the model's nonresident blocks.
`blockSize` is a power of two from 4 KiB to 32 MiB and must fit a matrix row.
The defaults are 2 MiB per block, 32 RAM blocks, 32 VRAM blocks, CUDA device
ordinal 0, a 512-token context, and at most 32 generated tokens. The weight
budgets cover resident model blocks. The per-request KV address space uses
64 KiB blocks (or larger if a row requires it), with separate kvRamBlocks
and kvVramBlocks settings. Defaults are 32 RAM and 16 VRAM blocks for KV.
The runtime also uses staging memory, a CUDA context, and device scratch
buffers.

The local model works with `/v1/completions` and `/v1/chat/completions`,
including SSE. A single user message is used as a raw prompt. Multiple
messages are formatted as role-prefixed lines. Temperature defaults to zero
for greedy selection. Set temperature above zero and topP in (0, 1] for
nucleus sampling.
The audit log records resident weight counts, swap reads, VRAM page-ins,
and matrix blocks processed for each successful request. Audit fields also
include KV swap reads, process-wide Gen 2 collection deltas, allocated bytes,
and the maximum observed generation step time.

To exercise the published Windows x64 executable with the
[TinyStories 260K model and tokenizer](https://huggingface.co/karpathy/tinyllamas/tree/0bd21da/stories260K),
put `stories260K.bin` and `tok512.bin` in the model asset directory and run:

```powershell
python tests/native-aot-paged-llama.py <published-exe> <model-assets-directory> <report-directory> 0
python tests/native-aot-standalone.py <published-exe> <model-assets-directory> <report-directory>
```

Pass `--cpu-only` to the standalone test to check RAM/SSD generation in a
Linux container without the NVIDIA driver. Pass `--oversized-device <ordinal>`
to generate a sparse FP32 GGUF larger than that device and verify real CUDA
page-ins during generation.

### Local inference governance

A registered local model can set these optional policy fields.

| Field | Values |
| --- | --- |
| `priority` | `backgroundBatch`, `normal`, `high`, `emergency` |
| `maxPcieSwapBytes` | Maximum estimated PCIe transfer bytes for one request |
| `maxTemperatureCelsius` | Maximum GPU temperature, with zero disabling the limit |
| `thermalMigrationDevices` | Ordered alternative CUDA ordinals. Omit to consider every visible CUDA device, or use `[]` to disallow migration. |
| `thermalDeviceLimits` | Per-device temperature limits keyed by CUDA ordinal. Devices without an entry use the model's `maxTemperatureCelsius`. |
| `batchPowerLimitPercent` | Batch throttle threshold as a percentage of the NVML power limit |
| `encryptSwap` | Use authenticated encryption for SSD swap |
| `zeroWipe` | Clear released RAM, device blocks, and logical swap ranges |

Priority comes from server-side model registration. Registrations on the
same CUDA device share a scheduler. A higher priority request can preempt
a lower priority request at its next transformer layer checkpoint.
The resource gate runs before the first forward pass. When required health
telemetry is unavailable at initialization, the model uses CPU/RAM/SSD.
A device above the temperature limit at admission is rejected unless the
previous request on the model failed to find a thermal migration target.
In that case, the next request can move the model to an eligible CUDA device
before allocating its own KV cache and starting prefill. If a device
crosses the limit after a layer, the request drains its resident pages to
backing storage, releases its device lease, and continues with the same KV
cache and activations on an eligible CUDA device. The destination must pass
the health and resource checks under its lease. A request visits each
physical GPU at most once and returns `device_thermal_migration_unavailable`
if no alternative can run it. Batch power limits still reject work.

For `device_temperature_exceeded` and `device_thermal_migration_unavailable`,
the API returns HTTP 503 with
`Retry-After: 5` and `error.retry_after_seconds: 5` in the JSON body. If
streaming has already started, the SSE error event carries the same JSON
field. The value is a polling interval. GPU eligibility is checked again
when the client retries.

Use `GET /v1/models/admission?model=<model-id>` with the Monolith bearer token
to check a model without starting inference. A registered model returns
HTTP 200 with `admissible`, a `reason` when blocked, and `retry_after_seconds`
for thermal denials and CUDA discovery failures. An unregistered model returns
HTTP 404. The response is a fresh snapshot of the model policy and route.
Prompt-dependent resource admission and GPU lease acquisition happen when an
inference request arrives.

For local models, `cuda_topology_state` is `retry_wait` between failed attempts,
`probing` while enumeration is running, `ready` once CUDA devices are known, or
`no_devices` when no CUDA driver or device is available. During `retry_wait`
and `probing`, local inference returns HTTP 503 with `cuda_device_unavailable`
and `Retry-After: 5`. Five seconds is the suggested polling interval.

While waiting, `cuda_next_retry_in_seconds` counts down to the next attempt,
rounded up to whole seconds. It is omitted while the probe is running.
`cuda_last_failure_reason` describes the latest failed attempt and remains
visible during the next probe. Its values are `probe_nonzero_exit`,
`probe_timeout`, `probe_start_failed`, `probe_failed` (communication or
process error), and `probe_invalid_inventory`. A successful probe clears both
fields, including when it confirms there are no CUDA devices.

For a local model, `device` is the current CUDA ordinal or the configured
ordinal before the first load. `admission_device` is the GPU the health gate
would choose, including an eligible recovery target after a failed migration.
Both fields are null on a CPU route. The `devices` array reports each visible
GPU’s PCI bus ID, NVML temperature, execution limit, and migration target
limit. A temperature limit of zero disables that temperature check. A null
temperature means the sample is unavailable. `migration_target_eligible`
checks the device allowlist and health limits. The actual migration also
checks available resources and the devices visited by the request. At startup,
a short-lived process resolves the CUDA PCI mapping without creating a context.
It exits before Monolith begins listening. Polls reuse the mapping and read
fresh NVML samples. Use a CUDA driver library that exports
`ak_get_device_pci_bus_id`.

This `local` registration fragment configures separate limits for two GPUs.

```json
"thermalDeviceLimits": [
  { "device": 0, "maxTemperatureCelsius": 68, "maxMigrationTargetTemperatureCelsius": 62 },
  { "device": 1, "maxTemperatureCelsius": 77, "maxMigrationTargetTemperatureCelsius": 75 }
]
```

Each entry requires a unique CUDA ordinal and a running limit from 1 to 120 °C.
The migration target limit defaults to the running limit and must be between
1 °C and that limit. A running GPU can reach its running limit before migration
is triggered. A candidate is eligible at or below its migration target limit,
including when a later request returns to a GPU used before. The model's
batch power limit also applies to devices with temperature overrides.
The inference audit records the last candidate rejected by the temperature
check, along with the PCI identifier, NVML temperature, and configured limit.

Run the GPU governance test with the same TinyStories fixture directory
used by the standalone test.

```powershell
python tests/native-aot-governance.py <published-exe> <model-assets-directory> <report-directory> --expect-nvml
```

Run the Windows admission VRAM regression with the same fixtures, a regular
Native AOT publication, an RTX 4000 and T1000, NVML, and `nvidia-smi` on
`PATH`. The test checks that the CUDA probe exits before Monolith listens
and that the Windows dedicated GPU memory counter stays at 0 B before,
during, and after the initial admission queries. It also runs inference on
both adapters to verify that the counter detects allocations.

```powershell
python tests/native-aot-admission-vram.py <published-exe> <model-assets-directory> <report-directory>
```

To also exercise an abnormal probe exit and a probe timeout, build the fixture
from an x64 Native Tools Command Prompt and pass the DLL to the regression.
The fixture uses the real CUDA driver and waits after writing the PCI mapping.
The test checks that Monolith starts with GPU admission unavailable and both
the probe process and its VRAM are released. It then allows a background
retry to finish and checks that admission and NVML temperatures recover in
the same process, with zero dedicated VRAM during queries.

The shutdown case sends Ctrl+C to Monolith while a retry is using both GPUs.
It checks the parent exit code, the retry process exit, and zero dedicated
VRAM after shutdown.

The forced shutdown case terminates Monolith while a retry is using both GPUs.
On Windows, each probe joins a Job Object before it initializes CUDA. The job
ends the probe when Monolith exits. The test verifies that the retry exits
and that the dedicated VRAM of both processes returns to 0 B.

On Linux, run the SIGKILL regression with an RTX 4000 and T1000, a Linux
Native AOT publication, Python with pidfd support, and NVIDIA per-process
VRAM accounting. Run the test in the PID namespace reported by NVML.
It verifies the retry owns allocations on both GPUs before sending SIGKILL
to Monolith and checks process exit and VRAM release afterward.

```sh
cc -std=c11 -shared -fPIC tests/fixtures/cuda-probe-fault-linux.c -o libcuda-probe-fault.so -ldl
python3 tests/native-aot-admission-vram-linux.py <published-exe> <model-assets-directory> <report-directory> ./libcuda-probe-fault.so
```

After a failed startup probe, Monolith waits five seconds between attempts.
Each attempt has a 15-second timeout. Until a complete CUDA inventory is
available, local inference returns HTTP 503 with `cuda_device_unavailable`.

```bat
cl /nologo /LD /MT tests/fixtures/cuda-probe-fault.c /link /OUT:cuda-probe-fault.dll
python tests/native-aot-admission-vram.py <published-exe> <model-assets-directory> <report-directory> --probe-fault-library cuda-probe-fault.dll
```

The Native AOT console project in `tests/AIKernel.Monolith.ThermalTests`
uses two physical CUDA devices and live NVML data. It injects over-limit
temperature samples during generation to check migration and refusal
without relying on a change in room temperature or GPU load. Run it with
the TinyStories checkpoint and tokenizer paths, followed by the source
and destination CUDA ordinals.

To exercise the same registered model twice using actual temperature changes,
run the Windows Native AOT regression with Python 3.11 or later, two NVIDIA GPUs
that support CUDA compute capability 7.5 or later, NVML, and `nvidia-smi` on
`PATH`. Use an OpenLLaMA 3B v2 Q4_0 GGUF with its tokenizer embedded and a new
or empty report directory on the SSD used for the swap files.

```powershell
python tests/native-aot-thermal-cancel.py <published-exe> <openllama-gguf> <report-directory> --devices 0 1 --limit 79 --return-margin 0
```

Use `--device-limits 68 62 77 75` to exercise the example above. The four
values are the source running and migration target limits followed by the
target running and migration target limits. The test checks the limits and
accepted target temperatures recorded in the audit in both directions.

The test uses a bounded CUDA workload to reach the threshold and waits for the
source GPU to cool before heating the target for the return migration. Set the
threshold so that both GPUs can cross it under load and cool back to it between
requests. Use `--cooldown-timeout` to allow a longer cooldown and
`--return-margin` to set how far below its migration target limit the source must cool
before the test starts heating the target. Each cycle queues a successor,
disconnects the active request after SSD KV reads on the target and before the
first token, and checks that CUDA
released the KV allocations while retaining model weight allocations. The first
successor must return to the source GPU before the second cycle can start under
the same model ID and process. Both successors must match an isolated run using
the same executable and checkpoint. The report directory contains `result.json`,
inference audits, and the temperature and server logs.

Use the same runner to test rejection and recovery at the lower migration
target limit. With these settings, device 1 starts the request. The test heats
device 1 above 77 °C while keeping device 0 in the rejection band above 62 °C
and at or below 68 °C.

```powershell
python tests/native-aot-thermal-cancel.py <published-exe> <openllama-gguf> <report-directory> --scenario reuse-rejection --devices 0 1 --device-limits 68 62 77 75
```

The test checks the NVML temperature used to reject device 0 and verifies
that the first request releases its KV allocations and SSD swap. While device 1
is still above 77 °C and device 0 remains in the rejection band, the test sends
three concurrent requests to the same model ID. Each must receive HTTP 503
during admission, before the resource gate and KV allocation. After device 0
cools to 62 °C or below, a subsequent request to that model ID must migrate and
finish in the same process. Its output, finish reason, and prefill and forward
counts must match an isolated run on device 0.

## Retrieval-Augmented Generation

RAG uses registered embedding and chat models. The same Monolith bearer token,
request budget, Core policy checks, and inference audit apply.

| Method and route | JSON body | Result |
| --- | --- | --- |
| `GET /v1/rag/indexes` | No body | `indexes` array of index IDs in ordinal order |
| `DELETE /v1/rag/indexes/{indexId}` | No body | Delete the entire index and return `204 No Content` |
| `PUT /v1/rag/indexes/{indexId}` | `{"embeddingModel":"embed","documents":[{"id":"handbook","title":"Handbook","source":"https://example.com/handbook","text":"The office closes at 18:00."}]}` | Index ID, embedding model ID, and document and chunk counts |
| `GET /v1/rag/indexes/{indexId}/documents` | No body | Index ID and document summaries with IDs, titles, sources, and chunk counts |
| `PUT /v1/rag/indexes/{indexId}/documents` | `{"id":"handbook","title":"Updated handbook","text":"The office closes at 19:00."}` | Add or replace one document and return the current index counts |
| `DELETE /v1/rag/indexes/{indexId}/documents?documentId=handbook` | No body. URL-encode the document ID in the query string. | Delete one document and return the current index counts |
| `POST /v1/rag/indexes/{indexId}/search` | `{"query":"When does the office close?","topK":3}` | Ranked hits with citation labels, source metadata, chunk text, and retrieval scores |
| `POST /v1/rag/indexes/{indexId}/answer` | `{"model":"chat","question":"When does the office close?","topK":3}` | Completion ID, chat model ID, answer text, and the source chunks supplied to the model |

Use registered model IDs in place of `embed` and `chat`. Document IDs must
be unique within an index. The `title` and `source` fields are optional
metadata. Monolith indexes the supplied `text` directly. Each index holds up
to 64 documents and 256 chunks. Chunks are at most 1,000 UTF-16 code units
long and carry their document ID and one-based chunk number.

Index IDs use up to 64 lowercase ASCII letters, digits, hyphens, or underscores.
Indexes are saved under `rag/` in the writable config mount, including
their text and embeddings. A successful PUT to the index route replaces the
entire named index. Indexes survive a restart with the same config mount.
Rebuild an index when its embedding provider endpoint or upstream model changes.
Deleting an index removes its stored snapshot. Subsequent document, search,
and answer requests for that ID return `404 index_not_found`.

Document changes require an existing index. A document PUT replaces all
chunks and metadata for the matching ID, or adds the document if its ID is
new. Only the supplied document is embedded. DELETE removes all chunks for
the requested ID and returns `404 document_not_found` for an absent document.
Both operations atomically save the updated index snapshot and reuse the
vectors of other documents. An index remains available after its last
document is deleted and returns empty document and search lists. Document
summaries are sorted by document ID in ordinal order.

Search embeds the query using the model recorded in the index. It returns up
to `topK` chunks with positive cosine similarity, defaulting to five and
allowing up to ten. The answer endpoint passes the same ranked chunks to the
chat model as evidence and asks it to cite them using labels such as `[1]`.
The response's `sources` map these labels to the evidence supplied to the
model. An answer request with no matching chunks returns `422 no_context`.

Search and answer accept `"searchMode": "hybrid"` to combine vector similarity
and exact keyword matches. The default `"vector"` mode ranks by positive
cosine similarity. Hybrid mode measures coverage of the distinct query terms
in each chunk's text, document title, and document ID. It compares whole
tokens without regard to case and keeps internal hyphens, underscores, periods,
and slashes in identifiers such as `ZX-42`.

The hybrid score combines positive cosine similarity with twice the keyword
coverage and divides by three. If the query contains a term with both letters
and digits, the upper half of the score range is reserved for chunks with an
exact match to such a term. This lets an exact model number rank above a
semantically similar prefix match. Chunks can qualify through either signal.
Hybrid mode uses the stored text and embeddings and applies the same filters
before ranking. It works with both buffered and streaming answers.

Both search and answer accept an optional `filter` object. For example, a
filtered search body can select a document ID and its registered source.

```json
{
  "query": "When does the office close?",
  "topK": 3,
  "filter": {
    "documentIds": ["handbook"],
    "sources": ["https://example.com/handbook"]
  }
}
```

The filter uses exact, case-sensitive matches against the stored document ID
and source string. Values within either list are alternatives. If both lists
are supplied, a document must match both. Each supplied list must have 1 to
64 entries. Document IDs use the same validation as indexed IDs, and source
strings may have up to 2,048 UTF-16 code units. Omitted or null lists impose
no restriction. Filtering happens before similarity ranking and `topK`.
When no document matches, search returns an empty `hits` array and answer
returns `422 no_context`.

Set `"stream": true` in the answer request body to receive Server-Sent
Events. The first data event contains `id`, `model`, and the same `sources`
that a buffered answer would use. It is followed by chat completion chunks
with the same ID. A successful stream sends the final `finish_reason`
and `[DONE]`. Errors before the first event use the usual HTTP status
and JSON error body. Errors after output has started send an error event
and end without `[DONE]`.

## Container Stack

```bash
docker build --build-context aikernel-providers=../AIKernel.Providers --build-context aikernel-core=../AIKernel.Core -f docker/Dockerfile.base -t aikernel/monolith:base .
docker build -f docker/Dockerfile.linux -t aikernel/monolith:linux .
docker build -f docker/Dockerfile.windows -t aikernel/monolith:windows .
```

The base image contains the AOT-compiled monolith binary, entrypoint, and sample configuration. Linux and Windows images extend it with platform-specific native ABI layers.
The native build uses the sibling AIKernel.Core and AIKernel.Providers checkouts.
Build the C ABI library from AIKernel.Core/native/cuda with CMake and a C
compiler, and set AIKernelNativeComputeLibrary to the resulting DLL or SO
when publishing. Deploy the library beside the executable. The container
build compiles and copies it automatically.
Point `AIKernelKernelProject` and `ChatOpenAIProviderProject` at alternate source projects as needed.
Package-based builds can set `AIKernelKernelPackageVersion` and `ChatOpenAIProviderPackageVersion`
to versions with paged CUDA support and streaming/embedding support respectively.

Run with explicit mounts:

```bash
docker run --rm -p 8080:8080 \
  -v "$HOME/.aikernel/config:/config" \
  -v "$HOME/.aikernel/models:/models" \
  -v "$HOME/.aikernel/vfs:/vfs" \
  -v "$HOME/.aikernel/capabilities:/capabilities" \
  -e AIKERNEL_CONFIG_ROOT=/config \
  -e AIKERNEL_MODEL_ROOT=/models \
  -e AIKERNEL_VFS_ROOT=/vfs \
  -e AIKERNEL_CAPABILITY_ROOT=/capabilities \
  -e AIKERNEL_API_KEY \
  -e AIKERNEL_API_REQUEST_LIMIT \
  -e AIKERNEL_API_WINDOW_SECONDS \
  -e OPENAI_API_KEY \
  aikernel/monolith:linux
```

To verify a published Windows x64 Native AOT executable against a loopback Provider:

```powershell
python tests/native-aot-stream-embeddings.py <published-exe> <report-directory>
python tests/native-aot-api-access.py <published-exe> <report-directory>
python tests/native-aot-rag.py <published-exe> <report-directory>
```

## Python CLI

The `aikernel-monolith` package provides the `aikernel` command.

```bash
pip install ./python
aikernel init
aikernel up
aikernel status
aikernel stop
```

`aikernel up` forwards `AIKERNEL_API_KEY`, the two request budget settings, and
`OPENAI_API_KEY` from the host. It creates and mounts:

- `~/.aikernel/config`
- `~/.aikernel/models`
- `~/.aikernel/vfs`
- `~/.aikernel/capabilities`

## Determinism Rules

- Capability scanning uses ordinal ordering.
- Model metadata is listed by stable id/provider order.
- Persistent data belongs only in external mounts.
- Missing mount roots fail closed.

## Documentation

- [Architecture](docs/architecture.md)
- [Capability Loader](docs/capability-loader.md)
- [Routing](docs/routing.md)
- [VFS](docs/vfs.md)
- [Python CLI](docs/python-cli.md)
- [Container Stack](docs/container-stack.md)
