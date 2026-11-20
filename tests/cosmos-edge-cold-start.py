"""Compare two Windows Native AOT builds using fresh hosts and the pinned Edge text case."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

MODEL = "cosmos-edge"
PIN = "8f696d470adeb78ffcc949f804c7c9ac8d7dd5bd9ac350a4a162037120a6c882"
MEMORY = dict(maximumResidentBytes=7_000_000_000, weightCacheBytes=1 << 30, kvCacheBytes=128_000_000)
MESSAGES = [
    dict(role="system", content="You are helpful."),
    dict(role="user", content=[dict(type="text", text="What is 2 + 2? Answer briefly.")]),
]
MEASUREMENTS = ("model_load_ms", "checkpoint_verify_ms", "generation_ms", "prefill_ms", "decode_ms",
                "prefill_weight_page_misses", "weight_backing_reads", "kv_swap_reads")
PROFILE_FIELDS = {"prefill_start_utc": str, "prefill_process_cpu_ms": float, "prefill_projection_rows_ms": float,
                  "prefill_projection_page_setup_ms": float, "prefill_attention_ms": float,
                  "prefill_projection_page_visits": int, "prefill_tile_tokens": int, "prefill_bf16_dot_kernel": str}


def idle_samples():
    def times():
        values = [ctypes.c_uint64() for _ in range(3)]
        if not ctypes.windll.kernel32.GetSystemTimes(*(ctypes.byref(v) for v in values)):
            raise ctypes.WinError()
        return [v.value for v in values]

    consecutive = []
    previous = times()
    while len(consecutive) < 2:
        time.sleep(15)
        current = times()
        idle, kernel, user = [b - a for a, b in zip(previous, current)]
        previous = current
        percent = 100 * (1 - idle / (kernel + user))
        consecutive = consecutive + [percent] if percent <= 10 else []
    return consecutive


class MemoryCounters(ctypes.Structure):
    _fields_ = [("size", ctypes.c_uint32), ("faults", ctypes.c_uint32), ("peak", ctypes.c_size_t),
                ("workingSet", ctypes.c_size_t), ("other", ctypes.c_size_t * 6)]


def memory_counters(process):
    counters = MemoryCounters()
    counters.size = ctypes.sizeof(counters)
    if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(int(process._handle)),
                                                    ctypes.byref(counters), counters.size):
        raise ctypes.WinError()
    return counters


def mount(root, edge, profile_prefill=False):
    for name in ("models", "config", "vfs", "capabilities"):
        (root / name).mkdir()
    dest = root / "models/edge"
    dest.mkdir()
    raw = (edge / "import-manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != PIN:
        raise ValueError("Expected the pinned Cosmos 3 Edge checkpoint used for the 32-token case")
    manifest = json.loads(raw)
    names = {"import-manifest.json", "model.safetensors.index.json", *manifest["metadataSha256"],
             *(s["output"] for s in manifest["shards"])}
    for name in names:
        if not name or name in (".", "..") or Path(name).name != name or ":" in name:
            raise ValueError("Invalid checkpoint filename")
        os.link(edge / name, dest / name)
    registration = dict(id=MODEL, provider="providers.cosmos-cpu", model="cosmos3-edge",
                        policy=dict(logos="approve", ethos="approve", pathos="approve"),
                        cosmos=dict(directory="edge", manifestSha256=PIN, cpuThreads=4, memory=MEMORY))
    if profile_prefill:
        registration["cosmos"]["profilePrefill"] = True
    (root / "models/reg.json").write_text(json.dumps(registration), encoding="utf-8")


def read_audit(path):
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    outcomes = [(r["EventType"], r["Description"]) for r in rows]
    if outcomes != [("inference.policy", "allowed"), ("inference.execution", "succeeded")]:
        raise ValueError(f"Unexpected audit outcomes: {outcomes}")
    metadata = rows[-1]["Metadata"]
    if (metadata["model_manifest_sha256"], metadata["backend"], int(metadata["input_tokens"])) != (PIN, "cpu", 32):
        raise ValueError("Unexpected checkpoint, backend or token count")
    excluded = {"request_id", "completion_id", *MEASUREMENTS, *PROFILE_FIELDS}
    stable = [dict(type=r["EventType"], outcome=r["Description"], subject=r["Subject"], severity=r["Severity"],
                   metadata={k: v for k, v in r["Metadata"].items() if k not in excluded}) for r in rows]
    measurements = {k: int(metadata[k]) if k in metadata else None for k in MEASUREMENTS}
    measurements.update({k: convert(metadata[k]) for k, convert in PROFILE_FIELDS.items() if k in metadata})
    return stable, measurements


def run(exe, root):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        base = f"http://127.0.0.1:{sock.getsockname()[1]}"
    key = secrets.token_hex(32)
    opener = build_opener(ProxyHandler({}))

    def call(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = Request(base + path, data=data, headers={"Authorization": "Bearer " + key,
                                                           "Content-Type": "application/json"})
        with opener.open(request, timeout=1000 if data else 3) as response:
            return json.load(response)

    args = [str(exe), "--urls", base] + [f"--{option}-root={root / directory}" for option, directory in
                                         [("config", "config"), ("model", "models"), ("vfs", "vfs"), ("capability", "capabilities")]]
    process = subprocess.Popen(args, cwd=root, env=dict(os.environ, AIKERNEL_API_KEY=key, DOTNET_PROCESSOR_COUNT="4"),
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            if process.poll() is not None:
                raise RuntimeError(f"Host exited with {process.returncode}")
            try:
                if call("/v1/models/admission?model=" + MODEL)["admissible"]:
                    break
                raise RuntimeError("Checkpoint rejected by admission")
            except URLError:
                time.sleep(0.5)
        else:
            raise TimeoutError("Host did not become ready")
        before = memory_counters(process)
        begin = datetime.now(timezone.utc).isoformat()
        start = time.perf_counter()
        choice = call("/v1/chat/completions", dict(model=MODEL, messages=MESSAGES))["choices"][0]
        elapsed = time.perf_counter() - start
        end = datetime.now(timezone.utc).isoformat()
        after = memory_counters(process)
        if (choice["message"]["content"], choice["finish_reason"]) != ("4", "stop"):
            raise ValueError("The reply differed from the reference")
        stable, measurements = read_audit(root / "config/inference-audit.jsonl")
        if not 0 < after.peak <= MEMORY["maximumResidentBytes"]:
            raise ValueError(f"Peak working set outside budget: {after.peak}")
        return dict(pid=process.pid, beginUtc=begin, endUtc=end, httpSeconds=elapsed, peakWorkingSetBytes=after.peak,
                    processPageFaults=after.faults - before.faults, answer="4", finish="stop",
                    audit=stable, measurements=measurements)
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-prefill", choices=("candidate", "both"),
                        help="Record detailed prefill diagnostics in hosts built with profiling support.")
    for name in ("baseline", "candidate", "edge", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--token-validator", type=Path,
                        help="Candidate AIKernel.Inference.Optimization.Validation executable for exact token parity")
    opts = parser.parse_args()
    if os.name != "nt":
        parser.error("The working-set and CPU counters require Windows.")
    if opts.output.exists():
        parser.error("Choose a new output file for each comparison.")
    builds = {"baseline": opts.baseline.resolve(strict=True), "candidate": opts.candidate.resolve(strict=True)}
    validator = opts.token_validator.resolve(strict=True) if opts.token_validator else None
    hashes = {}
    for name, path in builds.items():
        with path.open("rb") as handle:
            hashes[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    if hashes["baseline"] == hashes["candidate"]:
        parser.error("The executables are identical.")
    edge = opts.edge.resolve(strict=True)
    with opts.output.open("x", encoding="utf-8") as report:
        reference_audit = None
        for label in ("baseline", "candidate", "candidate", "baseline"):
            with tempfile.TemporaryDirectory(dir=edge.parent, prefix="cosmos-edge-ab-") as folder:
                root = Path(folder)
                profile = opts.profile_prefill == "both" or opts.profile_prefill == label
                mount(root, edge, profile)
                samples = idle_samples()
                result = run(builds[label], root)
            if profile and not PROFILE_FIELDS.keys() <= result["measurements"].keys():
                raise ValueError("The host did not report the requested prefill profile")
            if label == "candidate":
                metrics = result["measurements"]
                if any(metrics[k] is None or metrics[k] < 0 for k in MEASUREMENTS):
                    raise ValueError("The candidate did not report all prefill and paging measurements")
                if not 0 < metrics["prefill_weight_page_misses"] <= metrics["weight_backing_reads"]:
                    raise ValueError("Invalid prefill page miss count for the pinned checkpoint")
            matches = reference_audit is None or result["audit"] == reference_audit
            report.write(json.dumps(dict(build=label, exeSha256=hashes[label], preRunCpuPercent=samples,
                                         auditMatchesBaseline=matches, **result)) + "\n")
            report.flush()
            if not matches:
                raise ValueError("Stable governance or inference audit fields changed")
            reference_audit = result["audit"]
        if validator:
            with tempfile.TemporaryDirectory(dir=edge.parent, prefix="cosmos-edge-token-check-") as folder:
                root = Path(folder)
                memory = dict(MEMORY, swapDirectory=str(root / "swap"), expectedImportManifestSha256=PIN)
                request = dict(system=MESSAGES[0]["content"], content=MESSAGES[1]["content"],
                               enableThinking=False, maxNewTokens=32, memory=memory)
                (root / "request.json").write_text(json.dumps(request), encoding="utf-8")
                idle_samples()
                subprocess.run([str(validator), "--cosmos-generate", str(edge), "request.json", "result.json"],
                               cwd=root, env=dict(os.environ, DOTNET_PROCESSOR_COUNT="4"), check=True, timeout=1250)
                result = json.loads((root / "result.json").read_bytes())
                reference = (result["InputTokens"], result["TokenIds"], result["Text"], result["FinishReason"])
                if reference != (32, [1052, 11], "4", "stop"):
                    raise ValueError("Generated token sequence differed from the original reference")
                report.write(json.dumps(dict(check="referenceTokens", tokenIds=result["TokenIds"], matches=True)) + "\n")


if __name__ == "__main__":
    main()
