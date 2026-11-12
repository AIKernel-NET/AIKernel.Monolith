"""Exercise two real thermal migrations and pre-token disconnects on one model session."""
import argparse
from contextlib import ExitStack, closing
import ctypes as c
from ctypes import wintypes as w
from http.client import HTTPConnection
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("exe", type=Path, help="published Windows Native AOT Monolith executable")
parser.add_argument("checkpoint", type=Path, help="OpenLLaMA 3B v2 Q4_0 GGUF with an embedded tokenizer")
parser.add_argument("output", type=Path, help="new report directory")
parser.add_argument("--devices", type=int, nargs=2, metavar=("SOURCE", "TARGET"), default=(0, 1))
parser.add_argument("--limit", type=int, default=78, help="real NVML temperature threshold in Celsius")
parser.add_argument("--device-limits", type=int, nargs=4,
    metavar=("SOURCE_RUN", "SOURCE_REUSE", "TARGET_RUN", "TARGET_REUSE"),
    help="per-device running and migration target limits in Celsius, in --devices order")
parser.add_argument("--arm-after", type=float, default=22, help="seconds to wait for a natural source crossing")
parser.add_argument("--cooldown-timeout", type=float, default=180, help="seconds to wait for cool GPUs before admission")
parser.add_argument("--return-margin", type=int, default=1, help="degrees below the threshold before heating the target")
args = parser.parse_args()
if sys.platform != "win32":
    parser.error("this regression test requires Windows process I/O counters")
source_ordinal, target_ordinal = args.devices
if min(args.devices) < 0 or source_ordinal == target_ordinal:
    parser.error("provide two distinct nonnegative CUDA ordinals")
source_limit, source_reuse, target_limit, target_reuse = args.device_limits or (args.limit,) * 4
if not (1 <= source_reuse <= source_limit <= 82 and 1 <= target_reuse <= target_limit <= 82 and
        min(source_limit, target_limit) >= 3 and 0 <= args.arm_after <= 90):
    parser.error("limit must be 3..82 °C and arm-after must be 0..90 seconds")
if not 1 <= args.cooldown_timeout <= 300 or not 0 <= args.return_margin < source_reuse:
    parser.error("cooldown-timeout must be 1..300 seconds and return-margin must be below the limit")
thermal_limits = [
    {"device": source_ordinal, "maxTemperatureCelsius": source_limit, "maxMigrationTargetTemperatureCelsius": source_reuse},
    {"device": target_ordinal, "maxTemperatureCelsius": target_limit, "maxMigrationTargetTemperatureCelsius": target_reuse},
]
exe = args.exe.resolve(strict=True)
checkpoint = args.checkpoint.resolve(strict=True)
args.output = args.output.resolve()
if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
    parser.error("use a new or empty report directory")
if checkpoint.suffix.lower() != ".gguf":
    parser.error("provide the OpenLLaMA 3B v2 GGUF checkpoint")
root = Path(__file__).resolve().parent
standalone = runpy.run_path(str(root / "native-aot-standalone.py"))
helpers, key = standalone["helpers"], standalone["KEY"]
bridgedll = exe.with_name("aikernel_cuda_driver.dll")
nvml = c.WinDLL(str(Path(os.environ["SystemRoot"]) / "System32/nvml.dll"))
cuda = c.CDLL(str(bridgedll))
assert nvml.nvmlInit_v2() == 0
devices = []
for ordinal in args.devices:
    ctx, bus, handle, name = c.c_void_p(), c.create_string_buffer(32), c.c_void_p(), c.create_string_buffer(96)
    assert cuda.ak_create(1, ordinal, c.byref(ctx)) == 0
    assert cuda.ak_get_pci_bus_id(ctx, bus, len(bus)) == 0
    assert cuda.ak_destroy(ctx) == 0
    assert nvml.nvmlDeviceGetHandleByPciBusId_v2(bus, c.byref(handle)) == 0
    assert nvml.nvmlDeviceGetName(handle, name, len(name)) == 0
    devices.append((handle, bus.value.decode(), name.value.decode()))
assert devices[0][1] != devices[1][1], "CUDA ordinals resolved to the same GPU"
source, target = devices[0][1], devices[1][1]
kernel32 = c.WinDLL("kernel32", use_last_error=True)
kernel32.OpenProcess.argtypes = (w.DWORD, w.BOOL, w.DWORD)
kernel32.OpenProcess.restype = w.HANDLE
kernel32.CloseHandle.argtypes = (w.HANDLE,)
class IO(c.Structure):
    _fields_ = [(name, c.c_uint64) for name in
        ("reads", "writes", "other", "read_bytes", "write_bytes", "other_bytes")]
kernel32.GetProcessIoCounters.argtypes = (w.HANDLE, c.POINTER(IO))


def temperatures():
    values = []
    for handle, _, _ in devices:
        temp = c.c_uint()
        assert nvml.nvmlDeviceGetTemperature(handle, 0, c.byref(temp)) == 0
        values.append(temp.value)
    return values


def context_devices(pid):
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
    entries = [line.split(",") for line in r.stdout.splitlines() if "," in line]
    buses = {bus.strip().lower().split(":", 1)[-1] for bus, owner in entries if owner.strip() == str(pid)}
    return [pci for _, pci, _ in devices if pci.lower().split(":", 1)[-1] in buses]


def fingerprint(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

report = {"binary_sha256": helpers["native_image"](exe), "bridge_sha256": fingerprint(bridgedll),
    "checkpoint_sha256": fingerprint(checkpoint),
    "gpus": [{"ordinal": ordinal, "pci": p, "name": n} for ordinal, (_, p, n) in zip(args.devices, devices)],
    "thermal_limits": thermal_limits, "return_margin": args.return_margin, "cycles": []}
if not args.device_limits:
    report["threshold"] = args.limit
args.output.mkdir(parents=True, exist_ok=True)
def save():
    (args.output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def request(conn, model, prompt, outcome):
    outcome.update(sent_at=time.perf_counter(), pieces=[], events=[])
    try:
        conn.request("POST", "/v1/completions", json.dumps({"model": model, "stream": True, "prompt": prompt}),
            {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        response = conn.getresponse()
        outcome["http"] = response.status
        if response.status != 200:
            outcome["error"] = json.loads(response.read())["error"]
            return
        for item in helpers["events"](response):
            if item == "[DONE]":
                outcome["events"].append("done")
                continue
            data = json.loads(item)
            if "error" in data:
                outcome["error"] = data["error"]
                continue
            outcome["id"] = data["id"]
            choice = data["choices"][0]
            if choice["text"]:
                outcome["pieces"].append(choice["text"])
                outcome["events"].append("text")
            if choice.get("finish_reason"):
                outcome["finish"] = choice["finish_reason"]
    except (OSError, EOFError) as ex:
        outcome["socket_error"] = str(ex)
    finally:
        outcome["ended_at"] = time.perf_counter()


def stop_process(process):
    if process.poll() is None:
        process.terminate()
        process.wait(25)


def start_worker(fn, *args):
    thread = threading.Thread(target=fn, args=args, daemon=True)
    thread.start()
    return thread


def launch(folder, stack, baseline=False):
    folder.mkdir()
    tmp = stack.enter_context(tempfile.TemporaryDirectory(dir=folder))
    paths = {kind: Path(tmp) / kind for kind in ("model", "config", "vfs", "capability")}
    for path in paths.values():
        path.mkdir()
    model_path = paths["model"] / checkpoint.name
    try:
        os.link(checkpoint, model_path)
    except OSError:
        shutil.copyfile(checkpoint, model_path)
    local = {"checkpoint": checkpoint.name, "device": source_ordinal, "blockSize": 33554432,
        "ramBlocks": 8, "vramBlocks": 8, "kvRamBlocks": 2, "kvVramBlocks": 26, "contextTokens": 64,
        "maxNewTokens": 3,
        "encryptSwap": True, "zeroWipe": True, "thermalMigrationDevices": list(args.devices)}
    if args.device_limits and not baseline:
        local["thermalDeviceLimits"] = thermal_limits
    else:
        local["maxTemperatureCelsius"] = 100 if baseline else args.limit
    entries = [{"id": "thermal", "provider": "providers.paged-llama",
        "policy": {"logos": "approve", "ethos": "approve", "pathos": "approve"}, "local": local}]
    (paths["model"] / "models.json").write_text(json.dumps(entries), encoding="utf-8")
    env = dict(os.environ, AIKERNEL_API_KEY=key, AIKERNEL_API_REQUEST_LIMIT="1000")
    env["Logging__LogLevel__Microsoft.Hosting.Lifetime"] = "Information"
    env["Logging__LogLevel__AIKernel.Monolith.Models.Local.PagedLlamaModel"] = "Debug"
    log_path = folder / "monolith.log"
    log = stack.enter_context(log_path.open("wb"))
    app = subprocess.Popen([str(exe), "--urls=http://127.0.0.1:0",
        *[f"--{name}-root={path}" for name, path in paths.items()]], cwd=tmp, env=env, stdout=log,
        stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
    stack.callback(stop_process, app)
    return app, helpers["wait_for_port"](app, log_path), paths["config"]


def audit(config):
    try:
        return helpers["audit_entries"](config / "inference-audit.jsonl")
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def swaps(config):
    result = {}
    for p in (config / "model-swap").glob(".aikernel-swap-*"):
        try:
            result[p.name] = p.stat().st_size
        except FileNotFoundError:
            pass
    return result


def heater(folder, stack, ordinal, preheat=False):
    prefix, pci = ("source", source) if ordinal == source_ordinal else ("target", target)
    limit = source_limit if ordinal == source_ordinal else target_limit
    ready, arm, stop = [(folder / (prefix + "-" + name)).resolve() for name in ("ready", "arm", "stop")]
    log = stack.enter_context((folder / (prefix + "-heat.jsonl")).open("wb"))
    command = [sys.executable, "-u", str(root / "thermal_cuda_load.py"),
        "--ordinal", str(ordinal), "--pci", pci, "--threshold", str(limit), "--seconds", "150",
        "--ready-file", str(ready), "--arm-file", str(arm), "--stop-file", str(stop)]
    if preheat:
        command.append("--preheat")
        if args.device_limits:
            command.extend(["--preheat-celsius", str(min(target_reuse - 1, target_limit - 2))])
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
    stack.callback(stop_process, process)
    return ready, arm, stop


def cycle(app, port, config, n, model):
    folder = args.output / f"cycle-{n}"
    folder.mkdir()
    origin = time.perf_counter()
    now = lambda: round(time.perf_counter() - origin, 6)
    evidence = {"pid": app.pid, "model": model, "nvml": [], "compute_contexts": [], "kv_swap": [], "read_counters": []}
    report["cycles"].append(evidence)
    lead = evidence["lead"] = {}
    follower = evidence["follower"] = {}
    with ExitStack() as stack:
        source_ready, source_arm, source_stop = heater(folder, stack, source_ordinal)
        reverse = n == 1
        target_ready, target_arm, target_stop = heater(folder, stack, target_ordinal, preheat=True) if reverse else (None, None, None)
        conn = stack.enter_context(closing(HTTPConnection("127.0.0.1", port, timeout=540)))
        queued = threading.Event()
        stop = threading.Event()
        process = kernel32.OpenProcess(0x1000, False, app.pid)
        if not process:
            raise c.WinError(c.get_last_error())
        stack.callback(kernel32.CloseHandle, process)
        old_requests = {row["Metadata"]["request_id"] for row in audit(config)}
        def current_audit():
            return [row for row in audit(config) if row["Metadata"]["request_id"] not in old_requests]
        def sample():
            previous_io = None
            previous_swap = None
            header = None
            samples = []
            io = IO()
            while not stop.is_set():
                temps = temperatures()
                evidence["nvml"].append([now(), *temps])
                swap = swaps(config)
                if swap != previous_swap:
                    evidence["kv_swap"].append([now(), swap])
                    previous_swap = swap
                if not kernel32.GetProcessIoCounters(process, c.byref(io)):
                    evidence["io_error"] = c.get_last_error()
                    break
                current_io = [now(), io.reads, io.read_bytes]
                if previous_io is not None:
                    operations = current_io[1] - previous_io[1]
                    byte_delta = current_io[2] - previous_io[2]
                    if operations or byte_delta:
                        evidence["read_counters"].append([*current_io, operations, byte_delta])
                    if (queued.is_set() and evidence.get("target_at", float("inf")) <= previous_io[0] and
                        "abort" not in evidence and len(swap) == 1):
                        match = None
                        if operations and operations % 2 == 0 and byte_delta == operations // 2 * 65564:
                            match = [previous_io, current_io]
                        elif (operations, byte_delta) == (1, 28):
                            header = [previous_io, current_io]
                        elif (operations, byte_delta) == (1, 65536) and header and previous_io[1:] == header[-1][1:]:
                            match = [*header, current_io]
                            header = None
                        elif operations or byte_delta:
                            header = None
                        if match is not None:
                            samples.append(match)
                        if (len(samples) >= 2 and samples[-1][-1][0] - samples[0][-1][0] >= .20 and
                            not lead.get("pieces") and "ended_at" not in lead and conn.sock is not None):
                            try:
                                conn.sock.shutdown(socket.SHUT_RDWR)
                                evidence["abort"] = {"at": now(), "read_pairs": samples, "swap": swap, "nvml": temps}
                                print(json.dumps({"cycle": n, "pid": app.pid, "abort": evidence["abort"]}), flush=True)
                            except OSError as ex:
                                evidence["abort_error"] = str(ex)
                if reverse and "abort" in evidence and not target_arm.exists() and not target_stop.exists():
                    if not set(evidence["abort"]["swap"]).intersection(swap) and 0 < temps[0] <= source_reuse - args.return_margin:
                        target_arm.touch()
                        evidence["target_heater_armed_at"] = now()
                        print(json.dumps({"cycle": n, "target_heat_at": now(), "nvml": temps}), flush=True)
                previous_io = current_io
                stop.wait(.006)
        def monitor_gpu():
            while not stop.is_set():
                active = context_devices(app.pid)
                evidence["compute_contexts"].append([now(), active])
                if active == [target] and "target_at" not in evidence:
                    evidence["target_at"] = now()
                    source_stop.touch()
                    print(json.dumps({"cycle": n, "target_at": evidence["target_at"], "nvml": temperatures()}), flush=True)
                if reverse and active == [source] and "abort" in evidence and "reverse_at" not in evidence:
                    evidence["reverse_at"] = now()
                    target_stop.touch()
                    print(json.dumps({"cycle": n, "return_to_source_at": evidence["reverse_at"], "nvml": temperatures()}), flush=True)
                stop.wait(.25)
        monitor = start_worker(monitor_gpu)
        sampler = start_worker(sample)
        def ready_temperatures():
            source_temp, target_temp = temperatures()
            return 0 < source_temp <= source_limit and 0 < target_temp <= target_reuse

        try:
            deadline = time.monotonic() + args.cooldown_timeout
            while (not source_ready.exists() or (reverse and not target_ready.exists()) or
                   not ready_temperatures()):
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"GPUs did not cool to {source_limit}/{target_reuse} °C before admission: {temperatures()}")
                time.sleep(.05)
            for attempt in range(5):
                lead.clear()
                lead_thread = start_worker(request, conn, model, "Hello. "*12 + "How are you?", lead)
                started = time.monotonic()
                while not any(row["EventType"] == "inference.policy" for row in current_audit()):
                    if not lead_thread.is_alive() or time.monotonic() - started > 12:
                        raise RuntimeError("Lead did not reach admission")
                    time.sleep(.01)
                # A display workload can briefly warm the source between the NVML sample and server admission.
                lead_thread.join(.65)
                if lead.get("error", {}).get("code") != "device_temperature_exceeded":
                    break
                conn.close()
                old_requests.update(row["Metadata"]["request_id"] for row in current_audit())
                while not ready_temperatures() and time.monotonic() < deadline:
                    time.sleep(.07)
            assert lead_thread.is_alive(), lead
            follow_conn = stack.enter_context(closing(HTTPConnection("127.0.0.1", port, timeout=540)))
            follow_thread = start_worker(request, follow_conn, model, "Once", follower)
            wait_until = time.monotonic() + args.arm_after
            while time.monotonic() < wait_until:
                rows = current_audit()
                if len(rows) == 2 and all(row["EventType"] == "inference.policy" for row in rows):
                    evidence.setdefault("queued_at", now())
                    queued.set()
                if "abort" in evidence or not lead_thread.is_alive():
                    break
                time.sleep(.02)
            if "target_at" not in evidence and "abort" not in evidence:
                source_arm.touch()
                print(json.dumps({"cycle": n, "source_heat_at": now(), "nvml": temperatures()}), flush=True)
            lead_thread.join(170)
            if lead_thread.is_alive():
                if conn.sock is not None:
                    conn.sock.shutdown(socket.SHUT_RDWR)
                lead_thread.join(15)
            follow_thread.join(550)
            assert not follow_thread.is_alive(), "Follower timed out"
            deadline = time.monotonic() + 5
            while len([r for r in current_audit() if r["EventType"] == "inference.execution"]) < 2 and time.monotonic() < deadline:
                time.sleep(.05)
        finally:
            stop.set()
            source_stop.touch()
            if reverse:
                target_stop.touch()
            monitor.join(12)
            sampler.join(5)
            evidence.update(audit=current_audit(), remaining_swap=swaps(config), process_alive=app.poll() is None)
            (folder / "audit.json").write_text(json.dumps(evidence["audit"], indent=2), encoding="utf-8")
            save()
    return evidence


def metadata(evidence, description):
    return next((r["Metadata"] for r in evidence["audit"] if r["Description"] == description), {})


def live(evidence):
    lead = metadata(evidence, "canceled")
    follower = metadata(evidence, "succeeded")
    abort = evidence.get("abort", {})
    name = next(iter(abort.get("swap", {})), None)
    swap_states = evidence["kv_swap"]
    removed = [at for at, files in swap_states if at > abort.get("at", float("inf")) and name not in files]
    before = int(lead.get("kv_cuda_vram_bytes_before_dispose", 0))
    weights = int(lead.get("weight_cuda_vram_bytes_before_kv_dispose", 0))
    target_at = evidence.get("target_at", 0)
    lead_rows = [r for r in evidence["audit"] if r["EventType"] == "inference.policy"]
    checks = {
        "nvml_crossing": bool(target_at and any(target_at - 10 <= t <= target_at and a > source_limit and
            0 < b <= target_reuse for t, a, b in evidence["nvml"])),
        "committed_source_to_target": lead.get("cuda_device_ordinal") == str(target_ordinal) and lead.get("gpu_migrations") == "1" and
            lead.get("gpu_migration_from_pci") == source and lead.get("gpu_migration_to_pci") == target,
        "canceled_after_target_page_in_before_token": len(abort.get("read_pairs", [])) >= 2 and
            int(lead.get("kv_swap_reads_since_migration", 0)) >= 2 and lead.get("gpu_migration_after_generated_tokens") == "0" and
            "local_prefill_tokens" not in lead and not evidence["lead"]["events"],
        "kv_freed_in_cuda": before > 0 and before == int(lead.get("kv_cuda_vram_bytes_released_on_dispose", 0)) and
            lead.get("kv_cuda_vram_bytes_after_dispose") == "0",
        "weights_retained_in_cuda": weights > 0 and weights == int(lead.get("weight_cuda_vram_bytes_after_kv_dispose", 0)),
        "no_stale_swap": bool(name and removed and not evidence["remaining_swap"] and
            all(set(files) <= {name} for at, files in swap_states if at < removed[0])),
        "follower_was_waiting": len(lead_rows) == 2 and evidence.get("queued_at", float("inf")) < abort.get("at", 0) and
            {r["Metadata"]["request_id"] for r in lead_rows} == {lead.get("request_id"), follower.get("request_id")},
        "follower_finished_in_same_process": evidence["process_alive"] and complete(evidence["follower"]) and
            follower.get("completion_id") == evidence["follower"].get("id"),
        "follower_left_no_kv": follower.get("kv_cuda_vram_bytes_after_dispose") == "0",
    }
    if args.device_limits:
        checks["per_device_limits_applied"] = accepted_limits(lead, source_limit, target_reuse)
    return checks


def accepted_limits(record, running_limit, reuse_limit):
    return (record.get("gpu_migration_source_temperature_limit_celsius") == str(running_limit) and
        record.get("gpu_migration_target_temperature_limit_celsius") == str(reuse_limit) and
        0 < int(record.get("gpu_migration_target_temperature_celsius", 0)) <= reuse_limit)


def complete(outcome):
    return outcome.get("http") == 200 and "error" not in outcome and outcome["events"][-1:] == ["done"]


def isolated():
    with ExitStack() as stack:
        app, port, config = launch(args.output / "isolated", stack, baseline=True)
        outcome = {}
        with closing(HTTPConnection("127.0.0.1", port, timeout=300)) as conn:
            request(conn, "thermal", "Once", outcome)
        result = {"pid": app.pid, "follower": outcome, "audit": audit(config)}
        assert complete(outcome), outcome
        return result

try:
    with ExitStack() as stack:
        app, port, config = launch(args.output / "monolith", stack)
        model = "thermal"
        for n in (1, 2):
            evidence = cycle(app, port, config, n, model)
            evidence["checks"] = live(evidence)
            if n == 1:
                successor = metadata(evidence, "succeeded")
                reverse_at = evidence.get("reverse_at", 0)
                armed_at = evidence.get("target_heater_armed_at", float("inf"))
                evidence["checks"]["follower_returned_to_cool_source"] = (
                    successor.get("gpu_migrations") == "1" and successor.get("gpu_migration_from_pci") == target and
                    successor.get("gpu_migration_to_pci") == source and successor.get("cuda_device_ordinal") == str(source_ordinal) and
                    bool(reverse_at and any(max(armed_at, reverse_at - 12) <= t <= reverse_at and
                        0 < r <= source_reuse and x > target_limit for t, r, x in evidence["nvml"])))
                if args.device_limits:
                    evidence["checks"]["return_per_device_limits_applied"] = accepted_limits(successor, target_limit, source_reuse)
            save()
            print(json.dumps({"cycle": n, "checks": evidence["checks"], "lead": metadata(evidence, "canceled"),
                "follower": evidence["follower"]["pieces"]}), flush=True)
            assert all(evidence["checks"].values()), "Cycle verification failed"
    baseline = report["isolated"] = isolated()
    reference = metadata(baseline, "succeeded")
    assert (reference.get("cuda_device_ordinal") == str(source_ordinal) and "gpu_migrations" not in reference and
        reference.get("completion_id") == baseline["follower"].get("id") and
        "local_prefill_tokens" in reference and "local_forward_layers" in reference), "Invalid isolated reference"
    for e in report["cycles"]:
        successor = metadata(e, "succeeded")
        e["checks"]["matches_isolated_output"] = all(e["follower"].get(k) == baseline["follower"].get(k)
            for k in ("pieces", "finish")) and all(successor.get(k) == reference.get(k)
            for k in ("local_prefill_tokens", "local_forward_layers"))
    report["verified"] = len(report["cycles"]) == 2 and len({(e["pid"], e["model"]) for e in report["cycles"]}) == 1 and all(
        all(e["checks"].values()) for e in report["cycles"])
    assert report["verified"], "Successors did not match the isolated run"
    save()
    print(json.dumps({"verified": report.get("verified"), "reference": baseline["follower"]["pieces"]}), flush=True)
except Exception as error:
    report["verified"] = False
    report["error"] = str(error)
    save()
    raise
finally:
    nvml.nvmlShutdown()
