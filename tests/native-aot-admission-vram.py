"""Verify cold admission queries use zero dedicated VRAM on Windows with two real GPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import csv
import ctypes as C
from ctypes import wintypes as W
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import tempfile
import threading
import time
from urllib.parse import urlencode


PDH_MORE_DATA = 0x800007D2
PDH_FMT_LARGE = 0x400
PDH_CSTATUS_NO_INSTANCE = 0x800007D1
PDH_NO_DATA = 0x800007D5

class CounterValue(C.Structure):
    _fields_ = [("status", W.DWORD), ("bytes", C.c_longlong)]

class CounterItem(C.Structure):
    _fields_ = [("name", W.LPWSTR), ("value", CounterValue)]

class DedicatedVramCounter:
    def __init__(self):
        pdh = C.WinDLL("pdh.dll")
        self._open = pdh.PdhOpenQueryW
        self._open.argtypes = (W.LPCWSTR, C.c_size_t, C.POINTER(W.HANDLE))
        self._add = pdh.PdhAddEnglishCounterW
        self._add.argtypes = (W.HANDLE, W.LPCWSTR, C.c_size_t, C.POINTER(W.HANDLE))
        self._collect = pdh.PdhCollectQueryData
        self._collect.argtypes = (W.HANDLE,)
        self._array = pdh.PdhGetFormattedCounterArrayW
        self._array.argtypes = (W.HANDLE, W.DWORD, C.POINTER(W.DWORD), C.POINTER(W.DWORD), C.c_void_p)
        self._close = pdh.PdhCloseQuery
        self._close.argtypes = (W.HANDLE,)
        for fn in (self._open, self._add, self._collect, self._array, self._close):
            fn.restype = W.DWORD
        self.query = W.HANDLE()
        self.counter = W.HANDLE()
        self._check(self._open(None, 0, C.byref(self.query)))
        try:
            self._check(self._add(self.query, r"\GPU Process Memory(*)\Dedicated Usage", 0, C.byref(self.counter)))
        except BaseException:
            self.close()
            raise

    def close(self):
        self._check(self._close(self.query))

    @staticmethod
    def _check(status):
        if status:
            raise RuntimeError(f"PDH error 0x{status:08x}")

    def sample(self, pid):
        self._check(self._collect(self.query))
        size = W.DWORD()
        count = W.DWORD()
        status = self._array(self.counter, PDH_FMT_LARGE, C.byref(size), C.byref(count), None)
        if status in (PDH_CSTATUS_NO_INSTANCE, PDH_NO_DATA):
            return {}
        if status != PDH_MORE_DATA:
            self._check(status)
        while size.value:
            buffer = C.create_string_buffer(size.value)
            status = self._array(self.counter, PDH_FMT_LARGE, C.byref(size), C.byref(count), buffer)
            if status == PDH_MORE_DATA:
                continue
            self._check(status)
            rows = C.cast(buffer, C.POINTER(CounterItem))
            return {row.name: row.value.bytes for row in rows[:count.value]
                    if row.name.startswith(f"pid_{pid}_") and row.value.status in (0, 1)}
        return {}


def process_api():
    kernel = C.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (W.DWORD, W.BOOL, W.DWORD)
    kernel.OpenProcess.restype = W.HANDLE
    kernel.WaitForSingleObject.argtypes = (W.HANDLE, W.DWORD)
    kernel.WaitForSingleObject.restype = W.DWORD
    kernel.CloseHandle.argtypes = (W.HANDLE,)
    kernel.TerminateProcess.argtypes = (W.HANDLE, W.UINT)
    kernel.TerminateProcess.restype = W.BOOL
    kernel.GetExitCodeProcess.argtypes = (W.HANDLE, C.POINTER(W.DWORD))
    kernel.GetExitCodeProcess.restype = W.BOOL
    return kernel


def assert_probe_exited(log_path, monolith_pid):
    log = log_path.read_text(encoding="utf-8")
    probes = list(re.finditer(r"CUDA topology probe process ([0-9]+) exited with code ([0-9]+)\.", log))
    assert len(probes) == 1, log
    probe = probes[0]
    probe_pid, exit_code = map(int, probe.groups())
    listening = log.index("Now listening on:")
    assert probe_pid != monolith_pid and exit_code == 0 and probe.end() < listening, log

    kernel = process_api()
    with_handle = kernel.OpenProcess(0x00100000, False, probe_pid)  # SYNCHRONIZE
    if with_handle:
        try:
            assert kernel.WaitForSingleObject(with_handle, 0) == 0, f"CUDA probe {probe_pid} is still running"
        finally:
            kernel.CloseHandle(with_handle)
    elif C.get_last_error() != 87:  # ERROR_INVALID_PARAMETER when the PID no longer exists
        raise C.WinError(C.get_last_error())
    return probe_pid


def pci_key(bus_id):
    domain, bus, slot = bus_id.strip().lower().split(":")
    return int(domain, 16), int(bus, 16), slot


def verify_devices(status):
    listing = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=pci.bus_id,name", "--format=csv,noheader"],
        text=True, timeout=15)
    names = {pci_key(bus_id): name.strip() for bus_id, name in csv.reader(listing.splitlines())}
    devices = status["devices"]
    assert [gpu["device"] for gpu in devices] == [0, 1], devices
    observed = []
    for gpu in devices:
        assert gpu["telemetry_error"] is None and gpu["temperature_celsius"] > 2, gpu
        assert gpu["migration_target_eligible"] == (gpu["device"] == 1), gpu
        name = names[pci_key(gpu["pci_bus_id"])]
        observed.append({"name": name, **gpu})
    assert sum("RTX 4000" in gpu["name"] for gpu in observed) == 1, observed
    assert sum("T1000" in gpu["name"] for gpu in observed) == 1, observed
    return observed


def measure_queries(counter, pid, query):
    before = sum(counter.sample(pid).values())
    first_sent = threading.Event()
    first_sampled = threading.Event()
    first_request_bytes = None
    peak = 0
    samples = 0

    def sample_before_receiving_first_response():
        first_sent.set()
        assert first_sampled.wait(15), "VRAM sample missing after sending first query"

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(query, sample_before_receiving_first_response)
        try:
            while not pending.done():
                first_is_pending = first_sent.is_set() and first_request_bytes is None
                used = sum(counter.sample(pid).values())
                peak = max(peak, used)
                samples += 1
                if first_is_pending:
                    first_request_bytes = used
                    first_sampled.set()
                time.sleep(0.01)
        finally:
            first_sampled.set()
        responses = pending.result()
    after = sum(counter.sample(pid).values())
    measurement = {
        "before_bytes": before, "first_request_bytes": first_request_bytes,
        "peak_during_bytes": peak, "after_bytes": after, "samples_during": samples,
    }
    assert before == first_request_bytes == peak == after == 0 and samples >= 5, measurement
    return responses, measurement


def send(port, key, method, path, body=None, sent=None):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    with closing(HTTPConnection("127.0.0.1", port, timeout=60)) as connection:
        connection.request(method, path, body, headers)
        if sent is not None:
            sent()
        response = connection.getresponse()
        payload = json.loads(response.read())
        assert response.status == 200, (response.status, payload)
        return payload


def admission(port, key, name, sent=None):
    path = "/v1/models/admission?" + urlencode({"model": name})
    return send(port, key, "GET", path, sent=sent)


def wait_for_fault_probe(process, log_path):
    marker = re.compile(r"CUDA probe fault fixture ready pid=([0-9]+) devices=([0-9]+)")
    deadline = time.monotonic() + 25
    while process.poll() is None and time.monotonic() < deadline:
        log = log_path.read_text(encoding="utf-8", errors="replace")
        assert "Now listening on:" not in log, log
        if match := marker.search(log):
            assert int(match[2]) >= 2, log
            return int(match[1])
        time.sleep(0.05)
    raise RuntimeError(log_path.read_text(encoding="utf-8", errors="replace"))


def verify_probe_fault(process, log_path, helpers, key, fault):
    probe_pid = wait_for_fault_probe(process, log_path)
    assert probe_pid != process.pid
    kernel = process_api()
    # SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_TERMINATE
    handle = kernel.OpenProcess(0x00101001, False, probe_pid)
    if not handle:
        raise C.WinError(C.get_last_error())
    try:
        with closing(DedicatedVramCounter()) as counter:
            deadline = time.monotonic() + 5
            while True:
                before = counter.sample(probe_pid)
                if sum(value > 0 for value in before.values()) >= 2:
                    break
                assert time.monotonic() < deadline, before
                time.sleep(0.05)
            assert kernel.WaitForSingleObject(handle, 0) == 0x102, "CUDA probe exited before fault injection"
            assert "Now listening on:" not in log_path.read_text(encoding="utf-8")
            if fault == "abnormal-exit" and not kernel.TerminateProcess(handle, 73):
                raise C.WinError(C.get_last_error())

            port = helpers["wait_for_port"](process, log_path)
            log = log_path.read_text(encoding="utf-8")
            expected = (f"CUDA topology probe process {probe_pid} exited with code 73."
                        if fault == "abnormal-exit" else "CUDA device enumeration timed out.")
            assert log.index(expected) < log.index("Now listening on:"), log
            assert kernel.WaitForSingleObject(handle, 0) == 0, f"CUDA probe {probe_pid} is still running"
            exit_code = W.DWORD()
            if not kernel.GetExitCodeProcess(handle, C.byref(exit_code)):
                raise C.WinError(C.get_last_error())
            if fault == "abnormal-exit":
                assert exit_code.value == 73, exit_code.value
            else:
                assert exit_code.value != 0, exit_code.value

            deadline = time.monotonic() + 3
            while probe_after := sum(counter.sample(probe_pid).values()):
                assert time.monotonic() < deadline, probe_after
                time.sleep(0.05)

            def query_models(first_sent):
                first = admission(port, key, "first", sent=first_sent)
                second = admission(port, key, "second")
                for _ in range(5):
                    admission(port, key, "first")
                    admission(port, key, "second")
                    time.sleep(0.05)
                return first, second

            statuses, measurement = measure_queries(counter, process.pid, query_models)
            for ordinal, status in enumerate(statuses):
                assert not status["admissible"] and status["reason"] == "cuda_device_unavailable", status
                assert status["backend"] == "unknown" and status["admission_device"] is None, status
                assert [gpu["device"] for gpu in status["devices"]] == [ordinal], status
                for gpu in status["devices"]:
                    assert not gpu["migration_target_eligible"] and gpu["pci_bus_id"] is None, gpu
                    assert gpu["temperature_celsius"] is None, gpu
            probe_after_queries = sum(counter.sample(probe_pid).values())
            assert probe_after_queries == 0, probe_after_queries
            return {
                "fault": fault, "monolith_pid": process.pid, "probe_pid": probe_pid,
                "probe_exit_code": exit_code.value, "probe_exited_before_listen": True,
                "probe_vram_before_fault_by_adapter": before,
                "probe_vram_after_exit_bytes": probe_after, "probe_vram_after_queries_bytes": probe_after_queries,
                "dedicated_vram": measurement, "model_statuses": statuses,
            }
    finally:
        if kernel.WaitForSingleObject(handle, 0) == 0x102:
            kernel.TerminateProcess(handle, 1)
            kernel.WaitForSingleObject(handle, 5000)
        kernel.CloseHandle(handle)


def test(exe, assets, output, fault_library=None, fault=None):
    output.mkdir(parents=True, exist_ok=True)
    standalone = runpy.run_path(str(Path(__file__).with_name("native-aot-standalone.py")))
    helpers = standalone["helpers"]
    native_sha256 = helpers["native_image"](exe)
    key = standalone["KEY"]
    with tempfile.TemporaryDirectory(prefix="admission-vram-", dir=output) as directory:
        work = Path(directory)
        mounts = {name: work / name for name in ("config", "model", "vfs", "capability")}
        for mount in mounts.values():
            mount.mkdir()
        standalone["fixtures"](assets, mounts["model"])
        (mounts["model"] / "broken-tokenizer.json").unlink()

        limits = [
            {"device": 0, "maxTemperatureCelsius": 120, "maxMigrationTargetTemperatureCelsius": 1},
            {"device": 1, "maxTemperatureCelsius": 119, "maxMigrationTargetTemperatureCelsius": 118},
        ]

        def model(name, device, **policy):
            return {"id": name, "provider": "providers.paged-llama",
                    "policy": {"logos": "approve", "ethos": "approve", "pathos": "approve"},
                    "local": {"checkpoint": "stories.gguf", "device": device, "blockSize": 4096,
                              "ramBlocks": 4, "vramBlocks": 2, "kvRamBlocks": 1, "kvVramBlocks": 1,
                              "contextTokens": 64, "maxNewTokens": 8, **policy}}

        registrations = [
            model("first", 0, thermalDeviceLimits=limits),
            model("second", 1, thermalDeviceLimits=limits),
            model("thermal", 0, maxTemperatureCelsius=1),
        ]
        (mounts["model"] / "models.json").write_text(json.dumps(registrations), encoding="utf-8")
        env = dict(os.environ, AIKERNEL_API_KEY=key, AIKERNEL_API_REQUEST_LIMIT="1000",
                   Logging__LogLevel__Microsoft__Hosting__Lifetime="Information")
        env["Logging__LogLevel__AIKernel.Monolith.Models.CudaTopology"] = "Debug"
        executable = exe
        if fault is not None:
            executable = work / exe.name
            shutil.copyfile(exe, executable)
            assert helpers["native_image"](executable) == native_sha256
            shutil.copyfile(fault_library, work / "aikernel_cuda_driver.dll")
            real_driver = work / "aikernel_cuda_driver.real.dll"
            shutil.copyfile(exe.with_name("aikernel_cuda_driver.dll"), real_driver)
            env["AIKERNEL_TEST_REAL_CUDA_LIBRARY"] = str(real_driver)

        log_path = output / "monolith.log"
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                [str(executable), "--urls=http://127.0.0.1:0",
                 *[f"--{name}-root={mount}" for name, mount in mounts.items()]],
                cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                if fault is not None:
                    report = verify_probe_fault(process, log_path, helpers, key, fault)
                    assert not (mounts["config"] / "inference-audit.jsonl").exists()
                    assert not (mounts["config"] / "model-swap").exists()
                    report["native_sha256"] = native_sha256
                    write_report(output, report)
                    return

                port = helpers["wait_for_port"](process, log_path)
                probe_pid = assert_probe_exited(log_path, process.pid)

                def query_models(first_sent):
                    first = admission(port, key, "first", sent=first_sent)
                    second = admission(port, key, "second")
                    hot = admission(port, key, "thermal")
                    for _ in range(10):
                        admission(port, key, "first")
                        admission(port, key, "second")
                        admission(port, key, "thermal")
                        time.sleep(0.05)
                    return first, second, hot

                with closing(DedicatedVramCounter()) as counter:
                    (first, second, hot), measurement = measure_queries(counter, process.pid, query_models)
                    for ordinal, status in enumerate((first, second)):
                        assert status["admissible"] and status["backend"] == "cuda", status
                        assert status["device"] == status["admission_device"] == ordinal, status
                    assert not hot["admissible"] and hot["admission_device"] is None, hot
                    assert hot["reason"] == "device_temperature_exceeded" and hot["retry_after_seconds"] == 5, hot
                    devices = verify_devices(first)
                    assert not (mounts["config"] / "inference-audit.jsonl").exists()
                    assert not (mounts["config"] / "model-swap").exists()

                    # Prove the counter detects real allocations in the same PID on both adapters.
                    for name in ("first", "second"):
                        completion = send(port, key, "POST", "/v1/completions", json.dumps(
                            {"model": name, "prompt": standalone["PROMPT"]}))
                        assert completion["choices"][0]["text"] == standalone["EXPECTED"], completion
                    for _ in range(30):
                        control = counter.sample(process.pid)
                        if sum(value > 0 for value in control.values()) >= 2:
                            break
                        time.sleep(0.1)
                    assert sum(value > 0 for value in control.values()) >= 2, control

                report = {
                    "native_sha256": native_sha256,
                    "monolith_pid": process.pid, "probe_pid": probe_pid, "probe_exited_before_listen": True,
                    "admission_requests": 33, "dedicated_vram": measurement, "devices": devices,
                    "admitted_models": [first["model"], second["model"]], "thermal_denial": hot["reason"],
                    "dedicated_vram_after_inference_by_adapter": control,
                }
                write_report(output, report)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=15)


def write_report(output, report):
    encoded = json.dumps(report, indent=2)
    (output / "admission-vram-result.json").write_text(encoded, encoding="utf-8")
    print(encoded, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe", type=Path)
    parser.add_argument("assets", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--probe-fault-library", type=Path, help="Run abnormal-exit and timeout cases with this DLL")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("Windows GPU Process Memory counters are required")
    test(args.exe.resolve(), args.assets.resolve(), args.output.resolve())
    if args.probe_fault_library:
        for fault in ("abnormal-exit", "timeout"):
            test(args.exe.resolve(), args.assets.resolve(), args.output.resolve() / fault,
                 fault_library=args.probe_fault_library.resolve(), fault=fault)
