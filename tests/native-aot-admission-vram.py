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
import sys
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

    assert_process_stopped(probe_pid)
    return probe_pid


def assert_process_stopped(pid):
    kernel = process_api()
    with_handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
    if with_handle:
        try:
            assert kernel.WaitForSingleObject(with_handle, 0) == 0, f"CUDA probe {pid} is still running"
        finally:
            kernel.CloseHandle(with_handle)
    elif C.get_last_error() != 87:  # ERROR_INVALID_PARAMETER when the PID no longer exists
        raise C.WinError(C.get_last_error())


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


def send(port, key, method, path, body=None, sent=None, expected_status=200, retry_after=None):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    with closing(HTTPConnection("127.0.0.1", port, timeout=60)) as connection:
        connection.request(method, path, body, headers)
        if sent is not None:
            sent()
        response = connection.getresponse()
        payload = json.loads(response.read())
        assert response.status == expected_status, (response.status, payload)
        if retry_after is not None:
            assert response.getheader("Retry-After") == str(retry_after), payload
        return payload


def admission(port, key, name, sent=None):
    path = "/v1/models/admission?" + urlencode({"model": name})
    return send(port, key, "GET", path, sent=sent)


FAULT_PROBE_MARKER = re.compile(r"CUDA probe fault fixture ready pid=([0-9]+) devices=([0-9]+)")


def wait_for_fault_probe(process, log_path):
    deadline = time.monotonic() + 25
    while process.poll() is None and time.monotonic() < deadline:
        log = log_path.read_text(encoding="utf-8", errors="replace")
        assert "Now listening on:" not in log, log
        if match := FAULT_PROBE_MARKER.search(log):
            assert int(match[2]) >= 2, log
            return int(match[1])
        time.sleep(0.05)
    raise RuntimeError(log_path.read_text(encoding="utf-8", errors="replace"))


def assert_unavailable(statuses, topology_state, failure_reason):
    for ordinal, status in enumerate(statuses):
        assert not status["admissible"] and status["reason"] == "cuda_device_unavailable", status
        assert status["backend"] == "unknown" and status["admission_device"] is None, status
        assert status["cuda_topology_state"] == topology_state and status["retry_after_seconds"] == 5, status
        assert status["cuda_last_failure_reason"] == failure_reason, status
        if topology_state == "retry_wait":
            assert 0 <= status["cuda_next_retry_in_seconds"] <= 5, status
        else:
            assert "cuda_next_retry_in_seconds" not in status, status
        assert [gpu["device"] for gpu in status["devices"]] == [ordinal], status
        gpu = status["devices"][0]
        assert not gpu["migration_target_eligible"] and gpu["pci_bus_id"] is None, gpu
        assert gpu["temperature_celsius"] is None, gpu


def wait_for_retry_probe(process, log_path, previous_probe_pid):
    deadline = time.monotonic() + 30
    while process.poll() is None and time.monotonic() < deadline:
        log = log_path.read_text(encoding="utf-8")
        pids = [int(pid) for pid, _ in FAULT_PROBE_MARKER.findall(log)]
        if previous_probe_pid in pids:
            next_index = pids.index(previous_probe_pid) + 1
            if next_index < len(pids):
                return pids[next_index]
        time.sleep(0.05)
    raise AssertionError(f"Monolith did not start another CUDA probe: {log_path.read_text(encoding='utf-8')}")


def verify_probe_recovery(counter, process, log_path, port, key, previous_probe_pid, release_path, failure_reason):
    recovery_pid = wait_for_retry_probe(process, log_path, previous_probe_pid)
    probing = [admission(port, key, name) for name in ("first", "second")]
    assert_unavailable(probing, "probing", failure_reason)
    release_path.touch()

    def query_models(first_sent):
        first = admission(port, key, "first", sent=first_sent)
        deadline = time.monotonic() + 30
        while True:
            second = admission(port, key, "second")
            if first["admissible"] and second["admissible"]:
                break
            assert process.poll() is None and time.monotonic() < deadline, (first, second)
            time.sleep(0.05)
            first = admission(port, key, "first")

        for _ in range(5):
            admission(port, key, "first")
            admission(port, key, "second")
            time.sleep(0.05)
        return first, second

    (first, second), measurement = measure_queries(counter, process.pid, query_models)
    for ordinal, status in enumerate((first, second)):
        assert status["backend"] == "cuda" and status["reason"] is None, status
        assert status["cuda_topology_state"] == "ready" and status["retry_after_seconds"] is None, status
        assert "cuda_last_failure_reason" not in status and "cuda_next_retry_in_seconds" not in status, status
        assert status["device"] == status["admission_device"] == ordinal, status
    devices = verify_devices(first)
    verify_devices(second)
    assert process.poll() is None

    log = log_path.read_text(encoding="utf-8")
    assert log.count("Now listening on:") == 1, log
    probe_pids = [int(pid) for pid, _ in FAULT_PROBE_MARKER.findall(log)]
    assert probe_pids[-2:] == [previous_probe_pid, recovery_pid], log
    assert f"CUDA topology probe process {recovery_pid} exited with code 0." in log, log
    for pid in probe_pids:
        assert_process_stopped(pid)
    probe_vram = {pid: sum(counter.sample(pid).values()) for pid in probe_pids}
    assert all(used == 0 for used in probe_vram.values()), probe_vram
    return {
        "monolith_pid": process.pid, "probe_pids": probe_pids, "probe_vram_bytes": probe_vram,
        "dedicated_vram": measurement, "devices": devices, "model_statuses": [first, second],
        "probing_statuses": probing,
    }


def read_countdown(port, key, first_status, failure_reason, stop_at):
    assert_unavailable([first_status], "retry_wait", failure_reason)
    countdown = [first_status["cuda_next_retry_in_seconds"]]
    deadline = time.monotonic() + 3
    while countdown[-1] > stop_at:
        assert time.monotonic() < deadline, countdown
        time.sleep(0.1)
        status = admission(port, key, "first")
        assert_unavailable([status], "retry_wait", failure_reason)
        remaining = status["cuda_next_retry_in_seconds"]
        assert remaining <= countdown[-1], countdown + [remaining]
        countdown.append(remaining)
    return countdown


def verify_retry_timeout(process, log_path, port, key, first_probe_pid, initial_remaining):
    retry_pid = wait_for_retry_probe(process, log_path, first_probe_pid)
    probing = admission(port, key, "first")
    assert_unavailable([probing], "probing", "probe_nonzero_exit")

    deadline = time.monotonic() + 20
    while True:
        status = admission(port, key, "first")
        if status["cuda_topology_state"] == "retry_wait":
            break
        assert_unavailable([status], "probing", "probe_nonzero_exit")
        assert process.poll() is None and time.monotonic() < deadline, status
        time.sleep(0.1)

    assert_unavailable([status], "retry_wait", "probe_timeout")
    reset_remaining = status["cuda_next_retry_in_seconds"]
    assert 4 <= reset_remaining <= 5 and reset_remaining > initial_remaining, (initial_remaining, status)
    countdown = read_countdown(port, key, status, "probe_timeout", reset_remaining - 1)
    assert_process_stopped(retry_pid)
    log = log_path.read_text(encoding="utf-8")
    assert log.index(f"ready pid={retry_pid} ") < log.index("CUDA device enumeration timed out."), log
    return {
        "probe_pid": retry_pid, "probing_status": probing, "retry_wait_status": status,
        "retry_countdown_seconds": countdown,
    }


def signal_console_shutdown(pid):
    kernel = C.WinDLL("kernel32", use_last_error=True)
    kernel.AttachConsole.argtypes = (W.DWORD,)
    kernel.GetConsoleProcessList.argtypes = (C.POINTER(W.DWORD), W.DWORD)
    kernel.GetConsoleProcessList.restype = W.DWORD
    kernel.SetConsoleCtrlHandler.argtypes = (C.c_void_p, W.BOOL)
    kernel.GenerateConsoleCtrlEvent.argtypes = (W.DWORD, W.DWORD)
    if not kernel.FreeConsole() or not kernel.AttachConsole(pid):
        raise C.WinError(C.get_last_error())
    try:
        attached = (W.DWORD * 8)()
        count = kernel.GetConsoleProcessList(attached, len(attached))
        # The control event reaches Monolith and this helper. The CUDA probe has no console.
        assert count == 2 and set(attached[:count]) == {pid, os.getpid()}, list(attached[:count])
        if not kernel.SetConsoleCtrlHandler(None, True) or not kernel.GenerateConsoleCtrlEvent(0, 0):
            raise C.WinError(C.get_last_error())
    finally:
        kernel.FreeConsole()


def wait_for_probe_vram(counter, pid):
    deadline = time.monotonic() + 5
    while True:
        usage = counter.sample(pid)
        if sum(value > 0 for value in usage.values()) >= 2:
            return usage
        assert time.monotonic() < deadline, usage
        time.sleep(0.05)


def wait_for_zero_vram(counter, *pids):
    deadline = time.monotonic() + 3
    while True:
        usage = {pid: sum(counter.sample(pid).values()) for pid in pids}
        if not any(usage.values()):
            return usage
        assert time.monotonic() < deadline, usage
        time.sleep(0.05)


def verify_shutdown_during_retry(counter, process, log_path, first_probe_pid, forced):
    retry_pid = wait_for_retry_probe(process, log_path, first_probe_pid)
    kernel = process_api()
    # SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_TERMINATE
    retry = kernel.OpenProcess(0x00101001, False, retry_pid)
    if not retry:
        raise C.WinError(C.get_last_error())
    try:
        retry_vram = wait_for_probe_vram(counter, retry_pid)
        parent_vram = sum(counter.sample(process.pid).values())
        assert parent_vram == 0, parent_vram
        assert kernel.WaitForSingleObject(retry, 0) == 0x102, "Retry exited before shutdown"
        if forced:
            process.kill()  # Terminate only Monolith on Windows.
        else:
            signal = subprocess.run(
                [sys.executable, "-c",
                 "import runpy,sys; runpy.run_path(sys.argv[1])['signal_console_shutdown'](int(sys.argv[2]))",
                 str(Path(__file__).resolve()), str(process.pid)],
                creationflags=subprocess.CREATE_NO_WINDOW, capture_output=True, text=True, timeout=5)
            assert signal.returncode == 0, (signal.returncode, signal.stdout, signal.stderr)

        exit_code = process.wait(timeout=8)
        log = log_path.read_text(encoding="utf-8")
        if forced:
            assert exit_code != 0 and "Application is shutting down" not in log, (exit_code, log)
        else:
            assert exit_code == 0 and "Application is shutting down" in log, (exit_code, log)
        assert "CUDA device enumeration timed out." not in log, log
        assert [int(pid) for pid, _ in FAULT_PROBE_MARKER.findall(log)] == [first_probe_pid, retry_pid], log
        assert kernel.WaitForSingleObject(retry, 3000) == 0, f"CUDA retry {retry_pid} survived Monolith"
        retry_exit = W.DWORD()
        if not kernel.GetExitCodeProcess(retry, C.byref(retry_exit)):
            raise C.WinError(C.get_last_error())
        if not forced:
            assert retry_exit.value != 0, retry_exit.value
        usage = wait_for_zero_vram(counter, process.pid, first_probe_pid, retry_pid)
        return {
            "forced": forced, "monolith_exit_code": exit_code, "retry_pid": retry_pid,
            "retry_exit_code": retry_exit.value, "monolith_vram_before_shutdown_bytes": parent_vram,
            "retry_vram_before_shutdown_by_adapter": retry_vram, "vram_after_shutdown_bytes": usage,
        }
    finally:
        if kernel.WaitForSingleObject(retry, 0) == 0x102:
            kernel.TerminateProcess(retry, 1)
            kernel.WaitForSingleObject(retry, 5000)
        kernel.CloseHandle(retry)


def verify_probe_fault(process, log_path, helpers, key, fault, release_path, prompt):
    probe_pid = wait_for_fault_probe(process, log_path)
    assert probe_pid != process.pid
    kernel = process_api()
    # SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_TERMINATE
    handle = kernel.OpenProcess(0x00101001, False, probe_pid)
    if not handle:
        raise C.WinError(C.get_last_error())
    try:
        with closing(DedicatedVramCounter()) as counter:
            before = wait_for_probe_vram(counter, probe_pid)
            initial_timeout = fault == "timeout"
            failure_reason = "probe_timeout" if initial_timeout else "probe_nonzero_exit"
            assert kernel.WaitForSingleObject(handle, 0) == 0x102, "CUDA probe exited before fault injection"
            assert "Now listening on:" not in log_path.read_text(encoding="utf-8")
            if not initial_timeout and not kernel.TerminateProcess(handle, 73):
                raise C.WinError(C.get_last_error())

            port = helpers["wait_for_port"](process, log_path)
            log = log_path.read_text(encoding="utf-8")
            expected = (f"CUDA topology probe process {probe_pid} exited with code 73."
                        if not initial_timeout else "CUDA device enumeration timed out.")
            assert log.index(expected) < log.index("Now listening on:"), log
            assert kernel.WaitForSingleObject(handle, 0) == 0, f"CUDA probe {probe_pid} is still running"
            exit_code = W.DWORD()
            if not kernel.GetExitCodeProcess(handle, C.byref(exit_code)):
                raise C.WinError(C.get_last_error())
            if not initial_timeout:
                assert exit_code.value == 73, exit_code.value
            else:
                assert exit_code.value != 0, exit_code.value

            probe_after = wait_for_zero_vram(counter, probe_pid)[probe_pid]

            def query_models(first_sent):
                first = admission(port, key, "first", sent=first_sent)
                second = admission(port, key, "second")
                for name in ("first", "second"):
                    rejection = send(port, key, "POST", "/v1/completions",
                                     json.dumps({"model": name, "prompt": prompt}), expected_status=503, retry_after=5)
                    assert rejection["error"]["code"] == "cuda_device_unavailable", rejection
                    assert rejection["error"]["retry_after_seconds"] == 5, rejection
                for _ in range(5):
                    admission(port, key, "first")
                    admission(port, key, "second")
                    time.sleep(0.05)
                stop_at = first["cuda_next_retry_in_seconds"] - 1
                if fault == "abnormal-exit-then-timeout":
                    stop_at = min(stop_at, 3)
                countdown = read_countdown(port, key, first, failure_reason, stop_at)
                return [first, second], countdown

            (statuses, countdown), measurement = measure_queries(counter, process.pid, query_models)
            assert_unavailable(statuses, "retry_wait", failure_reason)
            probe_after_queries = sum(counter.sample(probe_pid).values())
            assert probe_after_queries == 0, probe_after_queries
            report = {
                "fault": fault, "monolith_pid": process.pid, "probe_pid": probe_pid,
                "probe_exit_code": exit_code.value, "probe_exited_before_listen": True,
                "probe_vram_before_fault_by_adapter": before,
                "probe_vram_after_exit_bytes": probe_after, "probe_vram_after_queries_bytes": probe_after_queries,
                "dedicated_vram": measurement, "model_statuses": statuses,
                "retry_countdown_seconds": countdown,
            }
            if fault in ("shutdown", "forced-shutdown"):
                report["shutdown"] = verify_shutdown_during_retry(
                    counter, process, log_path, probe_pid, forced=fault == "forced-shutdown")
            else:
                previous_probe_pid = probe_pid
                if fault == "abnormal-exit-then-timeout":
                    timed_out = verify_retry_timeout(process, log_path, port, key, probe_pid, countdown[-1])
                    report["timed_out_retry"] = timed_out
                    previous_probe_pid = timed_out["probe_pid"]
                    failure_reason = "probe_timeout"
                report["recovery"] = verify_probe_recovery(
                    counter, process, log_path, port, key, previous_probe_pid, release_path, failure_reason)
                expected_probes = 3 if fault == "abnormal-exit-then-timeout" else 2
                assert len(set(report["recovery"]["probe_pids"])) == expected_probes, report
            return report
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
        release_path = work / "release-probe"
        if fault is not None:
            executable = work / exe.name
            shutil.copyfile(exe, executable)
            assert helpers["native_image"](executable) == native_sha256
            shutil.copyfile(fault_library, work / "aikernel_cuda_driver.dll")
            real_driver = work / "aikernel_cuda_driver.real.dll"
            shutil.copyfile(exe.with_name("aikernel_cuda_driver.dll"), real_driver)
            env["AIKERNEL_TEST_REAL_CUDA_LIBRARY"] = str(real_driver)
            env["AIKERNEL_TEST_PROBE_RELEASE_FILE"] = str(release_path)

        log_path = output / "monolith.log"
        startup = None
        creationflags = subprocess.CREATE_NO_WINDOW
        if fault == "shutdown":
            startup = subprocess.STARTUPINFO()
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = subprocess.SW_HIDE
            creationflags = subprocess.CREATE_NEW_CONSOLE
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                [str(executable), "--urls=http://127.0.0.1:0",
                 *[f"--{name}-root={mount}" for name, mount in mounts.items()]],
                cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT,
                creationflags=creationflags, startupinfo=startup)
            try:
                if fault is not None:
                    report = verify_probe_fault(process, log_path, helpers, key, fault, release_path,
                                                standalone["PROMPT"])
                    records = helpers["audit_entries"](mounts["config"] / "inference-audit.jsonl")
                    assert [row["Description"] for row in records if row["EventType"] == "inference.rejected"] == [
                        "cuda_device_unavailable", "cuda_device_unavailable"]
                    assert all(row["EventType"] != "inference.execution" for row in records)
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
                        assert status["cuda_topology_state"] == "ready" and status["retry_after_seconds"] is None, status
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
                if fault is not None:
                    release_path.touch(exist_ok=True)
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
    parser.add_argument("--probe-fault-library", type=Path, help="Run probe failure, recovery, and shutdown cases with this DLL")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("Windows GPU Process Memory counters are required")
    test(args.exe.resolve(), args.assets.resolve(), args.output.resolve())
    if args.probe_fault_library:
        for fault in ("abnormal-exit", "timeout", "abnormal-exit-then-timeout", "shutdown", "forced-shutdown"):
            test(args.exe.resolve(), args.assets.resolve(), args.output.resolve() / fault,
                 fault_library=args.probe_fault_library.resolve(), fault=fault)
