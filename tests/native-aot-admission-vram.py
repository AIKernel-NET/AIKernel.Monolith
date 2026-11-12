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


def assert_probe_exited(log_path, monolith_pid):
    log = log_path.read_text(encoding="utf-8")
    probes = list(re.finditer(r"CUDA topology probe process ([0-9]+) exited with code ([0-9]+)\.", log))
    assert len(probes) == 1, log
    probe = probes[0]
    probe_pid, exit_code = map(int, probe.groups())
    listening = log.index("Now listening on:")
    assert probe_pid != monolith_pid and exit_code == 0 and probe.end() < listening, log

    kernel = C.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (W.DWORD, W.BOOL, W.DWORD)
    kernel.OpenProcess.restype = W.HANDLE
    kernel.WaitForSingleObject.argtypes = (W.HANDLE, W.DWORD)
    kernel.WaitForSingleObject.restype = W.DWORD
    kernel.CloseHandle.argtypes = (W.HANDLE,)
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


def test(exe, assets, output):
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
        log_path = output / "monolith.log"
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                [str(exe), "--urls=http://127.0.0.1:0",
                 *[f"--{name}-root={mount}" for name, mount in mounts.items()]],
                cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                port = helpers["wait_for_port"](process, log_path)
                probe_pid = assert_probe_exited(log_path, process.pid)
                headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

                def send(method, path, body=None, sent=None):
                    with closing(HTTPConnection("127.0.0.1", port, timeout=60)) as connection:
                        connection.request(method, path, body, headers)
                        if sent is not None:
                            sent()
                        response = connection.getresponse()
                        payload = json.loads(response.read())
                        assert response.status == 200, (response.status, payload)
                        return payload

                def admission(name, sent=None):
                    return send("GET", "/v1/models/admission?" + urlencode({"model": name}), sent=sent)

                def query_models(first_sent):
                    first = admission("first", sent=first_sent)
                    second = admission("second")
                    hot = admission("thermal")
                    for _ in range(10):
                        admission("first")
                        admission("second")
                        admission("thermal")
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
                        completion = send("POST", "/v1/completions", json.dumps(
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
                encoded = json.dumps(report, indent=2)
                (output / "admission-vram-result.json").write_text(encoded, encoding="utf-8")
                print(encoded)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe", type=Path)
    parser.add_argument("assets", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("Windows GPU Process Memory counters are required")
    test(args.exe.resolve(), args.assets.resolve(), args.output.resolve())
