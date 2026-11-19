"""Verify Linux CUDA retry teardown and per-process VRAM after a parent SIGKILL."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time


def pci_key(bus_id):
    domain, bus, slot = bus_id.strip().lower().split(":")
    return f"{int(domain, 16):04x}:{int(bus, 16):02x}:{slot}"


def nvidia_smi(query):
    rows = subprocess.check_output(
        ["nvidia-smi", query, "--format=csv,noheader,nounits"], text=True, timeout=10)
    return [[field.strip() for field in row] for row in csv.reader(rows.splitlines()) if row]


def process_vram(pid, devices):
    usage = {}
    unknown = set()
    for bus, owner, memory in nvidia_smi("--query-compute-apps=gpu_bus_id,pid,used_gpu_memory"):
        pci = pci_key(bus)
        if owner.isdecimal() and int(owner) == pid and pci in devices:
            if memory.isdecimal():
                usage[pci] = usage.get(pci, 0) + int(memory) * 1024 * 1024
            else:
                unknown.add(pci)
    for pci in unknown:
        usage[pci] = None
    return usage


def wait_for_retry(process, log_path):
    deadline = time.monotonic() + 25
    while process.poll() is None and time.monotonic() < deadline:
        log = log_path.read_text(encoding="utf-8")
        if match := re.search(r"CUDA retry ready pid=([0-9]+)", log):
            failure = re.search(r"CUDA topology probe process ([0-9]+) exited with code 73[.]", log)
            assert failure and failure.end() < log.index("Now listening on:") < match.start(), log
            allocated = {pci_key(bus): int(size) for size, bus in
                         re.findall(r"CUDA retry allocated ([0-9]+) B on ([0-9a-fA-F:.]+)", log)}
            return int(failure[1]), int(match[1]), allocated
        time.sleep(0.05)
    raise AssertionError(log_path.read_text(encoding="utf-8"))


def verify_shutdown(process, log_path, devices):
    initial_pid, retry_pid, allocated = wait_for_retry(process, log_path)
    assert len({process.pid, initial_pid, retry_pid}) == 3
    assert set(allocated) == set(devices) and all(size > 0 for size in allocated.values()), allocated
    retry_fd = os.pidfd_open(retry_pid)
    retry_exit = select.poll()
    retry_exit.register(retry_fd, select.POLLIN)
    try:
        measured = process_vram(retry_pid, devices)
        before = {pci: measured.get(pci) for pci in devices}
        accounted = all(used is not None and used >= allocated[pci] for pci, used in before.items())
        assert not retry_exit.poll(0) and process.poll() is None, log_path.read_text(encoding="utf-8")

        os.kill(process.pid, signal.SIGKILL)
        assert process.wait(timeout=5) == -signal.SIGKILL
        assert retry_exit.poll(3000), f"CUDA retry {retry_pid} survived Monolith SIGKILL"

        after = None
        if accounted:
            deadline = time.monotonic() + 5
            while True:
                remaining = {pid: process_vram(pid, devices) for pid in (process.pid, initial_pid, retry_pid)}
                if not any(remaining.values()):
                    after = {pid: dict.fromkeys(devices, 0) for pid in remaining}
                    break
                assert time.monotonic() < deadline, remaining
                time.sleep(0.05)
        return {
            "monolith_pid": process.pid, "monolith_exit_code": process.returncode,
            "initial_probe_pid": initial_pid, "retry_pid": retry_pid, "retry_exited": True,
            "cuda_allocated_bytes": allocated, "nvml_probe_vram_before_bytes": before,
            "nvml_vram_after_bytes": after, "nvml_pid_accounting_verified": accounted,
        }
    finally:
        if not retry_exit.poll(0):
            try:
                signal.pidfd_send_signal(retry_fd, signal.SIGKILL)
            except ProcessLookupError:
                pass
            retry_exit.poll(3000)
        os.close(retry_fd)


def test(exe, assets, output, fault_library):
    output.mkdir(parents=True, exist_ok=True)
    image = exe.read_bytes()
    assert image[:6] == bytes.fromhex("7f454c460201"), exe
    assert not exe.with_name(exe.name + ".dll").exists(), "Use the Native AOT publication"
    standalone = runpy.run_path(str(Path(__file__).with_name("native-aot-standalone.py")))
    devices = {pci_key(bus): name for bus, name in nvidia_smi("--query-gpu=pci.bus_id,name")}
    assert len(devices) == 2, devices
    assert sum("RTX 4000" in name for name in devices.values()) == 1, devices
    assert sum("T1000" in name for name in devices.values()) == 1, devices

    with tempfile.TemporaryDirectory(prefix="admission-vram-linux-") as temp:
        work = Path(temp)
        mounts = {name: work / name for name in ("config", "model", "vfs", "capability")}
        for mount in mounts.values():
            mount.mkdir()
        standalone["fixtures"](assets, mounts["model"])
        (mounts["model"] / "broken-tokenizer.json").unlink()
        models = [{"id": "local", "provider": "providers.paged-llama",
                   "policy": {"logos": "approve", "ethos": "approve", "pathos": "approve"},
                   "local": {"checkpoint": "stories.gguf", "device": 0, "blockSize": 4096,
                             "ramBlocks": 4, "vramBlocks": 2, "kvRamBlocks": 1, "kvVramBlocks": 1,
                             "contextTokens": 64, "maxNewTokens": 8}}]
        (mounts["model"] / "models.json").write_text(json.dumps(models), encoding="utf-8")

        binary = work / exe.name
        shutil.copyfile(exe, binary)
        binary.chmod(0o755)
        real_driver = work / "libaikernel_cuda_driver.real.so"
        shutil.copyfile(exe.with_name("libaikernel_cuda_driver.so"), real_driver)
        shutil.copyfile(fault_library, work / "libaikernel_cuda_driver.so")
        env = dict(os.environ, AIKERNEL_API_KEY=standalone["KEY"],
                   AIKERNEL_TEST_REAL_CUDA_LIBRARY=str(real_driver), AIKERNEL_TEST_PROBE_MARKER=str(work / "probe"),
                   Logging__LogLevel__Microsoft__Hosting__Lifetime="Information")
        env["Logging__LogLevel__AIKernel.Monolith.Models.CudaTopology"] = "Debug"
        log_path = output / "monolith.log"
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                [str(binary), "--urls=http://127.0.0.1:0",
                 *[f"--{name}-root={mount}" for name, mount in mounts.items()]],
                cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                report = verify_shutdown(process, log_path, devices)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)

    report["native_sha256"] = hashlib.sha256(image).hexdigest()
    report["devices"] = devices
    encoded = json.dumps(report, indent=2)
    (output / "admission-vram-result.json").write_text(encoded + "\n", encoding="utf-8")
    print(encoded, flush=True)
    assert report["nvml_pid_accounting_verified"], "NVML did not account for the live CUDA allocation by Linux PID"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("exe", "assets", "output", "fault_library"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if not sys.platform.startswith("linux") or not hasattr(os, "pidfd_open"):
        parser.error("Linux with Python and kernel pidfd support is required")
    test(args.exe.resolve(), args.assets.resolve(), args.output.resolve(), args.fault_library.resolve())
