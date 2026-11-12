"""Run a bounded CUDA workload so the thermal regression can use live NVML readings."""
import argparse
import ctypes as c
import json
import os
from pathlib import Path
import time


def check(call, *args):
    status = call(*args)
    if status:
        raise RuntimeError(f"{call.__name__}: {status}")


def run_heat(ordinal, pci_bus_id, threshold, seconds, arm_file, ready_file, stop_file, preheat, preheat_celsius):
    ptx = b"""
.version 7.0
.target sm_75
.address_size 64
.visible .entry warm(.param .u64 output) {
    .reg .pred p;
    .reg .b32 thread, block, idx, i;
    .reg .b64 ptr, offset;
    .reg .f32 a, b, c, d, e, f, g, h;
    mov.u32 thread, %tid.x;
    mov.u32 block, %ctaid.x;
    mad.lo.u32 idx, block, 128, thread;
    mov.f32 a, 0f3F400000;
    cvt.rn.f32.u32 b, idx;
    mov.f32 c, 0f3F300000;
    mov.f32 d, 0f3E880000;
    mov.f32 e, 0f3F180000;
    mov.f32 f, 0f3E180000;
    mov.f32 g, 0f3F200000;
    mov.f32 h, 0f3E200000;
    mov.u32 i, 16384;
again:
    fma.rn.f32 a, a, 0f3F7FF001, 0f3A100001;
    fma.rn.f32 b, b, 0f3F7FF002, 0f3A100002;
    fma.rn.f32 c, c, 0f3F7FF003, 0f3A100003;
    fma.rn.f32 d, d, 0f3F7FF004, 0f3A100004;
    fma.rn.f32 e, e, 0f3F7FF005, 0f3A100005;
    fma.rn.f32 f, f, 0f3F7FF006, 0f3A100006;
    fma.rn.f32 g, g, 0f3F7FF007, 0f3A100007;
    fma.rn.f32 h, h, 0f3F7FF008, 0f3A100008;
    sub.u32 i, i, 1;
    setp.ne.u32 p, i, 0;
    @p bra again;
    add.f32 a, a, b;
    add.f32 c, c, d;
    add.f32 e, e, f;
    add.f32 g, g, h;
    add.f32 a, a, c;
    add.f32 e, e, g;
    add.f32 a, a, e;
    ld.param.u64 ptr, [output];
    mul.wide.u32 offset, idx, 4;
    add.u64 ptr, ptr, offset;
    st.global.f32 [ptr], a;
    ret;
}

.visible .entry traffic(.param .u64 output) {
    .reg .b32 thread, block, idx, k;
    .reg .b64 ptr, offset;
    .reg .f32 a,b,c,d;
    mov.u32 thread, %tid.x;
    mov.u32 block, %ctaid.x;
    mad.lo.u32 idx, block, 128, thread;
    ld.param.u64 ptr, [output];
    mul.wide.u32 offset, idx, 16;
    add.u64 ptr, ptr, offset;
    .reg .pred p;
    mov.u32 k, 0;
next_vec:
    ld.global.cg.v4.f32 {a,b,c,d}, [ptr];
    fma.rn.f32 a,a,0f3F7FF101,0f3A100001;
    fma.rn.f32 b,b,0f3F7FF102,0f3A100002;
    fma.rn.f32 c,c,0f3F7FF103,0f3A100003;
    fma.rn.f32 d,d,0f3F7FF104,0f3A100004;
    st.global.wt.v4.f32 [ptr], {a,b,c,d};
    add.u64 ptr, ptr, 8388608;
    add.u32 k, k, 1;
    setp.lt.u32 p, k, 16;
    @p bra next_vec;
    ret;
}
"""
    system = Path(os.environ["SystemRoot"]) / "System32"
    nv = c.WinDLL(str(system / "nvml.dll"))
    drv = c.WinDLL(str(system / "nvcuda.dll"))
    ctx, module, fn, handle = (c.c_void_p() for _ in range(4))
    dev, output = c.c_int(), c.c_uint64()
    pci = c.create_string_buffer(32)
    check(nv.nvmlInit_v2)
    try:
        check(drv.cuInit, 0)
        check(drv.cuDeviceGet, c.byref(dev), ordinal)
        check(drv.cuDeviceGetPCIBusId, pci, len(pci), dev)
        if pci.value.decode().lower() != pci_bus_id.lower():
            raise RuntimeError(f"Unexpected PCI bus: {pci.value}")
        check(nv.nvmlDeviceGetHandleByPciBusId_v2, pci, c.byref(handle))
        check(drv.cuCtxCreate_v2, c.byref(ctx), 0, dev)
        check(drv.cuModuleLoadData, c.byref(module), c.c_char_p(ptx))
        check(drv.cuModuleGetFunction, c.byref(fn), module, b"warm")
        check(drv.cuMemAlloc_v2, c.byref(output), c.c_size_t(512 * 128 * 4))
        params = (c.c_void_p * 1)(c.addressof(output))
        traffic_fn, traffic_data, compute_stream, memory_stream = c.c_void_p(), c.c_uint64(), c.c_void_p(), c.c_void_p()
        if preheat:
            check(drv.cuModuleGetFunction, c.byref(traffic_fn), module, b"traffic")
            check(drv.cuMemAlloc_v2, c.byref(traffic_data), c.c_size_t(128 * 1024 * 1024))
            check(drv.cuMemsetD8_v2, traffic_data, 0, c.c_size_t(128 * 1024 * 1024))
            check(drv.cuStreamCreate, c.byref(compute_stream), 1)
            check(drv.cuStreamCreate, c.byref(memory_stream), 1)
        traffic_params = (c.c_void_p * 1)(c.addressof(traffic_data))
        temp = c.c_uint()
        if ready_file:
            ready_file.touch()
        start = time.monotonic()
        active_at = start if arm_file is None else None
        while time.monotonic() - start < 600:
            if stop_file and stop_file.exists():
                break
            now = time.monotonic()
            if active_at is None and arm_file.exists():
                active_at = now
            if active_at is not None and now - active_at >= seconds:
                break
            check(nv.nvmlDeviceGetTemperature, handle, 0, c.byref(temp))
            elapsed = now - start
            active = active_at is not None
            print(json.dumps({"s": round(elapsed, 3), "active": active, "temperature": temp.value}), flush=True)
            if temp.value >= 84:
                break
            if not active and (not preheat or temp.value >= preheat_celsius):
                time.sleep(0.18)
                continue
            if active and (temp.value >= 83 or not preheat and (now - active_at) % 3.0 > 2.2):
                time.sleep(0.18)
                continue
            batch_end = time.monotonic() + 0.38
            while time.monotonic() < batch_end:
                for _ in range(8):
                    check(drv.cuLaunchKernel, fn, 512, 1, 1, 128, 1, 1, 0, compute_stream, params, None)
                    if preheat:
                        check(drv.cuLaunchKernel, traffic_fn, 4096, 1, 1, 128, 1, 1, 0, memory_stream, traffic_params, None)
                check(drv.cuCtxSynchronize)
    finally:
        if ctx.value:
            check(drv.cuCtxDestroy_v2, ctx)
        check(nv.nvmlShutdown)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ordinal", type=int, default=0)
    p.add_argument("--pci", required=True)
    p.add_argument("--threshold", type=int, required=True)
    p.add_argument("--seconds", type=int, default=45)
    p.add_argument("--arm-file", type=Path)
    p.add_argument("--ready-file", type=Path)
    p.add_argument("--stop-file", type=Path)
    p.add_argument("--preheat", action="store_true")
    p.add_argument("--preheat-celsius", type=int)
    a = p.parse_args()
    if not 0 < a.seconds <= 180 or not 3 <= a.threshold <= 82:
        p.error("workload duration or thermal threshold is out of range")
    preheat_celsius = a.threshold - 2 if a.preheat_celsius is None else a.preheat_celsius
    if not 0 <= preheat_celsius < a.threshold:
        p.error("preheat temperature must be below the running threshold")
    run_heat(a.ordinal, a.pci, a.threshold, a.seconds, a.arm_file, a.ready_file, a.stop_file, a.preheat, preheat_celsius)
