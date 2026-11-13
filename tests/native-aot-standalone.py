"""Run mapped GGUF/SafeTensors inference on CUDA and in a CPU-only topology."""
import argparse
from contextlib import closing
import hashlib
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import runpy
import struct
import subprocess
import tempfile


helpers = runpy.run_path(str(Path(__file__).with_name("native-aot-stream-embeddings.py")))
KEY = "standalone-native-test"
PROMPT = "Once upon a time"
EXPECTED = ", there was a little girl"


def fixtures(assets, models):
    checkpoint = (assets / "stories260K.bin").read_bytes()
    tokenizer = (assets / "tok512.bin").read_bytes()
    assert hashlib.sha256(checkpoint).hexdigest() == "b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696"
    width, hidden, layers, heads, kvheads, vocab, context = struct.unpack_from("<7i", checkpoint)
    kvwidth = width // heads * kvheads
    tokens, scores = [], []
    offset = 4
    for _ in range(vocab):
        score, length = struct.unpack_from("<fi", tokenizer, offset)
        offset += 8
        tokens.append(tokenizer[offset:offset + length].decode())
        scores.append(score)
        offset += length

    tensors = []
    offset = 28

    def add(gguf, safe, rows, cols, permute=False):
        nonlocal offset
        size = rows * cols * 4
        tensors.append((gguf, safe, rows, cols, checkpoint[offset:offset + size], permute))
        offset += size

    def layer(gguf, safe, rows, cols, permute=False):
        for i in range(layers):
            add(f"blk.{i}.{gguf}.weight", f"model.layers.{i}.{safe}.weight", rows, cols, permute)

    add("token_embd.weight", "model.embed_tokens.weight", vocab, width)
    layer("attn_norm", "input_layernorm", 1, width)
    layer("attn_q", "self_attn.q_proj", width, width, True)
    layer("attn_k", "self_attn.k_proj", kvwidth, width, True)
    layer("attn_v", "self_attn.v_proj", kvwidth, width)
    layer("attn_output", "self_attn.o_proj", width, width)
    layer("ffn_norm", "post_attention_layernorm", 1, width)
    layer("ffn_gate", "mlp.gate_proj", hidden, width)
    layer("ffn_down", "mlp.down_proj", width, hidden)
    layer("ffn_up", "mlp.up_proj", hidden, width)
    add("output_norm.weight", "model.norm.weight", 1, width)

    def text(value):
        encoded = value.encode()
        return struct.pack("<Q", len(encoded)) + encoded

    meta = {
        "general.architecture": (8, text("llama")),
        "llama.embedding_length": (4, struct.pack("<I", width)),
        "llama.feed_forward_length": (4, struct.pack("<I", hidden)),
        "llama.block_count": (4, struct.pack("<I", layers)),
        "llama.attention.head_count": (4, struct.pack("<I", heads)),
        "llama.attention.head_count_kv": (4, struct.pack("<I", kvheads)),
        "llama.context_length": (4, struct.pack("<I", context)),
        "tokenizer.ggml.model": (8, text("llama")),
        "tokenizer.ggml.tokens": (9, struct.pack("<IQ", 8, vocab) + b"".join(map(text, tokens))),
        "tokenizer.ggml.scores": (9, struct.pack("<IQ", 6, vocab) + struct.pack(f"<{vocab}f", *scores)),
    }

    def gguf(name, invalid_type=False):
        header = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(meta)))
        for key, (kind, value) in meta.items():
            header += text(key) + struct.pack("<I", kind) + value
        data = bytearray()
        for index, (key, _, rows, cols, raw, _) in enumerate(tensors):
            data += bytes(-len(data) % 32)
            shape = [cols] if rows == 1 else [cols, rows]
            header += text(key) + struct.pack("<I", len(shape)) + struct.pack(f"<{len(shape)}Q", *shape)
            header += struct.pack("<IQ", 99 if invalid_type and index == 0 else 0, len(data))
            data += raw
        header += bytes(-len(header) % 32)
        (models / name).write_bytes(header + data)

    gguf("stories.gguf")
    gguf("unsupported.gguf", invalid_type=True)
    (models / "corrupt.gguf").write_bytes((models / "stories.gguf").read_bytes()[:-32])

    safe_header, data = {}, bytearray()
    head_size = width // heads
    for _, key, rows, cols, raw, permute in tensors:
        if permute:
            stride = cols * 4
            order = [start + i for start in range(0, rows, head_size)
                     for parity in (0, 1) for i in range(parity, head_size, 2)]
            raw = b"".join(raw[row * stride:(row + 1) * stride] for row in order)
        vals = struct.unpack(f"<{rows * cols}f", raw)
        half = struct.pack(f"<{len(vals)}e", *vals)
        safe_header[key] = {"dtype": "F16", "shape": [cols] if rows == 1 else [rows, cols],
                            "data_offsets": [len(data), len(data) + len(half)]}
        data += half
    encoded = json.dumps(safe_header).encode()
    encoded += b" " * (-len(encoded) % 8)
    (models / "stories.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
    (models / "config.json").write_text(json.dumps({
        "model_type": "llama", "hidden_size": width, "intermediate_size": hidden, "num_hidden_layers": layers,
        "num_attention_heads": heads, "num_key_value_heads": kvheads, "vocab_size": vocab,
        "max_position_embeddings": context,
    }), encoding="utf-8")

    def varint(value):
        encoded = bytearray()
        while value >= 128:
            encoded.append((value & 127) | 128)
            value >>= 7
        encoded.append(value)
        return encoded

    def message(field, payload):
        return varint(field * 8 + 2) + varint(len(payload)) + payload

    sentencepiece = bytearray(message(2, bytes([24, 2])))
    for piece, score in zip(tokens, scores):
        sentencepiece += message(1, message(1, piece.encode()) + bytes([21]) + struct.pack("<f", score))
    (models / "tokenizer.model").write_bytes(sentencepiece)
    (models / "broken-tokenizer.json").write_text("{", encoding="utf-8")

    vocabulary = {piece.replace(" ", "▁"): token for token, piece in enumerate(tokens)}
    merge = []
    for piece, token in vocabulary.items():
        for index in range(1, len(piece)):
            if piece[:index] in vocabulary and piece[index:] in vocabulary:
                merge.append((scores[token], [piece[:index], piece[index:]]))
    merge.sort(key=lambda item: -item[0])
    (models / "tokenizer.json").write_text(json.dumps({
        "model": {"type": "BPE", "vocab": vocabulary, "merges": [pair for _, pair in merge]},
    }), encoding="utf-8")



def device_capacity(ordinal):
    import ctypes

    driver = ctypes.WinDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so.1")
    device = ctypes.c_int()
    capacity = ctypes.c_size_t()
    assert driver.cuInit(0) == 0
    assert driver.cuDeviceGet(ctypes.byref(device), ordinal) == 0
    assert driver.cuDeviceTotalMem_v2(ctypes.byref(capacity), device) == 0
    return capacity.value


def oversized_fixture(path, capacity):
    # Every matrix participates in forward propagation. Unspecified weights are zero,
    # so the sparse file occupies little disk space while exercising real page faults.
    width, hidden, heads, vocab = 8192, 45056, 32, 5
    kvwidth = width // heads
    layers = capacity // (4 * (2 * width * width + 2 * kvwidth * width + 3 * hidden * width)) + 1
    assert layers <= 96

    def text(value):
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw

    integers = {"llama.embedding_length": width, "llama.feed_forward_length": hidden,
                "llama.block_count": layers, "llama.attention.head_count": heads,
                "llama.attention.head_count_kv": 1, "llama.context_length": 8}
    metadata = [(k, 4, struct.pack("<I", v)) for k, v in integers.items()]
    metadata += [
        ("general.architecture", 8, text("llama")),
        ("tokenizer.ggml.model", 8, text("llama")),
        ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, vocab) +
         b"".join(map(text, ("<unk>", "<s>", "</s>", "a", "x")))),
        ("tokenizer.ggml.scores", 9, struct.pack("<IQ", 6, vocab) + bytes(vocab * 4)),
    ]
    tensors = [("token_embd.weight", vocab, width, 3), ("output_norm.weight", 1, width, 0),
               ("output.weight", vocab, width, 4)]
    for layer in range(layers):
        for suffix, rows, cols in [
            ("attn_norm", 1, width), ("attn_q", width, width),
            ("attn_k", kvwidth, width), ("attn_v", kvwidth, width),
            ("attn_output", width, width), ("ffn_norm", 1, width),
            ("ffn_gate", hidden, width), ("ffn_up", hidden, width),
            ("ffn_down", width, hidden),
        ]:
            tensors.append((f"blk.{layer}.{suffix}.weight", rows, cols, None))

    header = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(metadata)))
    for key, kind, value in metadata:
        header += text(key) + struct.pack("<I", kind) + value
    offset, nonzero = 0, []
    for name, rows, cols, active_row in tensors:
        shape = [cols] if rows == 1 else [cols, rows]
        header += text(name) + struct.pack("<I", len(shape)) + struct.pack(f"<{len(shape)}Q", *shape)
        header += struct.pack("<IQ", 0, offset)
        if active_row is not None:
            nonzero.append(offset + active_row * cols * 4)
        offset += rows * cols * 4
    header += bytes(-len(header) % 32)
    with path.open("wb") as target:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            import msvcrt

            control = ctypes.WinDLL("kernel32", use_last_error=True).DeviceIoControl
            control.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                                wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                                wintypes.LPVOID]
            returned = wintypes.DWORD()
            if not control(msvcrt.get_osfhandle(target.fileno()), 0x900C4, None, 0, None, 0,
                           ctypes.byref(returned), None):
                raise ctypes.WinError(ctypes.get_last_error())
        target.write(header)
        target.seek(len(header) + offset - 1)
        target.write(bytes(1))
        for position in nonzero:
            target.seek(len(header) + position)
            target.write(struct.pack("<f", 1))
    assert path.stat().st_size > capacity


def smoke(exe, assets, output, cpu_only, oversized_device):
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="standalone-", dir=output) as temp:
        work = Path(temp)
        roots = {name: work / name for name in ("config", "model", "vfs", "capability")}
        for directory in roots.values():
            directory.mkdir()
        fixtures(assets, roots["model"])

        def registration(alias, checkpoint, ordinal, **extras):
            return {"id": alias, "provider": "providers.paged-llama", "local": {
                "checkpoint": checkpoint, "blockSize": 4096, "ramBlocks": 4, "vramBlocks": 2,
                "device": ordinal, "contextTokens": 64, "maxNewTokens": 8,
                "kvRamBlocks": 1, "kvVramBlocks": 1, **extras},
                "policy": {"logos": "approve", "ethos": "approve", "pathos": "approve"}}

        models = [registration("gpu-gguf", "stories.gguf", 0),
                  registration("cpu-gguf", "stories.gguf", 2147483647),
                  registration("safe", "stories.safetensors", 0, config="config.json", tokenizer="tokenizer.json"),
                  registration("top-p", "stories.gguf", 0, temperature=0.7, topP=0.0001, tokenizer="tokenizer.model"),
                  registration("bad-tokenizer", "stories.gguf", 0, tokenizer="broken-tokenizer.json"),
                  registration("unknown", "unsupported.gguf", 0),
                  registration("corrupt", "corrupt.gguf", 0)]
        capacity = None
        oversized_block_size = 32 * 1024 * 1024
        if oversized_device is not None:
            capacity = device_capacity(oversized_device)
            oversized_fixture(roots["model"] / "oversized.gguf", capacity)
            models.append(registration("oversized", "oversized.gguf", oversized_device,
                                       blockSize=oversized_block_size, maxNewTokens=1, contextTokens=8))
        (roots["model"] / "models.json").write_text(json.dumps(models), encoding="utf-8")
        env = {k: v for k, v in os.environ.items()
               if not k.upper().startswith(("CUDA_PATH", "CUDA_HOME", "LIBTORCH", "TORCH", "HF_"))}
        env["PATH"] = os.pathsep.join(p for p in env.get("PATH", "").split(os.pathsep)
                                      if "NVIDIA GPU Computing Toolkit" not in p)
        env.update(AIKERNEL_API_KEY=KEY, AIKERNEL_API_REQUEST_LIMIT="1000",
                   Logging__LogLevel__Microsoft__Hosting__Lifetime="Information")
        log_path = output / "standalone-monolith.log"
        audit_path = roots["config"] / "inference-audit.jsonl"
        with log_path.open("wb") as log:
            process = subprocess.Popen([str(exe), "--urls=http://127.0.0.1:0",
                                        *[f"--{key}-root={value}" for key, value in roots.items()]],
                                       cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                port = helpers["wait_for_port"](process, log_path)

                if cpu_only:
                    with closing(HTTPConnection("127.0.0.1", port, timeout=10)) as connection:
                        connection.request("GET", "/v1/models/admission?model=gpu-gguf",
                                           headers={"Authorization": f"Bearer {KEY}"})
                        response = connection.getresponse()
                        admission = json.loads(response.read())
                        assert response.status == 200, admission
                        assert admission["cuda_topology_state"] == "no_devices", admission
                        assert admission["admissible"] and admission["backend"] == "cpu", admission
                        assert admission["retry_after_seconds"] is None and admission["devices"] == [], admission

                def request(model, status=200, prompt=PROMPT, timeout=150):
                    body = json.dumps({"model": model, "prompt": prompt})
                    with closing(HTTPConnection("127.0.0.1", port, timeout=timeout)) as connection:
                        connection.request("POST", "/v1/completions", body,
                                           {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
                        response = connection.getresponse()
                        data = response.read()
                        assert response.status == status, (model, response.status, data)
                        return json.loads(data)

                assert request("unknown", 422)["error"]["code"] == "unsupported_model_format"
                assert request("corrupt", 502)["error"]["code"] == "provider_error"
                assert request("bad-tokenizer", 502)["error"]["code"] == "provider_error"
                results = {}
                default_backend = "cpu" if cpu_only else "cuda"
                profiles = [("gpu-gguf", default_backend), ("cpu-gguf", "cpu"),
                            ("safe", default_backend), ("top-p", default_backend)]
                for model, backend in profiles:
                    response = request(model)
                    text = response["choices"][0]["text"]
                    assert text == EXPECTED, (model, text)
                    entries = helpers["audit_entries"](audit_path)
                    entry = next(e for e in entries if e["EventType"] == "inference.execution"
                                 and e["Metadata"]["completion_id"] == response["id"])
                    counters = entry["Metadata"]
                    assert entry["Description"] == "succeeded" and counters["backend"] == backend
                    if cpu_only:
                        assert counters["cuda_status"] == "2", counters
                    assert int(counters["mapped_weight_reads"]) > 0
                    assert int(counters["kv_swap_reads"]) > 0
                    assert int(counters["generation_gen2_collections"]) == 0
                    assert int(counters[f"{backend}_matvec_blocks"]) > 0
                    results[model] = {"text": text, **counters}
                report = {"results": results, "unsupported_format_rejected": True, "corrupt_model_rejected": True,
                          "corrupt_tokenizer_rejected": True}
                if capacity is not None:
                    response = request("oversized", prompt="a", timeout=900)
                    text = response["choices"][0]["text"]
                    assert text == "x", text
                    entry = next(e for e in helpers["audit_entries"](audit_path)
                                 if e["EventType"] == "inference.execution"
                                 and e["Metadata"]["completion_id"] == response["id"])
                    counters = entry["Metadata"]
                    assert counters["backend"] == "cuda", counters
                    assert int(counters["checkpoint_bytes"]) > capacity
                    assert int(counters["mapped_weight_reads"]) * oversized_block_size > capacity
                    assert int(counters["kv_swap_reads"]) > 0
                    assert int(counters["generation_gen2_collections"]) == 0, counters
                    report["oversized"] = {"text": text, "physical_vram_bytes": capacity, **counters}
                (output / "standalone-result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                print(json.dumps(report, indent=2))
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe", type=Path)
    parser.add_argument("assets", type=Path)
    parser.add_argument("output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cpu-only", action="store_true", help="Assert CUDA driver is absent.")
    mode.add_argument("--oversized-device", type=int, help="Test a synthetic checkpoint larger than this CUDA device.")
    args = parser.parse_args()
    smoke(args.exe.resolve(), args.assets.resolve(), args.output.resolve(), args.cpu_only, args.oversized_device)
