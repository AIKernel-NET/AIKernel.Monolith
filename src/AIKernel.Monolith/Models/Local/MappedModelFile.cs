using System.Buffers.Binary;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using AIKernel.Kernel.Memory;

namespace AIKernel.Monolith.Models.Local;

internal enum TensorEncoding { F32, F16, BF16, Q4_0, MxFp4, Unsupported }
internal sealed record TensorRegion(string Name, long Offset, int[] Shape, TensorEncoding Encoding);
internal sealed record ModelDimensions(int Width, int Hidden, int Layers, int Heads, int KvHeads,
    int Vocab, int Context, float RopeBase, float Epsilon, bool SplitRope);

internal sealed unsafe class MappedModelFile : IDisposable
{
    private static readonly UTF8Encoding Utf8 = new(false, true);
    private readonly MemoryMappedFile map;
    private readonly MemoryMappedViewAccessor view;
    private readonly Dictionary<string, TensorRegion> tensors = new(StringComparer.Ordinal);
    private byte* address;

    internal ModelDimensions Dimensions { get; private set; } = null!;
    internal string[]? Tokens { get; private set; }
    internal float[]? Scores { get; private set; }
    internal bool Gguf { get; }
    internal long Length { get; }

    internal MappedModelFile(string path, string? config)
    {
        using var file = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        Length = file.Length;
        if (Length < 16 || !BitConverter.IsLittleEndian)
        {
            throw new InvalidDataException("Expected a nonempty little-endian model file.");
        }

        Gguf = Path.GetExtension(path).Equals(".gguf", StringComparison.OrdinalIgnoreCase);
        map = MemoryMappedFile.CreateFromFile(file, null, 0, MemoryMappedFileAccess.Read,
            HandleInheritability.None, leaveOpen: false);
        try
        {
            view = map.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            try
            {
                view.SafeMemoryMappedViewHandle.AcquirePointer(ref address);
                using var reader = new BinaryReader(map.CreateViewStream(0, 0, MemoryMappedFileAccess.Read), Utf8);
                if (Gguf) ReadGguf(reader);
                else ReadSafeTensors(reader, config);
            }
            catch
            {
                if (address != null) view.SafeMemoryMappedViewHandle.ReleasePointer();
                view.Dispose();
                throw;
            }
        }
        catch
        {
            map.Dispose();
            throw;
        }
    }

    internal TensorRegion Require(string name, int rows, int columns)
    {
        if (!tensors.TryGetValue(name, out var tensor) ||
            (tensor.Shape.Length != 1 || rows != 1 || tensor.Shape[0] != columns) &&
            (tensor.Shape.Length != 2 || tensor.Shape[0] != rows || tensor.Shape[1] != columns))
        {
            throw new InvalidDataException($"Invalid or missing tensor {name}.");
        }

        var size = (long)rows * RowBytes(tensor.Encoding, columns);
        if (tensor.Offset < 0 || tensor.Offset > Length || size > Length - tensor.Offset)
        {
            throw new InvalidDataException($"Tensor {name} extends outside the mapped file.");
        }

        return tensor;
    }

    internal bool Contains(string name) => tensors.ContainsKey(name);

    internal ReadOnlySpan<byte> Slice(long offset, int length)
    {
        ObjectDisposedException.ThrowIf(address == null, this);
        if (offset < 0 || length < 0 || offset > Length || length > Length - offset)
        {
            throw new InvalidDataException("Mapped tensor read exceeded its file bounds.");
        }

        return new ReadOnlySpan<byte>(address + view.PointerOffset + offset, length);
    }

    internal static int RowBytes(TensorEncoding encoding, int columns) => encoding switch
    {
        TensorEncoding.F32 => checked(columns * 4),
        TensorEncoding.F16 or TensorEncoding.BF16 => checked(columns * 2),
        TensorEncoding.Q4_0 when columns % 32 == 0 => columns / 32 * 18,
        TensorEncoding.MxFp4 when columns % 32 == 0 => columns / 32 * 17,
        _ => throw new UnsupportedModelException("Unsupported tensor encoding or quantized row width.")
    };

    private void ReadGguf(BinaryReader reader)
    {
        if (reader.ReadUInt32() != 0x46554747 || reader.ReadUInt32() is not (2 or 3))
        {
            throw new UnsupportedModelException("Expected a GGUF v2 or v3 file.");
        }

        var count = Limit(reader.ReadUInt64(), 65536);
        var metaCount = Limit(reader.ReadUInt64(), 65536);
        var metadata = new Dictionary<string, object>(StringComparer.Ordinal);
        for (var i = 0; i < metaCount; i++)
        {
            var name = ReadString(reader);
            var type = reader.ReadUInt32();
            if (name is "tokenizer.ggml.tokens" or "tokenizer.ggml.scores")
            {
                if (type != 9) throw new InvalidDataException("Invalid GGUF vocabulary array.");
                var itemType = reader.ReadUInt32();
                var length = Limit(reader.ReadUInt64(), 200000);
                if (name == "tokenizer.ggml.tokens" && itemType == 8)
                    Tokens = Enumerable.Range(0, length).Select(_ => ReadString(reader)).ToArray();
                else if (name == "tokenizer.ggml.scores" && itemType == 6)
                    Scores = Enumerable.Range(0, length).Select(_ => reader.ReadSingle()).ToArray();
                else throw new InvalidDataException("Invalid GGUF vocabulary data.");
            }
            else if (type == 9)
            {
                var itemType = reader.ReadUInt32();
                var length = Limit(reader.ReadUInt64(), 1000000);
                for (var item = 0; item < length; item++) SkipValue(reader, itemType);
            }
            else if (name.StartsWith("llama.", StringComparison.Ordinal) ||
                     name is "general.architecture" or "general.alignment" or "tokenizer.ggml.model" or
                         "tokenizer.ggml.bos_token_id" or "tokenizer.ggml.eos_token_id")
            {
                metadata.Add(name, ReadScalar(reader, type));
            }
            else
            {
                SkipValue(reader, type);
            }
        }

        if (metadata.GetValueOrDefault("general.architecture") is not "llama" ||
            metadata.Keys.Any(k => k.StartsWith("llama.rope.scaling.", StringComparison.Ordinal)))
        {
            throw new UnsupportedModelException("Expected an unscaled Llama GGUF architecture.");
        }

        var alignment = Integer("general.alignment", 32);
        if (alignment is < 1 or > 4096 || (alignment & (alignment - 1)) != 0)
            throw new InvalidDataException("Invalid GGUF alignment.");
        for (var i = 0; i < count; i++)
        {
            var name = ReadString(reader);
            var rank = Limit(reader.ReadUInt32(), 4);
            if (rank == 0) throw new InvalidDataException("Empty GGUF tensor shape.");
            var shape = new int[rank];
            for (var dim = rank - 1; dim >= 0; dim--) shape[dim] = Limit(reader.ReadUInt64(), int.MaxValue);
            var encoding = reader.ReadUInt32() switch
            {
                0 => TensorEncoding.F32, 1 => TensorEncoding.F16, 2 => TensorEncoding.Q4_0,
                30 => TensorEncoding.BF16, 39 => TensorEncoding.MxFp4, _ => TensorEncoding.Unsupported
            };
            var offset = checked((long)reader.ReadUInt64());
            tensors.Add(name, new TensorRegion(name, offset, shape, encoding));
        }

        var data = checked((reader.BaseStream.Position + alignment - 1) & ~(alignment - 1L));
        foreach (var tensor in tensors.Values.ToArray())
        {
            if (tensor.Offset % alignment != 0)
                throw new InvalidDataException("Misaligned GGUF tensor.");
            tensors[tensor.Name] = tensor with { Offset = checked(data + tensor.Offset) };
        }

        var width = Integer("llama.embedding_length");
        var heads = Integer("llama.attention.head_count");
        if (heads <= 0 || Integer("llama.rope.dimension_count", width / heads) != width / heads)
            throw new UnsupportedModelException("Partial rotary embeddings are unsupported.");
        if (!tensors.TryGetValue("token_embd.weight", out var embedding) || embedding.Shape.Length != 2)
            throw new InvalidDataException("Missing GGUF embedding tensor.");
        var vocab = embedding.Shape[0];
        if (metadata.GetValueOrDefault("tokenizer.ggml.model") is not "llama" ||
            Integer("tokenizer.ggml.bos_token_id", 1) != 1 || Integer("tokenizer.ggml.eos_token_id", 2) != 2)
        {
            Tokens = null;
            Scores = null;
        }

        Dimensions = new(width, Integer("llama.feed_forward_length"), Integer("llama.block_count"),
            heads, Integer("llama.attention.head_count_kv", heads), vocab, Integer("llama.context_length"),
            Floating("llama.rope.freq_base", 10000), Floating("llama.attention.layer_norm_rms_epsilon", 1e-5f), false);

        int Integer(string key, int fallback = -1) => metadata.TryGetValue(key, out var value)
            ? Convert.ToInt32(value, System.Globalization.CultureInfo.InvariantCulture) : fallback;
        float Floating(string key, float fallback) => metadata.TryGetValue(key, out var value)
            ? Convert.ToSingle(value, System.Globalization.CultureInfo.InvariantCulture) : fallback;
    }

    private void ReadSafeTensors(BinaryReader reader, string? config)
    {
        if (config is null) throw new UnsupportedModelException("SafeTensors requires a Llama config.json.");
        var headerSize = Limit(reader.ReadUInt64(), 64 * 1024 * 1024);
        using var header = JsonDocument.Parse(Slice(8, headerSize).ToArray());
        foreach (var member in header.RootElement.EnumerateObject())
        {
            if (member.NameEquals("__metadata__")) continue;
            var value = member.Value;
            var shape = value.GetProperty("shape").EnumerateArray().Select(n => n.GetInt32()).ToArray();
            var offsets = value.GetProperty("data_offsets");
            if (offsets.GetArrayLength() != 2) throw new InvalidDataException("Invalid SafeTensors offsets.");
            var start = offsets[0].GetInt64();
            var end = offsets[1].GetInt64();
            var data = 8L + headerSize;
            if (start < 0 || end < start || end > Length - data)
                throw new InvalidDataException("SafeTensors offset outside the file.");
            var encoding = value.GetProperty("dtype").GetString() switch
            {
                "F32" => TensorEncoding.F32, "F16" => TensorEncoding.F16, "BF16" => TensorEncoding.BF16,
                _ => TensorEncoding.Unsupported
            };
            if (encoding != TensorEncoding.Unsupported && shape.Length is 1 or 2 &&
                (long)(shape.Length == 2 ? shape[0] : 1) * RowBytes(encoding, shape[^1]) != end - start)
                throw new InvalidDataException("SafeTensors shape does not match its byte span.");
            tensors.Add(member.Name, new TensorRegion(member.Name, data + start, shape, encoding));
        }

        using var configFile = File.OpenRead(config);
        if (configFile.Length > 1024 * 1024) throw new InvalidDataException("Model config exceeds 1 MiB.");
        using var doc = JsonDocument.Parse(configFile);
        var root = doc.RootElement;
        if (root.GetProperty("model_type").GetString() != "llama" ||
            root.TryGetProperty("rope_scaling", out var scaling) && scaling.ValueKind != JsonValueKind.Null ||
            root.TryGetProperty("attention_bias", out var bias) && bias.GetBoolean())
            throw new UnsupportedModelException("Expected an unscaled Llama SafeTensors architecture.");
        var heads = root.GetProperty("num_attention_heads").GetInt32();
        Dimensions = new(root.GetProperty("hidden_size").GetInt32(), root.GetProperty("intermediate_size").GetInt32(),
            root.GetProperty("num_hidden_layers").GetInt32(), heads,
            root.TryGetProperty("num_key_value_heads", out var kv) ? kv.GetInt32() : heads,
            root.GetProperty("vocab_size").GetInt32(), root.GetProperty("max_position_embeddings").GetInt32(),
            root.TryGetProperty("rope_theta", out var rope) ? rope.GetSingle() : 10000,
            root.TryGetProperty("rms_norm_eps", out var eps) ? eps.GetSingle() : 1e-5f, true);
    }

    private static int Limit(ulong count, int max) => count <= (ulong)max ? (int)count :
        throw new InvalidDataException("Model metadata exceeds its supported size.");

    private static string ReadString(BinaryReader reader)
    {
        var length = Limit(reader.ReadUInt64(), 1024 * 1024);
        var data = reader.ReadBytes(length);
        if (data.Length != length) throw new EndOfStreamException();
        return Utf8.GetString(data);
    }

    private static object ReadScalar(BinaryReader reader, uint type) => type switch
    {
        0 => reader.ReadByte(), 1 => reader.ReadSByte(), 2 => reader.ReadUInt16(), 3 => reader.ReadInt16(),
        4 => reader.ReadUInt32(), 5 => reader.ReadInt32(), 6 => reader.ReadSingle(), 7 => reader.ReadBoolean(),
        8 => ReadString(reader), 10 => reader.ReadUInt64(), 11 => reader.ReadInt64(), 12 => reader.ReadDouble(),
        _ => throw new InvalidDataException("Invalid GGUF metadata type.")
    };

    private static void SkipValue(BinaryReader reader, uint type)
    {
        var length = type switch
        {
            0 or 1 or 7 => 1, 2 or 3 => 2, 4 or 5 or 6 => 4,
            8 => Limit(reader.ReadUInt64(), 1024 * 1024), 10 or 11 or 12 => 8,
            _ => throw new InvalidDataException("Invalid GGUF array element type.")
        };
        if (length > reader.BaseStream.Length - reader.BaseStream.Position) throw new EndOfStreamException();
        reader.BaseStream.Position += length;
    }

    public void Dispose()
    {
        if (address == null) return;
        view.SafeMemoryMappedViewHandle.ReleasePointer();
        address = null;
        view.Dispose();
        map.Dispose();
    }
}

internal sealed class MappedTensorSource(MappedModelFile file, WeightMatrix[] matrices) : IVirtualMemorySource
{
    public void ReadBlock(long blockIndex, Span<byte> destination)
    {
        var lower = 0;
        var upper = matrices.Length;
        while (lower + 1 < upper)
        {
            var middle = (lower + upper) / 2;
            if (matrices[middle].FirstBlock <= blockIndex) lower = middle;
            else upper = middle;
        }

        var tensor = matrices[lower];
        var row = checked((int)((blockIndex - tensor.FirstBlock) * tensor.RowsPerBlock));
        var count = Math.Min(tensor.RowsPerBlock, tensor.Rows - row);
        var rowBytes = MappedModelFile.RowBytes(tensor.Encoding, tensor.Columns);
        var input = file.Slice(tensor.Offset + (long)row * rowBytes, count * rowBytes);
        destination.Clear();
        if (tensor.Encoding == TensorEncoding.F32)
        {
            input.CopyTo(destination);
            return;
        }

        var output = MemoryMarshal.Cast<byte, float>(destination)[..(count * tensor.Columns)];
        if (tensor.Encoding is TensorEncoding.F16 or TensorEncoding.BF16)
        {
            for (var i = 0; i < output.Length; i++)
            {
                var bits = BinaryPrimitives.ReadUInt16LittleEndian(input[(2 * i)..]);
                output[i] = tensor.Encoding == TensorEncoding.F16
                    ? (float)BitConverter.UInt16BitsToHalf(bits) : BitConverter.Int32BitsToSingle(bits << 16);
            }

            return;
        }

        ReadOnlySpan<float> fp4 = [0, 0.5f, 1, 1.5f, 2, 3, 4, 6, 0, -0.5f, -1, -1.5f, -2, -3, -4, -6];
        var mx = tensor.Encoding == TensorEncoding.MxFp4;
        var blockBytes = mx ? 17 : 18;
        for (var i = 0; i < output.Length; i += 32)
        {
            var block = input.Slice(i / 32 * blockBytes, blockBytes);
            var scale = mx ? MathF.ScaleB(1, block[0] - 127) :
                (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(block));
            for (var j = 0; j < 16; j++)
            {
                var pair = block[j + (mx ? 1 : 2)];
                var low = pair & 15;
                var high = pair >> 4;
                output[i + j] = scale * (mx ? fp4[low] : low - 8);
                output[i + j + 16] = scale * (mx ? fp4[high] : high - 8);
            }
        }
    }
}

internal sealed class UnsupportedModelException(string message) : Exception(message);
