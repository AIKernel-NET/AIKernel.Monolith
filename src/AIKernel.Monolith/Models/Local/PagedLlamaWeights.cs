using System.Runtime.InteropServices;
using AIKernel.Kernel.Memory;

namespace AIKernel.Monolith.Models.Local;

internal sealed unsafe class PagedLlamaWeights : IDisposable
{
    private readonly VirtualMemorySpace memory;
    private readonly TransformerCompute compute;
    private readonly VirtualMemoryBlockOperation multiply;
    private float* input;
    private float* output;
    private int columnCount;
    private int rowOffset;
    private int rowCount;
    private byte* staging;
    private Span<byte> Staging => new(staging, memory.BlockSize);
    private long cachedBlock = -1;

    internal PagedLlamaWeights(VirtualMemorySpace memory, TransformerCompute compute)
    {
        this.memory = memory;
        this.compute = compute;
        multiply = MultiplyBlock;
        staging = (byte*)NativeMemory.Alloc((nuint)memory.BlockSize);
    }

    internal void Load(FileStream checkpoint, IEnumerable<WeightMatrix> matrices, CancellationToken token)
    {
        foreach (var matrix in matrices)
        {
            checkpoint.Position = matrix.Offset;
            for (var row = 0; row < matrix.Rows; row += matrix.RowsPerBlock)
            {
                token.ThrowIfCancellationRequested();
                var count = Math.Min(matrix.RowsPerBlock, matrix.Rows - row) * matrix.Columns * sizeof(float);
                checkpoint.ReadExactly(Staging.Slice(0, count));
                Staging.Slice(count).Clear();
                memory.WriteBlock(matrix.FirstBlock + row / matrix.RowsPerBlock, Staging);
            }
        }
    }

    internal ReadOnlySpan<float> ReadRow(WeightMatrix matrix, int row)
    {
        var block = matrix.FirstBlock + row / matrix.RowsPerBlock;
        if (cachedBlock != block)
        {
            memory.ReadBlock(block, Staging);
            cachedBlock = block;
        }

        return MemoryMarshal.Cast<byte, float>(Staging)
            .Slice((row % matrix.RowsPerBlock) * matrix.Columns, matrix.Columns);
    }

    internal void Multiply(WeightMatrix matrix, ReadOnlySpan<float> vector, Span<float> result, CancellationToken token)
    {
        if (vector.Length != matrix.Columns || result.Length != matrix.Rows)
        {
            throw new ArgumentException("Matrix dimensions do not match the input and output spans.");
        }

        // The callback runs synchronously while both buffers are fixed.
        fixed (float* x = vector, y = result)
        {
            input = x;
            output = y;
            columnCount = matrix.Columns;
            try
            {
                for (var row = 0; row < matrix.Rows; row += matrix.RowsPerBlock)
                {
                    token.ThrowIfCancellationRequested();
                    rowOffset = row;
                    rowCount = Math.Min(matrix.RowsPerBlock, matrix.Rows - row);
                    memory.UseComputeBlock(matrix.FirstBlock + row / matrix.RowsPerBlock, multiply);
                    MatrixBlocks++;
                }
            }
            finally
            {
                input = output = null;
            }
        }
    }

    private void MultiplyBlock(ReadOnlySpan<byte> host, AIKernel.Core.Memory.IVirtualMemoryDeviceBlock? device)
        => compute.MultiplyRows(host, device, new ReadOnlySpan<float>(input, columnCount), new Span<float>(output + rowOffset, rowCount));

    public void Dispose()
    {
        NativeMemory.Free(staging);
        staging = null;
    }

    internal long MatrixBlocks { get; private set; }
}

internal sealed record WeightMatrix(int Rows, int Columns, int RowsPerBlock, long FirstBlock, long Offset,
    TensorEncoding Encoding = TensorEncoding.F32);
internal sealed record LlamaLayer(WeightMatrix Query, WeightMatrix Key, WeightMatrix Value, WeightMatrix Output,
    WeightMatrix Gate, WeightMatrix Down, WeightMatrix Up, WeightMatrix AttentionNorm, WeightMatrix FeedForwardNorm, int NormRow);

internal sealed class LlamaCheckpoint
{
    internal int Dimension { get; }
    internal int HiddenDimension { get; }
    internal int Heads { get; }
    internal int KvHeads { get; }
    internal int HeadSize => Dimension / Heads;
    internal int KvDimension => KvHeads * HeadSize;
    internal int Vocabulary { get; }
    internal int Context { get; }
    internal float RopeBase { get; } = 10000;
    internal float Epsilon { get; } = 1e-5f;
    internal bool SplitRope { get; }
    internal long BlockCount { get; }
    internal WeightMatrix Embedding { get; }
    internal WeightMatrix FinalNorm { get; }
    internal WeightMatrix Classifier { get; }
    internal LlamaLayer[] Layers { get; }
    internal List<WeightMatrix> Matrices { get; } = [];

    internal LlamaCheckpoint(FileStream file, int blockSize, int contextLimit)
    {
        if (!BitConverter.IsLittleEndian)
        {
            throw new PlatformNotSupportedException("FP32 checkpoints require a little-endian host.");
        }

        using var reader = new BinaryReader(file, System.Text.Encoding.UTF8, leaveOpen: true);
        Dimension = reader.ReadInt32();
        HiddenDimension = reader.ReadInt32();
        var layers = reader.ReadInt32();
        Heads = reader.ReadInt32();
        KvHeads = reader.ReadInt32();
        var signedVocab = reader.ReadInt32();
        var context = reader.ReadInt32();

        if (Dimension is < 2 or > 8192 || HiddenDimension is < 1 or > 65536 ||
            layers is < 1 or > 96 || Heads is < 1 or > 128 || KvHeads < 1 || KvHeads > Heads ||
            Dimension % Heads != 0 || Heads % KvHeads != 0 || HeadSize % 2 != 0 ||
            signedVocab is > 200000 or < -200000 or >= -3 and <= 3 || context is < 2 or > 131072 ||
            Math.Max(Dimension, HiddenDimension) > blockSize / sizeof(float))
        {
            throw new InvalidDataException("Invalid legacy llama2.c FP32 checkpoint header or block size.");
        }

        Context = Math.Min(context, contextLimit);
        if ((long)layers * Context * KvDimension * 2 * sizeof(float) > 512 * 1024 * 1024)
        {
            throw new InvalidDataException("The model exceeds the 512 MiB per-request KV cache limit.");
        }

        Vocabulary = Math.Abs(signedVocab);
        var offset = file.Position;
        long block = 0;
        Embedding = Reserve(Vocabulary, Dimension);
        var attentionNorm = Reserve(layers, Dimension);
        var queries = ReserveLayers(Dimension, Dimension);
        var keys = ReserveLayers(KvDimension, Dimension);
        var values = ReserveLayers(KvDimension, Dimension);
        var outputs = ReserveLayers(Dimension, Dimension);
        var feedForwardNorm = Reserve(layers, Dimension);
        var gates = ReserveLayers(HiddenDimension, Dimension);
        var downs = ReserveLayers(Dimension, HiddenDimension);
        var ups = ReserveLayers(HiddenDimension, Dimension);
        FinalNorm = Reserve(1, Dimension);
        offset += (long)context * HeadSize * sizeof(float); // Legacy stored RoPE tables.
        Classifier = signedVocab > 0 ? Embedding : Reserve(Vocabulary, Dimension);
        BlockCount = block;
        if (offset != file.Length)
        {
            throw new InvalidDataException("The checkpoint length does not match its model dimensions.");
        }

        Layers = new LlamaLayer[layers];
        for (var layer = 0; layer < layers; layer++)
        {
            Layers[layer] = new(queries[layer], keys[layer], values[layer], outputs[layer],
                gates[layer], downs[layer], ups[layer], attentionNorm, feedForwardNorm, layer);
        }

        WeightMatrix Reserve(int rows, int columns)
        {
            var rowsPerBlock = blockSize / (columns * sizeof(float));
            var matrix = new WeightMatrix(rows, columns, rowsPerBlock, block, offset);
            Matrices.Add(matrix);
            block += (rows + (long)rowsPerBlock - 1) / rowsPerBlock;
            offset += (long)rows * columns * sizeof(float);
            return matrix;
        }

        WeightMatrix[] ReserveLayers(int rows, int columns)
            => Enumerable.Range(0, layers).Select(_ => Reserve(rows, columns)).ToArray();
    }

    internal LlamaCheckpoint(MappedModelFile file, int blockSize, int contextLimit)
    {
        var dims = file.Dimensions;
        Dimension = dims.Width;
        HiddenDimension = dims.Hidden;
        Heads = dims.Heads;
        KvHeads = dims.KvHeads;
        Vocabulary = dims.Vocab;
        RopeBase = dims.RopeBase;
        Epsilon = dims.Epsilon;
        SplitRope = dims.SplitRope;
        Context = Math.Min(dims.Context, contextLimit);
        if (Dimension is < 2 or > 8192 || HiddenDimension is < 1 or > 65536 ||
            dims.Layers is < 1 or > 96 || Heads is < 1 or > 128 || KvHeads < 1 || KvHeads > Heads ||
            Dimension % Heads != 0 || Heads % KvHeads != 0 || HeadSize % 2 != 0 ||
            Vocabulary is < 4 or > 200000 || Context < 2 || !float.IsFinite(RopeBase) || RopeBase <= 0 ||
            !float.IsFinite(Epsilon) || Epsilon <= 0 || Math.Max(Dimension, HiddenDimension) > blockSize / 4)
        {
            throw new InvalidDataException("Invalid Llama model shape or block size.");
        }

        long block = 0;
        Embedding = Reserve(file.Gguf ? "token_embd.weight" : "model.embed_tokens.weight", Vocabulary, Dimension);
        Layers = new LlamaLayer[dims.Layers];
        for (var layer = 0; layer < Layers.Length; layer++)
        {
            var prefix = file.Gguf ? $"blk.{layer}." : $"model.layers.{layer}.";
            Layers[layer] = new(
                Reserve(prefix + (file.Gguf ? "attn_q.weight" : "self_attn.q_proj.weight"), Dimension, Dimension),
                Reserve(prefix + (file.Gguf ? "attn_k.weight" : "self_attn.k_proj.weight"), KvDimension, Dimension),
                Reserve(prefix + (file.Gguf ? "attn_v.weight" : "self_attn.v_proj.weight"), KvDimension, Dimension),
                Reserve(prefix + (file.Gguf ? "attn_output.weight" : "self_attn.o_proj.weight"), Dimension, Dimension),
                Reserve(prefix + (file.Gguf ? "ffn_gate.weight" : "mlp.gate_proj.weight"), HiddenDimension, Dimension),
                Reserve(prefix + (file.Gguf ? "ffn_down.weight" : "mlp.down_proj.weight"), Dimension, HiddenDimension),
                Reserve(prefix + (file.Gguf ? "ffn_up.weight" : "mlp.up_proj.weight"), HiddenDimension, Dimension),
                Reserve(prefix + (file.Gguf ? "attn_norm.weight" : "input_layernorm.weight"), 1, Dimension),
                Reserve(prefix + (file.Gguf ? "ffn_norm.weight" : "post_attention_layernorm.weight"), 1, Dimension), 0);
        }

        FinalNorm = Reserve(file.Gguf ? "output_norm.weight" : "model.norm.weight", 1, Dimension);
        var output = file.Gguf ? "output.weight" : "lm_head.weight";
        Classifier = file.Contains(output) ? Reserve(output, Vocabulary, Dimension) : Embedding;
        BlockCount = block;

        var ordered = Matrices.OrderBy(matrix => matrix.Offset).ToArray();
        for (var i = 1; i < ordered.Length; i++)
        {
            var previous = ordered[i - 1];
            if (previous.Offset + (long)previous.Rows * MappedModelFile.RowBytes(previous.Encoding, previous.Columns)
                > ordered[i].Offset)
            {
                throw new InvalidDataException("Mapped model tensors overlap.");
            }
        }

        WeightMatrix Reserve(string name, int rows, int cols)
        {
            var tensor = file.Require(name, rows, cols);
            var perBlock = blockSize / (cols * sizeof(float));
            var matrix = new WeightMatrix(rows, cols, perBlock, block, tensor.Offset, tensor.Encoding);
            block += (rows + (long)perBlock - 1) / perBlock;
            Matrices.Add(matrix);
            return matrix;
        }
    }

}
