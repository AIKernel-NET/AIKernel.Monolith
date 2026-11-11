using System.Runtime.InteropServices;
using AIKernel.Kernel.Memory;

namespace AIKernel.Monolith.Models.Local;

internal sealed class PagedLlamaWeights
{
    private readonly VirtualMemorySpace memory;
    private readonly CudaVirtualMemoryDevice cuda;
    private readonly byte[] staging;
    private long cachedBlock = -1;

    internal PagedLlamaWeights(VirtualMemorySpace memory, CudaVirtualMemoryDevice cuda)
    {
        this.memory = memory;
        this.cuda = cuda;
        staging = new byte[memory.BlockSize];
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
                checkpoint.ReadExactly(staging.AsSpan(0, count));
                staging.AsSpan(count).Clear();
                memory.WriteBlock(matrix.FirstBlock + row / matrix.RowsPerBlock, staging);
            }
        }
    }

    internal ReadOnlySpan<float> ReadRow(WeightMatrix matrix, int row)
    {
        var block = matrix.FirstBlock + row / matrix.RowsPerBlock;
        if (cachedBlock != block)
        {
            memory.ReadBlock(block, staging);
            cachedBlock = block;
        }

        return MemoryMarshal.Cast<byte, float>(staging)
            .Slice((row % matrix.RowsPerBlock) * matrix.Columns, matrix.Columns);
    }

    internal void Multiply(WeightMatrix matrix, float[] input, float[] output, CancellationToken token)
    {
        for (var row = 0; row < matrix.Rows; row += matrix.RowsPerBlock)
        {
            token.ThrowIfCancellationRequested();
            var count = Math.Min(matrix.RowsPerBlock, matrix.Rows - row);
            memory.UseDeviceBlock(matrix.FirstBlock + row / matrix.RowsPerBlock,
                block => cuda.MultiplyRows(block, input, output.AsSpan(row, count)));
            MatrixBlocks++;
        }
    }

    internal long MatrixBlocks { get; private set; }
}

internal sealed record WeightMatrix(int Rows, int Columns, int RowsPerBlock, long FirstBlock, long Offset);
internal sealed record LlamaLayer(WeightMatrix Query, WeightMatrix Key, WeightMatrix Value, WeightMatrix Output,
    WeightMatrix Gate, WeightMatrix Down, WeightMatrix Up);

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
    internal long BlockCount { get; }
    internal WeightMatrix Embedding { get; }
    internal WeightMatrix AttentionNorm { get; }
    internal WeightMatrix FeedForwardNorm { get; }
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
        AttentionNorm = Reserve(layers, Dimension);
        var queries = ReserveLayers(Dimension, Dimension);
        var keys = ReserveLayers(KvDimension, Dimension);
        var values = ReserveLayers(KvDimension, Dimension);
        var outputs = ReserveLayers(Dimension, Dimension);
        FeedForwardNorm = Reserve(layers, Dimension);
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
                gates[layer], downs[layer], ups[layer]);
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
}
