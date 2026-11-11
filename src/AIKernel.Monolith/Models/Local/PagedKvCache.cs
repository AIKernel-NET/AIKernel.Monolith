using System.Runtime.InteropServices;
using AIKernel.Core.Memory;
using AIKernel.Kernel.Memory;

namespace AIKernel.Monolith.Models.Local;

internal sealed class PagedKvCache : IDisposable
{
    private readonly VirtualMemorySpace memory;
    private readonly TransformerCompute compute;
    private readonly VirtualMemoryBlockOperation execute;
    private readonly int pagesPerLayer;
    private readonly int kvWidth;
    private readonly int rowsPerPage;
    private readonly float[] scores;
    private readonly float[] scaledQuery;
    private readonly float[] weightedValues;
    private int firstRow;
    private int rowCount;
    private int headOffset;
    private bool values;

    internal PagedKvCache(LlamaCheckpoint shape, PagedLlamaSettings settings, CudaVirtualMemoryDevice? cuda,
        TransformerCompute compute)
    {
        this.compute = compute;
        execute = ExecuteTile;
        kvWidth = shape.KvDimension;
        var blockSize = Math.Max(64 * 1024, 2 * kvWidth * sizeof(float));
        rowsPerPage = blockSize / (2 * kvWidth * sizeof(float));
        pagesPerLayer = (shape.Context + rowsPerPage - 1) / rowsPerPage;
        memory = new VirtualMemorySpace((long)shape.Layers.Length * pagesPerLayer, blockSize,
            settings.KvRamBlocks, settings.SwapDirectory, cuda is null ? 0 : settings.KvVramBlocks, cuda);
        scores = new float[shape.Context];
        scaledQuery = new float[shape.HeadSize];
        weightedValues = new float[shape.HeadSize];
    }

    internal void Write(int layer, int position, ReadOnlySpan<float> key, ReadOnlySpan<float> value)
    {
        var block = Block(layer, position);
        var offset = position % rowsPerPage * 2 * kvWidth * sizeof(float);
        memory.WriteBlockSlice(block, offset, MemoryMarshal.AsBytes(key));
        memory.WriteBlockSlice(block, offset + key.Length * sizeof(float), MemoryMarshal.AsBytes(value));
    }

    internal void Attend(int layer, int position, int offset, ReadOnlySpan<float> query, Span<float> output,
        CancellationToken token)
    {
        var scale = 1f / MathF.Sqrt(query.Length);
        for (var i = 0; i < query.Length; i++) scaledQuery[i] = query[i] * scale;

        headOffset = offset;
        values = false;
        for (firstRow = 0; firstRow <= position; firstRow += rowsPerPage)
        {
            token.ThrowIfCancellationRequested();
            rowCount = Math.Min(rowsPerPage, position + 1 - firstRow);
            memory.UseComputeBlock(Block(layer, firstRow), execute);
        }

        compute.Softmax(scores.AsSpan(0, position + 1));
        output.Clear();
        values = true;
        for (firstRow = 0; firstRow <= position; firstRow += rowsPerPage)
        {
            token.ThrowIfCancellationRequested();
            rowCount = Math.Min(rowsPerPage, position + 1 - firstRow);
            memory.UseComputeBlock(Block(layer, firstRow), execute);
            for (var i = 0; i < output.Length; i++) output[i] += weightedValues[i];
        }
    }

    private void ExecuteTile(ReadOnlySpan<byte> host, IVirtualMemoryDeviceBlock? device)
    {
        var pageScores = scores.AsSpan(firstRow, rowCount);
        ReadOnlySpan<float> input = values ? pageScores : scaledQuery;
        Span<float> output = values ? weightedValues : pageScores;
        compute.AttentionTile(host, device, input, output, rowCount, kvWidth, headOffset, scaledQuery.Length, values);
    }

    internal VirtualMemoryStatistics Statistics => memory.GetStatistics();
    private long Block(int layer, int position) => (long)layer * pagesPerLayer + position / rowsPerPage;

    public void Dispose() => memory.Dispose();
}
