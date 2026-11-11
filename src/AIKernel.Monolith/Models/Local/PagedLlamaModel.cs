using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using AIKernel.Kernel.Memory;
using AIKernel.Monolith.OpenAI;

namespace AIKernel.Monolith.Models.Local;

internal sealed class PagedLlamaModel : IDisposable
{
    private readonly CudaVirtualMemoryDevice? cuda;
    private readonly TransformerCompute compute;
    private readonly PagedLlamaSettings settings;
    private readonly int cudaStatus;
    private readonly VirtualMemorySpace memory;
    private readonly PagedLlamaWeights weights;
    private readonly LlamaCheckpoint shape;
    private readonly LlamaTokenizer tokenizer;
    private readonly int maxNewTokens;
    private readonly long checkpointBytes;
    private readonly MappedModelFile? mapped;

    internal PagedLlamaModel(PagedLlamaSettings settings, CancellationToken token)
    {
        this.settings = settings;
        var extension = Path.GetExtension(settings.Checkpoint);
        var isMapped = extension.Equals(".gguf", StringComparison.OrdinalIgnoreCase) ||
            extension.Equals(".safetensors", StringComparison.OrdinalIgnoreCase);
        using var legacy = isMapped ? null : File.OpenRead(settings.Checkpoint);
        mapped = isMapped ? new MappedModelFile(settings.Checkpoint, settings.Config) : null;
        try
        {
            checkpointBytes = mapped?.Length ?? legacy!.Length;
            shape = mapped is null ? new LlamaCheckpoint(legacy!, settings.BlockSize, settings.ContextTokens)
                : new LlamaCheckpoint(mapped, settings.BlockSize, settings.ContextTokens);
            tokenizer = settings.Tokenizer is { } tokenizerPath
                ? new LlamaTokenizer(tokenizerPath, shape.Vocabulary)
                : LlamaTokenizer.FromEmbedded(mapped, shape.Vocabulary);
            maxNewTokens = settings.MaxNewTokens;
            Directory.CreateDirectory(settings.SwapDirectory);
            cuda = CudaVirtualMemoryDevice.TryCreate(settings.Device, out cudaStatus);
            compute = new TransformerCompute(cuda);
            try
            {
                var source = mapped is null ? null : new MappedTensorSource(mapped, shape.Matrices.ToArray());
                memory = new VirtualMemorySpace(shape.BlockCount, settings.BlockSize, settings.RamBlocks,
                    settings.SwapDirectory, cuda is null ? 0 : settings.VramBlocks, cuda, source);
                try
                {
                    weights = new PagedLlamaWeights(memory, compute);
                    try
                    {
                        if (legacy is not null)
                        {
                            weights.Load(legacy, shape.Matrices, token);
                        }
                    }
                    catch
                    {
                        weights.Dispose();
                        throw;
                    }
                }
                catch
                {
                    memory.Dispose();
                    throw;
                }
            }
            catch
            {
                cuda?.Dispose();
                throw;
            }
        }
        catch
        {
            mapped?.Dispose();
            throw;
        }
    }

    internal async Task<string> GenerateAsync(IReadOnlyList<ChatMessage> messages, Func<string, Task> emit,
        Dictionary<string, string> audit, CancellationToken token)
    {
        var prompt = FormatPrompt(messages);
        var tokens = tokenizer.Encode(prompt, token);
        if (tokens.Count > shape.Context)
        {
            throw new LocalContextLengthException();
        }

        var initial = memory.GetStatistics();
        var matrixBlocks = weights.MatrixBlocks;
        using var kv = new PagedKvCache(shape, settings, cuda, compute);
        using var state = new ForwardState(shape, weights, compute, kv, settings);
        var decoder = Encoding.UTF8.GetDecoder();
        var gen2 = GC.CollectionCount(2);
        var allocated = GC.GetTotalAllocatedBytes(precise: false);
        long maxStepTicks = 0;
        for (var pos = 0; pos < tokens.Count - 1; pos++)
        {
            state.Forward(tokens[pos], pos, token);
        }

        var position = tokens.Count - 1;
        var current = tokens[^1];
        var finish = "length";
        for (var step = 0; step < maxNewTokens && position < shape.Context; step++, position++)
        {
            var stepStarted = Stopwatch.GetTimestamp();
            state.Forward(current, position, token);
            var next = state.Predict(token);
            maxStepTicks = Math.Max(maxStepTicks, Stopwatch.GetTimestamp() - stepStarted);
            if (next is LlamaTokenizer.Begin or LlamaTokenizer.End)
            {
                finish = "stop";
                break;
            }

            var piece = tokenizer.Decode(next, decoder);
            if (piece.Length > 0)
            {
                await emit(piece);
            }

            current = next;
        }

        var tail = LlamaTokenizer.Flush(decoder);
        if (tail.Length > 0)
        {
            await emit(tail);
        }

        var final = memory.GetStatistics();
        audit["checkpoint_bytes"] = checkpointBytes.ToString(System.Globalization.CultureInfo.InvariantCulture);
        audit["weight_blocks"] = shape.BlockCount.ToString(System.Globalization.CultureInfo.InvariantCulture);
        audit["weight_ram_blocks"] = initial.RamBlocks.ToString(System.Globalization.CultureInfo.InvariantCulture);
        audit["weight_vram_blocks"] = initial.VramBlocks.ToString(System.Globalization.CultureInfo.InvariantCulture);
        audit["weight_swap_blocks"] = initial.SwapBlocks.ToString(System.Globalization.CultureInfo.InvariantCulture);
        audit["weight_swap_reads"] = (final.SwapReads - initial.SwapReads).ToString(System.Globalization.CultureInfo.InvariantCulture);
        audit["weight_vram_page_ins"] = (final.DevicePageIns - initial.DevicePageIns).ToString(System.Globalization.CultureInfo.InvariantCulture);
        audit["backend"] = cuda is null ? "cpu" : "cuda";
        audit["cuda_status"] = cudaStatus.ToString(System.Globalization.CultureInfo.InvariantCulture);
        audit[cuda is null ? "cpu_matvec_blocks" : "cuda_matvec_blocks"] =
            (weights.MatrixBlocks - matrixBlocks).ToString(System.Globalization.CultureInfo.InvariantCulture);
        audit["mapped_weight_reads"] = (final.BackingReads - initial.BackingReads).ToString(System.Globalization.CultureInfo.InvariantCulture);
        audit["kv_swap_reads"] = kv.Statistics.SwapReads.ToString(System.Globalization.CultureInfo.InvariantCulture);
        audit["kv_vram_page_ins"] = kv.Statistics.DevicePageIns.ToString(System.Globalization.CultureInfo.InvariantCulture);
        audit["generation_gen2_collections"] = (GC.CollectionCount(2) - gen2).ToString(System.Globalization.CultureInfo.InvariantCulture);
        audit["generation_allocated_bytes"] = (GC.GetTotalAllocatedBytes(precise: false) - allocated)
            .ToString(System.Globalization.CultureInfo.InvariantCulture);
        audit["generation_max_step_us"] = ((long)(maxStepTicks * (1_000_000d / Stopwatch.Frequency)))
            .ToString(System.Globalization.CultureInfo.InvariantCulture);
        return finish;
    }

    public void Dispose()
    {
        weights.Dispose();
        try
        {
            memory.Dispose();
        }
        finally
        {
            try { cuda?.Dispose(); }
            finally { mapped?.Dispose(); }
        }
    }

    private static string FormatPrompt(IReadOnlyList<ChatMessage> messages)
    {
        if (messages.Sum(message => (long)message.Content.Length + message.Role.Length + 3) > 8192)
        {
            throw new LocalContextLengthException();
        }

        return messages is [{ Role: "user" } user]
            ? user.Content
            : string.Join('\n', messages.Select(message => $"{message.Role}: {message.Content}")) + "\nassistant:";
    }

    private sealed unsafe class ForwardState(LlamaCheckpoint shape, PagedLlamaWeights weights, TransformerCompute compute, PagedKvCache kv, PagedLlamaSettings settings) : IDisposable
    {
        private readonly float[] activation = new float[shape.Dimension];
        private readonly float[] normalized = new float[shape.Dimension];
        private readonly float[] query = new float[shape.Dimension];
        private readonly float[] key = new float[shape.KvDimension];
        private readonly float[] value = new float[shape.KvDimension];
        private readonly float[] attended = new float[shape.Dimension];
        private readonly float[] projected = new float[shape.Dimension];
        private readonly float[] sinCos = new float[shape.Dimension];
        private readonly float[] rotation = new float[shape.Dimension];
        private readonly float[] rope = Enumerable.Range(0, shape.HeadSize / 2)
            .Select(pair => MathF.Pow(shape.RopeBase, -2f * pair / shape.HeadSize)).ToArray();
        // Feed-forward and vocabulary buffers can each exceed the managed LOH threshold.
        private readonly float* scratch = (float*)NativeMemory.Alloc(
            (nuint)(2 * shape.HiddenDimension + 2 * shape.Vocabulary), sizeof(float));

        private Span<float> Gate => new(scratch, shape.HiddenDimension);
        private Span<float> Up => new(scratch + shape.HiddenDimension, shape.HiddenDimension);
        private Span<float> Logits => new(scratch + 2 * shape.HiddenDimension, shape.Vocabulary);
        private Span<int> SampleIndices => new(scratch + 2 * shape.HiddenDimension + shape.Vocabulary, shape.Vocabulary);

        public void Dispose() => NativeMemory.Free(scratch);

        internal void Forward(int tokenId, int position, CancellationToken token)
        {
            weights.ReadRow(shape.Embedding, tokenId).CopyTo(activation);
            for (var index = 0; index < shape.Layers.Length; index++)
            {
                var layer = shape.Layers[index];
                Normalize(weights.ReadRow(layer.AttentionNorm, layer.NormRow));
                weights.Multiply(layer.Query, normalized, query, token);
                weights.Multiply(layer.Key, normalized, key, token);
                weights.Multiply(layer.Value, normalized, value, token);
                Rotate(query, position);
                Rotate(key, position);
                kv.Write(index, position, key, value);
                Attend(index, position, token);
                weights.Multiply(layer.Output, attended, projected, token);
                AddResidual();

                Normalize(weights.ReadRow(layer.FeedForwardNorm, layer.NormRow));
                var gate = Gate;
                var up = Up;
                weights.Multiply(layer.Gate, normalized, gate, token);
                weights.Multiply(layer.Up, normalized, up, token);
                compute.SwiGlu(gate, up);

                weights.Multiply(layer.Down, gate, projected, token);
                AddResidual();
            }
        }

        internal int Predict(CancellationToken token)
        {
            Normalize(weights.ReadRow(shape.FinalNorm, 0));
            var logits = Logits;
            weights.Multiply(shape.Classifier, normalized, logits, token);
            var best = 0;
            for (var i = 0; i < logits.Length; i++)
            {
                if (!float.IsFinite(logits[i]))
                {
                    throw new InvalidDataException("The model produced non-finite logits.");
                }

                if (logits[i] > logits[best])
                {
                    best = i;
                }
            }

            if (settings.Temperature == 0)
            {
                return best;
            }

            var maximum = logits[best];
            var sampleIndices = SampleIndices;
            for (var i = 0; i < logits.Length; i++)
            {
                logits[i] = (logits[i] - maximum) / settings.Temperature;
                sampleIndices[i] = i;
            }

            compute.Softmax(logits);
            logits.Sort(sampleIndices);
            var cutoff = logits.Length - 1;
            var mass = logits[cutoff];
            while (cutoff > 0 && mass < settings.TopP)
            {
                mass += logits[--cutoff];
            }

            var draw = Random.Shared.NextSingle() * mass;
            for (var i = logits.Length - 1; i >= cutoff; i--)
            {
                draw -= logits[i];
                if (draw <= 0) return sampleIndices[i];
            }

            return sampleIndices[cutoff];
        }

        private void Normalize(ReadOnlySpan<float> scale)
            => compute.Normalize(activation, scale, normalized, shape.Epsilon);

        private void Rotate(Span<float> vector, int position)
        {
            var coefficients = sinCos.AsSpan(0, vector.Length);
            for (var i = 0; i < vector.Length; i += 2)
            {
                var (sin, cos) = MathF.SinCos(position * rope[(i % shape.HeadSize) / 2]);
                coefficients[i] = sin;
                coefficients[i + 1] = cos;
            }

            if (!shape.SplitRope)
            {
                compute.Rope(vector, coefficients);
                return;
            }

            var pairs = rotation.AsSpan(0, vector.Length);
            var half = shape.HeadSize / 2;
            for (var start = 0; start < vector.Length; start += shape.HeadSize)
            {
                for (var i = 0; i < half; i++)
                {
                    pairs[start + 2 * i] = vector[start + i];
                    pairs[start + 2 * i + 1] = vector[start + half + i];
                }
            }

            compute.Rope(pairs, coefficients);
            for (var start = 0; start < vector.Length; start += shape.HeadSize)
            {
                for (var i = 0; i < half; i++)
                {
                    vector[start + i] = pairs[start + 2 * i];
                    vector[start + half + i] = pairs[start + 2 * i + 1];
                }
            }
        }

        private void Attend(int layer, int position, CancellationToken token)
        {
            var width = shape.HeadSize;
            for (var head = 0; head < shape.Heads; head++)
            {
                var queryOffset = head * width;
                var kvOffset = head / (shape.Heads / shape.KvHeads) * width;
                kv.Attend(layer, position, kvOffset, query.AsSpan(queryOffset, width),
                    attended.AsSpan(queryOffset, width), token);
            }
        }

        private void AddResidual()
        {
            for (var i = 0; i < activation.Length; i++)
            {
                activation[i] += projected[i];
            }
        }
    }
}

internal sealed class LocalContextLengthException : Exception;
