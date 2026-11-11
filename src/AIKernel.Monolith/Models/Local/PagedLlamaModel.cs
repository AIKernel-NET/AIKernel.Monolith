using System.Text;
using AIKernel.Kernel.Memory;
using AIKernel.Monolith.OpenAI;

namespace AIKernel.Monolith.Models.Local;

internal sealed class PagedLlamaModel : IDisposable
{
    private readonly CudaVirtualMemoryDevice cuda;
    private readonly VirtualMemorySpace memory;
    private readonly PagedLlamaWeights weights;
    private readonly LlamaCheckpoint shape;
    private readonly LlamaTokenizer tokenizer;
    private readonly int maxNewTokens;
    private readonly long checkpointBytes;

    internal PagedLlamaModel(PagedLlamaSettings settings, CancellationToken token)
    {
        using var file = new FileStream(settings.Checkpoint, FileMode.Open, FileAccess.Read, FileShare.Read);
        checkpointBytes = file.Length;
        shape = new LlamaCheckpoint(file, settings.BlockSize, settings.ContextTokens);
        tokenizer = new LlamaTokenizer(settings.Tokenizer, shape.Vocabulary);
        maxNewTokens = settings.MaxNewTokens;
        Directory.CreateDirectory(settings.SwapDirectory);
        cuda = new CudaVirtualMemoryDevice(settings.Device);
        try
        {
            memory = new VirtualMemorySpace(shape.BlockCount, settings.BlockSize, settings.RamBlocks,
                settings.SwapDirectory, settings.VramBlocks, cuda);
            try
            {
                weights = new PagedLlamaWeights(memory, cuda);
                weights.Load(file, shape.Matrices, token);
            }
            catch
            {
                memory.Dispose();
                throw;
            }
        }
        catch
        {
            cuda.Dispose();
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
        var state = new ForwardState(shape, weights);
        var decoder = Encoding.UTF8.GetDecoder();
        for (var pos = 0; pos < tokens.Count - 1; pos++)
        {
            state.Forward(tokens[pos], pos, token);
        }

        var position = tokens.Count - 1;
        var current = tokens[^1];
        var finish = "length";
        for (var step = 0; step < maxNewTokens && position < shape.Context; step++, position++)
        {
            state.Forward(current, position, token);
            var next = state.Predict(token);
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
        audit["cuda_matvec_blocks"] = (weights.MatrixBlocks - matrixBlocks).ToString(System.Globalization.CultureInfo.InvariantCulture);
        return finish;
    }

    public void Dispose()
    {
        try
        {
            memory.Dispose();
        }
        finally
        {
            cuda.Dispose();
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

    private sealed class ForwardState(LlamaCheckpoint shape, PagedLlamaWeights weights)
    {
        private readonly float[] activation = new float[shape.Dimension];
        private readonly float[] normalized = new float[shape.Dimension];
        private readonly float[] query = new float[shape.Dimension];
        private readonly float[] key = new float[shape.KvDimension];
        private readonly float[] value = new float[shape.KvDimension];
        private readonly float[] attended = new float[shape.Dimension];
        private readonly float[] projected = new float[shape.Dimension];
        private readonly float[] gate = new float[shape.HiddenDimension];
        private readonly float[] up = new float[shape.HiddenDimension];
        private readonly float[] logits = new float[shape.Vocabulary];
        private readonly float[] scores = new float[shape.Context];
        private readonly float[][] keys = shape.Layers.Select(_ => new float[shape.Context * shape.KvDimension]).ToArray();
        private readonly float[][] values = shape.Layers.Select(_ => new float[shape.Context * shape.KvDimension]).ToArray();
        private readonly float[] rope = Enumerable.Range(0, shape.HeadSize / 2)
            .Select(pair => MathF.Pow(10000, -2f * pair / shape.HeadSize)).ToArray();

        internal void Forward(int tokenId, int position, CancellationToken token)
        {
            weights.ReadRow(shape.Embedding, tokenId).CopyTo(activation);
            for (var index = 0; index < shape.Layers.Length; index++)
            {
                var layer = shape.Layers[index];
                Normalize(weights.ReadRow(shape.AttentionNorm, index));
                weights.Multiply(layer.Query, normalized, query, token);
                weights.Multiply(layer.Key, normalized, key, token);
                weights.Multiply(layer.Value, normalized, value, token);
                Rotate(query, position);
                Rotate(key, position);
                key.CopyTo(keys[index], position * shape.KvDimension);
                value.CopyTo(values[index], position * shape.KvDimension);
                Attend(index, position);
                weights.Multiply(layer.Output, attended, projected, token);
                AddResidual();

                Normalize(weights.ReadRow(shape.FeedForwardNorm, index));
                weights.Multiply(layer.Gate, normalized, gate, token);
                weights.Multiply(layer.Up, normalized, up, token);
                for (var i = 0; i < gate.Length; i++)
                {
                    gate[i] = gate[i] / (1f + MathF.Exp(-gate[i])) * up[i];
                }

                weights.Multiply(layer.Down, gate, projected, token);
                AddResidual();
            }
        }

        internal int Predict(CancellationToken token)
        {
            Normalize(weights.ReadRow(shape.FinalNorm, 0));
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

            return best;
        }

        private void Normalize(ReadOnlySpan<float> scale)
        {
            var norm = 0f;
            foreach (var component in activation)
            {
                norm += component * component;
            }

            var inverse = 1f / MathF.Sqrt(norm / activation.Length + 1e-5f);
            for (var i = 0; i < activation.Length; i++)
            {
                normalized[i] = activation[i] * inverse * scale[i];
            }
        }

        private void Rotate(Span<float> vector, int position)
        {
            for (var i = 0; i < vector.Length; i += 2)
            {
                var (sin, cos) = MathF.SinCos(position * rope[(i % shape.HeadSize) / 2]);
                var first = vector[i];
                var second = vector[i + 1];
                vector[i] = first * cos - second * sin;
                vector[i + 1] = first * sin + second * cos;
            }
        }

        private void Attend(int layer, int position)
        {
            var width = shape.HeadSize;
            var scale = 1f / MathF.Sqrt(width);
            for (var head = 0; head < shape.Heads; head++)
            {
                var queryOffset = head * width;
                var kvOffset = head / (shape.Heads / shape.KvHeads) * width;
                var max = float.NegativeInfinity;
                for (var t = 0; t <= position; t++)
                {
                    var dot = 0f;
                    var offset = t * shape.KvDimension + kvOffset;
                    for (var j = 0; j < width; j++)
                    {
                        dot += query[queryOffset + j] * keys[layer][offset + j];
                    }

                    scores[t] = dot * scale;
                    max = MathF.Max(max, scores[t]);
                }

                var sum = 0f;
                for (var t = 0; t <= position; t++)
                {
                    scores[t] = MathF.Exp(scores[t] - max);
                    sum += scores[t];
                }

                var target = attended.AsSpan(queryOffset, width);
                target.Clear();
                for (var t = 0; t <= position; t++)
                {
                    var coefficient = scores[t] / sum;
                    var offset = t * shape.KvDimension + kvOffset;
                    for (var j = 0; j < width; j++)
                    {
                        target[j] += coefficient * values[layer][offset + j];
                    }
                }
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
