using System.Buffers.Binary;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using AIKernel.Common.Results;
using AIKernel.Core.Governance;
using AIKernel.Core.Memory;
using AIKernel.Kernel.Governance;
using System.Text;
using AIKernel.Kernel.Memory;
using AIKernel.Monolith.OpenAI;

namespace AIKernel.Monolith.Models.Local;

internal sealed class PagedLlamaModel : IDisposable
{
    private static readonly IResourceAdmissibilityPolicy Admission = new ResourceAdmissibilityPolicy();
    private static readonly ErrorContext HealthUnavailable = new("Required device telemetry is unavailable.",
        "device_telemetry_unavailable", true) { FailureKind = FailureKind.FailClosed };
    private CudaVirtualMemoryDevice? cuda;
    private readonly IDeviceHealthTelemetry? telemetry;
    private readonly NvmlDeviceTelemetry? ownedTelemetry;
    private string? pciBusId;
    private int deviceOrdinal;
    private readonly bool telemetryFallback;
    private readonly long parameters;
    private readonly int quantizationBits;
    private TransformerCompute compute;
    private readonly PagedLlamaSettings settings;
    private readonly int cudaStatus;
    private readonly VirtualMemorySpace memory;
    private readonly PagedLlamaWeights weights;
    private readonly LlamaCheckpoint shape;
    private readonly LlamaTokenizer tokenizer;
    private readonly int maxNewTokens;
    private readonly long checkpointBytes;
    private readonly MappedModelFile? mapped;

    internal PagedLlamaModel(PagedLlamaSettings settings, CancellationToken token, IDeviceHealthTelemetry? healthTelemetry = null)
    {
        this.settings = settings;
        deviceOrdinal = settings.Device;
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
            long encodedWeightBytes = 0;
            foreach (var matrix in shape.Matrices)
            {
                parameters = checked(parameters + (long)matrix.Rows * matrix.Columns);
                encodedWeightBytes = checked(encodedWeightBytes + (long)matrix.Rows *
                    MappedModelFile.RowBytes(matrix.Encoding, matrix.Columns));
            }
            quantizationBits = checked((int)((encodedWeightBytes * 8 + parameters - 1) / parameters));
            maxNewTokens = settings.MaxNewTokens;
            Directory.CreateDirectory(settings.SwapDirectory);
            cuda = CudaVirtualMemoryDevice.TryCreate(settings.Device, out cudaStatus);
            try
            {
                if (cuda is not null && settings.HealthLimits != default)
                {
                    ownedTelemetry = healthTelemetry is null ? NvmlDeviceTelemetry.Open().Value : null;
                    telemetry = healthTelemetry ?? ownedTelemetry;
                    pciBusId = cuda.GetPciBusId();
                    if (telemetry is null || pciBusId is null || !telemetry.Read(pciBusId).IsSuccess)
                    {
                        ownedTelemetry?.Dispose();
                        telemetry = null;
                        cuda.Dispose();
                        cuda = null;
                        telemetryFallback = true;
                    }
                }

                compute = new TransformerCompute(cuda);
                var source = mapped is null ? null : new MappedTensorSource(mapped, shape.Matrices.ToArray());
                memory = new VirtualMemorySpace(shape.BlockCount, settings.BlockSize, settings.RamBlocks,
                    settings.SwapDirectory, cuda is null ? 0 : settings.VramBlocks, cuda, source, settings.SwapPolicy);
                try
                {
                    weights = new PagedLlamaWeights(memory, compute, settings.SwapPolicy.ZeroWipe);
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
                ownedTelemetry?.Dispose();
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
        Dictionary<string, string> audit, LayerQosScheduler? scheduler, CancellationToken token,
        LocalContinuation continuation = default, Func<int, LayerQosScheduler>? schedulerForDevice = null)
    {
        var prompt = FormatPrompt(messages);
        var tokens = tokenizer.Encode(prompt, token);
        if (tokens.Count > shape.Context)
        {
            throw new LocalContextLengthException();
        }

        if (telemetryFallback) audit["device_route"] = "cpu_telemetry_unavailable";
        PagedKvCache? cache = null;
        using var execution = new DeviceExecution();
        if (cuda is not null)
        {
            scheduler ??= schedulerForDevice?.Invoke(deviceOrdinal);
            if (scheduler is not null)
                execution.Lease = Require(scheduler.Enter(settings.Priority, Spill, token));
            audit["cuda_device_ordinal"] = deviceOrdinal.ToString(System.Globalization.CultureInfo.InvariantCulture);
        }
        var initial = memory.GetStatistics();
        var available = memory.GetAvailableBlocks();
        var kvBlockBytes = PagedKvCache.BlockSizeFor(shape.KvDimension);
        var useGpu = cuda is not null;
        var weightVram = useGpu ? (available.VramBlocks + initial.VramBlocks) * (long)settings.BlockSize : 0;
        var kvVram = useGpu ? settings.KvVramBlocks * (long)kvBlockBytes : 0;
        var configuredVram = weightVram + kvVram;
        var physicalVram = configuredVram;
        var request = new InferenceResourceRequest(parameters, quantizationBits, shape.BlockCount * settings.BlockSize,
            shape.Layers.Length, 2 * shape.KvDimension * sizeof(float), tokens.Count,
            Math.Min(maxNewTokens, shape.Context - tokens.Count + 1));

        if (useGpu && settings.HealthLimits != default)
        {
            var health = Require(CheckHealth());
            var reclaimable = (ulong)initial.VramBlocks * (ulong)settings.BlockSize;
            physicalVram = (long)Math.Min((ulong)physicalVram,
                Math.Min(health.VramFreeBytes, (ulong)long.MaxValue - reclaimable) + reclaimable);
            audit["device_temperature_celsius"] = health.TemperatureCelsius.ToString(System.Globalization.CultureInfo.InvariantCulture);
        }

        var budget = new InferenceResourceBudget(settings.BlockSize, kvBlockBytes,
            (available.RamBlocks + initial.RamBlocks) * (long)settings.BlockSize +
                settings.KvRamBlocks * (long)kvBlockBytes, physicalVram,
            settings.MaxPcieSwapBytes, useGpu, weightVram, kvVram);
        var gate = ResultStep<InferenceResourceRequest, InferenceResourceEstimate>.FromResult(request,
            Admission.Evaluate(request, budget));
        var estimate = Require(gate.ToResult());
        audit["resource_minimum_ram_blocks"] = estimate.MinimumRamBlocks.ToString(System.Globalization.CultureInfo.InvariantCulture);
        audit["resource_minimum_vram_blocks"] = estimate.MinimumVramBlocks.ToString(System.Globalization.CultureInfo.InvariantCulture);
        audit["resource_estimated_pcie_bytes"] = estimate.EstimatedPcieBytes.ToString(System.Globalization.CultureInfo.InvariantCulture);
        audit["qos_priority"] = settings.Priority.ToString();
        var matrixBlocks = weights.MatrixBlocks;
        if (!continuation.FirstLogits.IsEmpty && continuation.FirstLogits.Length != shape.Vocabulary)
            throw new ArgumentException("The logits buffer must match the vocabulary.");
        using var restored = continuation.RestorePath is { } restorePath
            ? Require(VirtualMemorySpace.Restore(restorePath, ramBlocks: settings.KvRamBlocks,
                vramBlocks: useGpu ? settings.KvVramBlocks : 0, device: cuda,
                swapDirectory: settings.SwapDirectory, swapPolicy: settings.SwapPolicy))
            : null;
        if (restored is not null) ValidateContinuation(restored, tokens);
        using var kv = cache = new PagedKvCache(shape, settings, cuda, compute, restored);
        HashSet<string>? visitedDevices = null;
        HashSet<int>? visitedOrdinals = null;
        var migrations = 0;
        var generatedTokens = 0;
        long forwardLayers = 0;
        using var state = new ForwardState(shape, weights, compute, kv, settings, CheckLayer);
        Result<int> Spill()
        {
            var kvSpill = cache!.DemoteDeviceBlocks();
            if (!kvSpill.IsSuccess) return kvSpill;
            return memory.DemoteDeviceBlocks();
        }
        TransformerCompute CheckLayer(int position, int layer)
        {
            forwardLayers++;
            if (execution.Lease is { } lease) Require(lease.Checkpoint());
            if (cuda is null || settings.HealthLimits == default) return compute;
            var health = CheckHealth();
            if (health.Error?.Code == "device_temperature_exceeded" && schedulerForDevice is not null)
                Migrate(position, layer);
            else
                Require(health);
            return compute;
        }

        void Migrate(int position, int layer)
        {
            var previous = cuda!;
            var previousPci = pciBusId!;
            visitedDevices ??= new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            visitedDevices.Add(previousPci);
            visitedOrdinals ??= [];
            visitedOrdinals.Add(deviceOrdinal);
            var allowed = settings.ThermalMigrationDevices;
            var deviceCount = allowed?.Length ?? CudaVirtualMemoryDevice.GetDeviceCount().Value;
            var prepared = false;
            for (var index = 0; index < deviceCount; index++)
            {
                token.ThrowIfCancellationRequested();
                var ordinal = allowed is null ? index : allowed[index];
                if (visitedOrdinals.Contains(ordinal)) continue;
                var candidate = CudaVirtualMemoryDevice.TryCreate(ordinal, out _);
                if (candidate is null) continue;

                var accepted = false;
                LayerQosScheduler.Lease? targetLease = null;
                try
                {
                    var targetPci = candidate.GetPciBusId();
                    if (targetPci is null || visitedDevices.Contains(targetPci) || !CanMigrate(targetPci))
                        continue;

                    if (!prepared)
                    {
                        RequireMigration(memory.PrepareDeviceChange());
                        if (settings.KvVramBlocks != 0) RequireMigration(kv.PrepareDeviceChange());
                        previous.ZeroScratch();
                        execution.Release();
                        prepared = true;
                    }

                    var entered = schedulerForDevice!(ordinal).Enter(settings.Priority, Spill, token);
                    if (!entered.IsSuccess)
                    {
                        token.ThrowIfCancellationRequested();
                        continue;
                    }
                    targetLease = entered.Value!;
                    if (!CanMigrate(targetPci)) continue;

                    var targetCompute = new TransformerCompute(candidate);
                    RequireMigration(memory.BindPreparedDevice(candidate));
                    if (settings.KvVramBlocks != 0)
                    {
                        var bound = kv.BindPreparedDevice(candidate);
                        if (!bound.IsSuccess)
                        {
                            RequireMigration(memory.BindPreparedDevice(previous));
                            RequireMigration(bound);
                        }
                    }

                    cuda = candidate;
                    pciBusId = targetPci;
                    deviceOrdinal = ordinal;
                    compute = targetCompute;
                    weights.SetCompute(targetCompute);
                    kv.SetCompute(targetCompute);
                    execution.Lease = targetLease;
                    targetLease = null;
                    accepted = true;
                    visitedDevices.Add(targetPci);
                    visitedOrdinals.Add(ordinal);
                    migrations++;
                    audit["gpu_migrations"] = migrations.ToString(System.Globalization.CultureInfo.InvariantCulture);
                    audit["gpu_migration_from_pci"] = previousPci;
                    audit["gpu_migration_to_pci"] = targetPci;
                    audit["gpu_migration_position"] = position.ToString(System.Globalization.CultureInfo.InvariantCulture);
                    audit["gpu_migration_layer"] = layer.ToString(System.Globalization.CultureInfo.InvariantCulture);
                    audit["gpu_migration_after_generated_tokens"] = generatedTokens.ToString(System.Globalization.CultureInfo.InvariantCulture);
                    audit["cuda_device_ordinal"] = ordinal.ToString(System.Globalization.CultureInfo.InvariantCulture);
                    previous.Dispose();
                    return;
                }
                finally
                {
                    targetLease?.Dispose();
                    if (!accepted) candidate.Dispose();
                }
            }

            throw new InferenceGovernanceException(new ErrorContext("No other permitted CUDA device meets the thermal and resource limits.",
                "device_thermal_migration_unavailable", true) { FailureKind = FailureKind.FailClosed });
        }

        bool CanMigrate(string targetPci)
        {
            var health = CheckHealth(targetPci);
            return health.IsSuccess && Admission.Evaluate(request, budget with
            {
                AvailableVramBytes = (long)Math.Min((ulong)configuredVram, health.Value.VramFreeBytes)
            }).IsSuccess;
        }
        var decoder = Encoding.UTF8.GetDecoder();
        var gen2 = GC.CollectionCount(2);
        var allocated = GC.GetTotalAllocatedBytes(precise: false);
        long maxStepTicks = 0;
        if (restored is null)
        {
            for (var pos = 0; pos < tokens.Count - 1; pos++)
                state.Forward(tokens[pos], pos, token);
        }
        audit["local_prefill_tokens"] = (restored is null ? tokens.Count - 1 : 0)
            .ToString(System.Globalization.CultureInfo.InvariantCulture);
        if (continuation.SavePath is { } savePath)
        {
            Span<byte> checkpoint = stackalloc byte[40];
            WriteContinuation(tokens, checkpoint);
            Require(kv.Hibernate(savePath, checkpoint, continuation.AllowPlaintext));
        }

        var position = tokens.Count - 1;
        var current = tokens[^1];
        var finish = "length";
        for (var step = 0; step < maxNewTokens && position < shape.Context; step++, position++)
        {
            var stepStarted = Stopwatch.GetTimestamp();
            state.Forward(current, position, token);
            var next = state.Predict(token, step == 0 ? continuation.FirstLogits.Span : default);
            maxStepTicks = Math.Max(maxStepTicks, Stopwatch.GetTimestamp() - stepStarted);
            if (next is LlamaTokenizer.Begin or LlamaTokenizer.End)
            {
                finish = "stop";
                break;
            }

            generatedTokens++;
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
        audit["qos_preemptions"] = execution.Preemptions.ToString(System.Globalization.CultureInfo.InvariantCulture);
        audit["local_forward_layers"] = forwardLayers.ToString(System.Globalization.CultureInfo.InvariantCulture);
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
            finally
            {
                ownedTelemetry?.Dispose();
                mapped?.Dispose();
            }
        }
    }

    private void WriteContinuation(List<int> tokens, Span<byte> state)
    {
        BinaryPrimitives.WriteInt32LittleEndian(state, tokens.Count - 1);
        BinaryPrimitives.WriteInt32LittleEndian(state[4..], tokens[^1]);
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        Span<byte> asset = stackalloc byte[SHA256.HashSizeInBytes];
        using (var file = File.OpenRead(settings.Checkpoint))
            SHA256.HashData(file, asset);
        hash.AppendData(asset);
        foreach (var path in new[] { settings.Tokenizer, settings.Config })
        {
            asset.Clear();
            if (path is not null)
            {
                using var file = File.OpenRead(path);
                SHA256.HashData(file, asset);
            }
            hash.AppendData(asset);
        }
        hash.AppendData(MemoryMarshal.AsBytes(CollectionsMarshal.AsSpan(tokens)));
        hash.GetHashAndReset(state[8..]);
    }

    private void ValidateContinuation(VirtualMemorySpace restored, List<int> tokens)
    {
        Span<byte> actual = stackalloc byte[40];
        Span<byte> expected = stackalloc byte[40];
        if (restored.ReadSnapshotState(actual).Value != actual.Length)
            throw new InvalidDataException("The snapshot has no compatible model continuation.");
        WriteContinuation(tokens, expected);
        if (!CryptographicOperations.FixedTimeEquals(actual, expected))
            throw new InvalidDataException("The snapshot model or prompt does not match the request.");
    }

    private Result<DeviceHealthSnapshot> CheckHealth(string? targetPci = null)
        => DeviceHealthPolicy.Evaluate(telemetry is not null && (targetPci ?? pciBusId) is { } id
            ? telemetry.Read(id) : Result<DeviceHealthSnapshot>.Fail(HealthUnavailable),
            settings.HealthLimits, settings.Priority);

    private static void RequireMigration<T>(Result<T> result)
    {
        if (!result.IsSuccess)
            throw new InferenceGovernanceException(new ErrorContext(result.Error!.Message, "device_thermal_migration_failed", false)
            { FailureKind = FailureKind.FailClosed });
    }

    private sealed class DeviceExecution : IDisposable
    {
        private int preemptions;
        internal LayerQosScheduler.Lease? Lease { get; set; }
        internal int Preemptions => preemptions + (Lease?.Preemptions ?? 0);

        internal void Release()
        {
            preemptions = Preemptions;
            Lease?.Dispose();
            Lease = null;
        }

        public void Dispose() => Release();
    }

    private static T Require<T>(Result<T> result)
        => result.IsSuccess ? result.Value! : throw new InferenceGovernanceException(result.Error!);

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

    private sealed unsafe class ForwardState(LlamaCheckpoint shape, PagedLlamaWeights weights, TransformerCompute compute, PagedKvCache kv, PagedLlamaSettings settings, Func<int, int, TransformerCompute> checkpoint) : IDisposable
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

        public void Dispose()
        {
            if (settings.SwapPolicy.ZeroWipe)
            {
                CryptographicOperations.ZeroMemory(MemoryMarshal.AsBytes(new Span<float>(scratch,
                    2 * shape.HiddenDimension + 2 * shape.Vocabulary)));
                activation.AsSpan().Clear();
                normalized.AsSpan().Clear();
                query.AsSpan().Clear();
                key.AsSpan().Clear();
                value.AsSpan().Clear();
                attended.AsSpan().Clear();
                projected.AsSpan().Clear();
            }
            NativeMemory.Free(scratch);
        }

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
                compute = checkpoint(position, index);
            }
        }

        internal int Predict(CancellationToken token, Span<float> firstLogits)
        {
            Normalize(weights.ReadRow(shape.FinalNorm, 0));
            var logits = Logits;
            weights.Multiply(shape.Classifier, normalized, logits, token);
            if (!firstLogits.IsEmpty) logits.CopyTo(firstLogits);
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

internal readonly record struct LocalContinuation(string? SavePath, string? RestorePath,
    Memory<float> FirstLogits, bool AllowPlaintext = false);

internal sealed class InferenceGovernanceException(ErrorContext error) : Exception(error.Message)
{
    internal ErrorContext Error { get; } = error;
}

internal sealed class LocalContextLengthException : Exception;
