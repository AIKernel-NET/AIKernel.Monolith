using System.Collections.Concurrent;
using System.Text.Json;
using System.Text.Json.Serialization;
using AIKernel.Common.Results;
using AIKernel.Core.Governance;
using AIKernel.Core.Memory;
using AIKernel.Kernel.Governance;
using AIKernel.Monolith.OpenAI;

namespace AIKernel.Monolith.Models.Local;

internal sealed class PagedLlamaPool(ILogger<PagedLlamaModel> logger) : IDisposable
{
    private readonly ConcurrentDictionary<string, Session> sessions = new(StringComparer.Ordinal);
    private readonly ConcurrentDictionary<int, LayerQosScheduler> devices = new();

    internal async Task<Result<string>> GenerateAsync(string id, PagedLlamaSettings settings, IReadOnlyList<ChatMessage> messages,
        Func<string, Task> emit, Dictionary<string, string> audit, CancellationToken token)
    {
        var session = sessions.GetOrAdd(id, static _ => new Session());
        await session.Lock.WaitAsync(token);
        try
        {
            var model = session.Model;
            if (model is null)
            {
                model = new PagedLlamaModel(settings, token, logger: logger);
                Volatile.Write(ref session.Model, model);
            }
            return Result<string>.Success(await model.GenerateAsync(messages, emit, audit, null, token,
                schedulerForDevice: GetScheduler));
        }
        catch (InferenceGovernanceException ex)
        {
            return Result<string>.Fail(ex.Error);
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or UnauthorizedAccessException
            or InvalidOperationException or DllNotFoundException or EntryPointNotFoundException
            or NotSupportedException or OutOfMemoryException or System.Text.DecoderFallbackException or ArgumentException
            or JsonException or KeyNotFoundException or OverflowException or FormatException or AggregateException
            or UnsupportedModelException or LocalContextLengthException or System.Security.Cryptography.CryptographicException)
        {
            var code = ex switch
            {
                LocalContextLengthException => "context_length_exceeded",
                UnsupportedModelException => "unsupported_model_format",
                OutOfMemoryException => "local_memory_exhausted",
                _ => "provider_error"
            };
            return Result<string>.Fail(new ErrorContext(ex.Message, code, false)
            {
                FailureKind = FailureKind.FailClosed,
                OriginStep = OriginStep.Provider
            });
        }
        finally
        {
            session.Lock.Release();
        }
    }

    internal PagedLlamaPlacement? GetPlacement(string id)
        => sessions.TryGetValue(id, out var session) ? Volatile.Read(ref session.Model)?.Placement : null;

    private LayerQosScheduler GetScheduler(int ordinal) => devices.GetOrAdd(ordinal, static _ => new());

    public void Dispose()
    {
        foreach (var session in sessions.Values)
        {
            try
            {
                session.Model?.Dispose();
            }
            finally
            {
                session.Lock.Dispose();
            }
        }
    }

    private sealed class Session
    {
        internal SemaphoreSlim Lock { get; } = new(1);
        internal PagedLlamaModel? Model;
    }
}

internal sealed record PagedLlamaPlacement(int? Device, string? PciBusId = null, bool RecoveryPending = false);

internal sealed record PagedLlamaSettings(string Checkpoint, string? Tokenizer, string? Config, string SwapDirectory,
    int BlockSize, int RamBlocks, int VramBlocks, int Device, int ContextTokens, int MaxNewTokens,
    int KvRamBlocks, int KvVramBlocks, float Temperature, float TopP, InferencePriority Priority,
    long MaxPcieSwapBytes, DeviceHealthLimits HealthLimits, SwapEncryptionPolicy SwapPolicy,
    int[]? ThermalMigrationDevices = null, IReadOnlyDictionary<int, DeviceThermalLimits>? ThermalDeviceLimits = null)
{
    internal DeviceHealthLimits GetHealthLimits(int device, bool migrationTarget = false)
        => ThermalDeviceLimits?.TryGetValue(device, out var limits) == true
            ? HealthLimits with
            {
                MaximumTemperatureCelsius = migrationTarget ? limits.MigrationTargetTemperatureCelsius : limits.TemperatureCelsius
            }
            : HealthLimits;
}

internal readonly record struct DeviceThermalLimits(uint TemperatureCelsius, uint MigrationTargetTemperatureCelsius);

[JsonUnmappedMemberHandling(JsonUnmappedMemberHandling.Disallow)]
internal sealed record PagedLlamaRegistration([property: JsonPropertyName("checkpoint")] string? Checkpoint,
    [property: JsonPropertyName("tokenizer")] string? Tokenizer,
    [property: JsonPropertyName("config")] string? Config,
    [property: JsonPropertyName("blockSize")] int? BlockSize,
    [property: JsonPropertyName("ramBlocks")] int? RamBlocks,
    [property: JsonPropertyName("vramBlocks")] int? VramBlocks,
    [property: JsonPropertyName("device")] int? Device,
    [property: JsonPropertyName("contextTokens")] int? ContextTokens,
    [property: JsonPropertyName("maxNewTokens")] int? MaxNewTokens,
    [property: JsonPropertyName("kvRamBlocks")] int? KvRamBlocks,
    [property: JsonPropertyName("kvVramBlocks")] int? KvVramBlocks,
    [property: JsonPropertyName("temperature")] float? Temperature,
    [property: JsonPropertyName("topP")] float? TopP,
    [property: JsonPropertyName("priority")] string? Priority,
    [property: JsonPropertyName("maxPcieSwapBytes")] long? MaxPcieSwapBytes,
    [property: JsonPropertyName("maxTemperatureCelsius")] uint? MaxTemperatureCelsius,
    [property: JsonPropertyName("batchPowerLimitPercent")] uint? BatchPowerLimitPercent,
    [property: JsonPropertyName("encryptSwap")] bool? EncryptSwap,
    [property: JsonPropertyName("zeroWipe")] bool? ZeroWipe,
    [property: JsonPropertyName("thermalMigrationDevices")] int[]? ThermalMigrationDevices,
    [property: JsonPropertyName("thermalDeviceLimits")] ThermalDeviceLimitRegistration?[]? ThermalDeviceLimits);

[JsonUnmappedMemberHandling(JsonUnmappedMemberHandling.Disallow)]
internal sealed record ThermalDeviceLimitRegistration(
    [property: JsonPropertyName("device")] int? Device,
    [property: JsonPropertyName("maxTemperatureCelsius")] uint? MaxTemperatureCelsius,
    [property: JsonPropertyName("maxMigrationTargetTemperatureCelsius")] uint? MaxMigrationTargetTemperatureCelsius);
