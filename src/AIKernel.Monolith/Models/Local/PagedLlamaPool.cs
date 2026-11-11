using System.Collections.Concurrent;
using System.Text.Json;
using System.Text.Json.Serialization;
using AIKernel.Common.Results;
using AIKernel.Monolith.OpenAI;

namespace AIKernel.Monolith.Models.Local;

internal sealed class PagedLlamaPool : IDisposable
{
    private readonly ConcurrentDictionary<string, Session> sessions = new(StringComparer.Ordinal);

    internal async Task<Result<string>> GenerateAsync(string id, PagedLlamaSettings settings, IReadOnlyList<ChatMessage> messages,
        Func<string, Task> emit, Dictionary<string, string> audit, CancellationToken token)
    {
        var session = sessions.GetOrAdd(id, static _ => new Session());
        await session.Lock.WaitAsync(token);
        try
        {
            session.Model ??= new PagedLlamaModel(settings, token);
            return Result<string>.Success(await session.Model.GenerateAsync(messages, emit, audit, token));
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or UnauthorizedAccessException
            or InvalidOperationException or DllNotFoundException or EntryPointNotFoundException
            or NotSupportedException or OutOfMemoryException or System.Text.DecoderFallbackException or ArgumentException
            or JsonException or KeyNotFoundException or OverflowException or FormatException or AggregateException
            or UnsupportedModelException or LocalContextLengthException)
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
        internal PagedLlamaModel? Model { get; set; }
    }
}

internal sealed record PagedLlamaSettings(string Checkpoint, string? Tokenizer, string? Config, string SwapDirectory,
    int BlockSize, int RamBlocks, int VramBlocks, int Device, int ContextTokens, int MaxNewTokens,
    int KvRamBlocks, int KvVramBlocks, float Temperature, float TopP);

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
    [property: JsonPropertyName("topP")] float? TopP);
