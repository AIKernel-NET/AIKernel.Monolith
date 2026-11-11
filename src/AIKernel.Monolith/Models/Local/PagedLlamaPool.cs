using System.Collections.Concurrent;
using System.Text.Json.Serialization;
using AIKernel.Monolith.OpenAI;

namespace AIKernel.Monolith.Models.Local;

internal sealed class PagedLlamaPool : IDisposable
{
    private readonly ConcurrentDictionary<string, Session> sessions = new(StringComparer.Ordinal);

    internal async Task<string> GenerateAsync(string id, PagedLlamaSettings settings, IReadOnlyList<ChatMessage> messages,
        Func<string, Task> emit, Dictionary<string, string> audit, CancellationToken token)
    {
        var session = sessions.GetOrAdd(id, static _ => new Session());
        await session.Lock.WaitAsync(token);
        try
        {
            session.Model ??= new PagedLlamaModel(settings, token);
            return await session.Model.GenerateAsync(messages, emit, audit, token);
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or UnauthorizedAccessException
            or InvalidOperationException or DllNotFoundException or EntryPointNotFoundException
            or NotSupportedException or OutOfMemoryException or System.Text.DecoderFallbackException)
        {
            throw new LocalModelException(ex);
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

internal sealed record PagedLlamaSettings(string Checkpoint, string Tokenizer, string SwapDirectory,
    int BlockSize, int RamBlocks, int VramBlocks, int Device, int ContextTokens, int MaxNewTokens);

[JsonUnmappedMemberHandling(JsonUnmappedMemberHandling.Disallow)]
internal sealed record PagedLlamaRegistration([property: JsonPropertyName("checkpoint")] string? Checkpoint,
    [property: JsonPropertyName("tokenizer")] string? Tokenizer,
    [property: JsonPropertyName("blockSize")] int? BlockSize,
    [property: JsonPropertyName("ramBlocks")] int? RamBlocks,
    [property: JsonPropertyName("vramBlocks")] int? VramBlocks,
    [property: JsonPropertyName("device")] int? Device,
    [property: JsonPropertyName("contextTokens")] int? ContextTokens,
    [property: JsonPropertyName("maxNewTokens")] int? MaxNewTokens);

internal sealed class LocalModelException(Exception inner) : Exception("Local model execution failed.", inner);
