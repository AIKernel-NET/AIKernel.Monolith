using System.Text.Json;
using AIKernel.Monolith.Internal;
using AIKernel.Monolith.Models;
using Microsoft.AspNetCore.Http.Features;

namespace AIKernel.Monolith.OpenAI;

internal sealed class OpenAIStreamingResponse(HttpContext context, string model, bool chat, CancellationToken token)
{
    private readonly long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
    private bool sentChunk;

    internal string Id { get; } = GovernedModelInference.CreateCompletionId();

    internal Task WriteChunkAsync(string text) => WriteCompletionAsync(text, finishReason: null);

    internal async Task CompleteAsync(string finishReason)
    {
        await WriteCompletionAsync(null, finishReason);
        await WriteEventAsync("[DONE]");
    }

    internal Task WriteErrorAsync(InferenceError error) =>
        WriteEventAsync(JsonSerializer.Serialize(new OpenAIErrorResponse(new OpenAIError(error.Message, error.Code)),
            InferenceJsonContext.Default.OpenAIErrorResponse));

    private Task WriteCompletionAsync(string? content, string? finishReason)
    {
        var first = !sentChunk;
        sentChunk = true;
        var data = chat
            ? JsonSerializer.Serialize(new ChatCompletionChunk(Id, "chat.completion.chunk", created, model,
                [new ChatStreamChoice(0, new ChatStreamDelta(first ? "assistant" : null, content), finishReason)]),
                InferenceJsonContext.Default.ChatCompletionChunk)
            : JsonSerializer.Serialize(new TextCompletionChunk(Id, "text_completion", created, model,
                [new TextStreamChoice(content ?? string.Empty, 0, finishReason)]),
                InferenceJsonContext.Default.TextCompletionChunk);
        return WriteEventAsync(data);
    }

    internal async Task WriteEventAsync(string data)
    {
        var response = context.Response;
        if (!response.HasStarted)
        {
            context.Features.Get<IHttpResponseBodyFeature>()?.DisableBuffering();
            response.ContentType = "text/event-stream";
            response.Headers.CacheControl = "no-cache";
        }

        await response.WriteAsync("data: " + data + "\n\n", token);
        await response.Body.FlushAsync(token);
    }
}
