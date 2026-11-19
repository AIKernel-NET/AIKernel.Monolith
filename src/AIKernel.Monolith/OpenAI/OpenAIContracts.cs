using System.Text.Json;
using System.Text.Json.Serialization;

namespace AIKernel.Monolith.OpenAI;

/// <summary>
/// EN: OpenAI-compatible chat completion request. JA: OpenAI 互換 chat completion request です。
/// </summary>
public sealed record ChatCompletionRequest(
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("messages")] IReadOnlyList<ChatMessage> Messages)
{
    /// <summary>Gets whether to return an event stream.</summary>
    [JsonPropertyName("stream")]
    public bool? Stream { get; init; }
}

/// <summary>
/// EN: OpenAI-compatible chat message. JA: OpenAI 互換 chat message です。
/// </summary>
public sealed record ChatMessage(
    [property: JsonPropertyName("role")] string Role,
    [property: JsonPropertyName("content")] string Content) : AIKernel.Abstractions.Providers.IModelMessage
{
    internal JsonElement Parts { get; init; }
}

internal sealed record ChatApiMessage([property: JsonPropertyName("role")] string? Role,
    [property: JsonPropertyName("content")] JsonElement Content);
internal sealed record ChatApiRequest([property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("messages")] ChatApiMessage?[]? Messages,
    [property: JsonPropertyName("stream")] bool? Stream = null)
{
    internal ChatMessage[]? ReadMessages()
    {
        if (Messages is not { Length: > 0 and <= 256 }) return null;
        var result = new ChatMessage[Messages.Length];
        for (int i=0; i<result.Length; i++)
        {
            var message = Messages[i];
            if (message?.Role is null) return null;
            if (message.Content.ValueKind == JsonValueKind.String)
                result[i] = new(message.Role,message.Content.GetString()!);
            else if (message.Role == "user" && message.Content.ValueKind == JsonValueKind.Array)
                result[i] = new(message.Role,"") { Parts=message.Content };
            else return null;
        }
        return result;
    }
}

/// <summary>
/// EN: OpenAI-compatible completion request. JA: OpenAI 互換 completion request です。
/// </summary>
public sealed record CompletionRequest(
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("prompt")] string Prompt)
{
    /// <summary>Gets whether to return an event stream.</summary>
    [JsonPropertyName("stream")]
    public bool? Stream { get; init; }
}

/// <summary>
/// EN: OpenAI-compatible embedding request. JA: OpenAI 互換 embedding request です。
/// </summary>
public sealed record EmbeddingRequest(
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("input")] string Input);

/// <summary>
/// EN: OpenAI-compatible model list response. JA: OpenAI 互換 model list response です。
/// </summary>
public sealed record ModelsResponse(
    [property: JsonPropertyName("object")] string Object,
    [property: JsonPropertyName("data")] IReadOnlyList<ModelObject> Data);

/// <summary>
/// EN: OpenAI-compatible model object. JA: OpenAI 互換 model object です。
/// </summary>
public sealed record ModelObject(
    [property: JsonPropertyName("id")] string Id,
    [property: JsonPropertyName("object")] string Object,
    [property: JsonPropertyName("owned_by")] string OwnedBy);

/// <summary>
/// EN: OpenAI-compatible completion response. JA: OpenAI 互換 completion response です。
/// </summary>
public sealed record CompletionResponse(
    [property: JsonPropertyName("id")] string Id,
    [property: JsonPropertyName("object")] string Object,
    [property: JsonPropertyName("choices")] IReadOnlyList<CompletionChoice> Choices);

/// <summary>
/// EN: OpenAI-compatible completion choice. JA: OpenAI 互換 completion choice です。
/// </summary>
public sealed record CompletionChoice(
    [property: JsonPropertyName("text")] string Text,
    [property: JsonPropertyName("index")] int Index,
    [property: JsonPropertyName("finish_reason")] string FinishReason);

/// <summary>
/// EN: OpenAI-compatible chat completion response. JA: OpenAI 互換 chat completion response です。
/// </summary>
public sealed record ChatCompletionResponse(
    [property: JsonPropertyName("id")] string Id,
    [property: JsonPropertyName("object")] string Object,
    [property: JsonPropertyName("choices")] IReadOnlyList<ChatChoice> Choices);

/// <summary>
/// EN: OpenAI-compatible chat choice. JA: OpenAI 互換 chat choice です。
/// </summary>
public sealed record ChatChoice(
    [property: JsonPropertyName("index")] int Index,
    [property: JsonPropertyName("message")] ChatResponseMessage Message,
    [property: JsonPropertyName("finish_reason")] string FinishReason);

/// <summary>
/// EN: OpenAI-compatible chat response message. JA: OpenAI 互換 chat response message です。
/// </summary>
public sealed record ChatResponseMessage(
    [property: JsonPropertyName("role")] string Role,
    [property: JsonPropertyName("content")] string Content);

/// <summary>
/// EN: OpenAI-compatible embedding response. JA: OpenAI 互換 embedding response です。
/// </summary>
public sealed record EmbeddingResponse(
    [property: JsonPropertyName("object")] string Object,
    [property: JsonPropertyName("data")] IReadOnlyList<EmbeddingData> Data,
    [property: JsonPropertyName("model")] string Model);

/// <summary>
/// EN: OpenAI-compatible embedding item. JA: OpenAI 互換 embedding item です。
/// </summary>
public sealed record EmbeddingData(
    [property: JsonPropertyName("object")] string Object,
    [property: JsonPropertyName("index")] int Index,
    [property: JsonPropertyName("embedding")] IReadOnlyList<float> Embedding);

internal sealed record OpenAIErrorResponse(
    [property: JsonPropertyName("error")] OpenAIError Error);

internal sealed record OpenAIError(
    [property: JsonPropertyName("message")] string Message,
    [property: JsonPropertyName("code")] string Code,
    [property: JsonPropertyName("retry_after_seconds")]
    [property: JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] int? RetryAfterSeconds = null);

internal sealed record EmbeddingApiRequest(
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("input")] JsonElement Input,
    [property: JsonPropertyName("encoding_format")] string? EncodingFormat)
{
    internal IReadOnlyList<string>? GetTexts()
    {
        if (EncodingFormat is not (null or "float"))
        {
            return null;
        }

        if (Input.ValueKind == JsonValueKind.String)
        {
            var text = Input.GetString();
            return string.IsNullOrWhiteSpace(text) ? null : [text];
        }

        if (Input.ValueKind != JsonValueKind.Array || Input.GetArrayLength() is 0 or > 2048)
        {
            return null;
        }

        var texts = new List<string>(Input.GetArrayLength());
        foreach (var item in Input.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(item.GetString()))
            {
                return null;
            }
            texts.Add(item.GetString()!);
        }
        return texts;
    }
}

internal sealed record ChatCompletionChunk(
    [property: JsonPropertyName("id")] string Id,
    [property: JsonPropertyName("object")] string Object,
    [property: JsonPropertyName("created")] long Created,
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("choices")] IReadOnlyList<ChatStreamChoice> Choices);

internal sealed record ChatStreamChoice(
    [property: JsonPropertyName("index")] int Index,
    [property: JsonPropertyName("delta")] ChatStreamDelta Delta,
    [property: JsonPropertyName("finish_reason")] string? FinishReason);

internal sealed record ChatStreamDelta(
    [property: JsonPropertyName("role"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] string? Role,
    [property: JsonPropertyName("content"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] string? Content);

internal sealed record TextCompletionChunk(
    [property: JsonPropertyName("id")] string Id,
    [property: JsonPropertyName("object")] string Object,
    [property: JsonPropertyName("created")] long Created,
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("choices")] IReadOnlyList<TextStreamChoice> Choices);

internal sealed record TextStreamChoice(
    [property: JsonPropertyName("text")] string Text,
    [property: JsonPropertyName("index")] int Index,
    [property: JsonPropertyName("finish_reason")] string? FinishReason);
