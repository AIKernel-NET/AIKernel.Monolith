using System.Text.Json.Serialization;

namespace AIKernel.Monolith.OpenAI;

/// <summary>
/// EN: OpenAI-compatible chat completion request. JA: OpenAI 互換 chat completion request です。
/// </summary>
public sealed record ChatCompletionRequest(
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("messages")] IReadOnlyList<ChatMessage> Messages);

/// <summary>
/// EN: OpenAI-compatible chat message. JA: OpenAI 互換 chat message です。
/// </summary>
public sealed record ChatMessage(
    [property: JsonPropertyName("role")] string Role,
    [property: JsonPropertyName("content")] string Content);

/// <summary>
/// EN: OpenAI-compatible completion request. JA: OpenAI 互換 completion request です。
/// </summary>
public sealed record CompletionRequest(
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("prompt")] string Prompt);

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
