using AIKernel.Monolith.Hosting;
using AIKernel.Monolith.Models;
using AIKernel.Monolith.OpenAI;
using AIKernel.Monolith.VFS;
using System.Text.Json.Serialization;

namespace AIKernel.Monolith.Internal;

/// <summary>
/// EN: JSON source-generation context for Native AOT-safe API responses. JA: Native AOT 安全な API 応答向け JSON source-generation context です。
/// </summary>
[JsonSerializable(typeof(RuntimeDescription))]
[JsonSerializable(typeof(HealthResponse))]
[JsonSerializable(typeof(ModelsResponse))]
[JsonSerializable(typeof(ModelObject))]
[JsonSerializable(typeof(CompletionResponse))]
[JsonSerializable(typeof(CompletionChoice))]
[JsonSerializable(typeof(ChatCompletionResponse))]
[JsonSerializable(typeof(ChatChoice))]
[JsonSerializable(typeof(ChatResponseMessage))]
[JsonSerializable(typeof(EmbeddingResponse))]
[JsonSerializable(typeof(EmbeddingData))]
[JsonSerializable(typeof(ChatCompletionRequest))]
[JsonSerializable(typeof(ChatMessage))]
[JsonSerializable(typeof(CompletionRequest))]
[JsonSerializable(typeof(EmbeddingRequest))]
[JsonSerializable(typeof(MountDescription))]
[JsonSerializable(typeof(ModelMetadata))]
public sealed partial class MonolithJsonContext : JsonSerializerContext;
