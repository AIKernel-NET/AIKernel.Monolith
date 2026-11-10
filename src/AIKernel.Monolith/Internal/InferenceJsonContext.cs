using System.Text.Json.Serialization;
using AIKernel.Dtos.Events;
using AIKernel.Monolith.Models;
using AIKernel.Monolith.OpenAI;

namespace AIKernel.Monolith.Internal;

[JsonSerializable(typeof(ModelRegistration))]
[JsonSerializable(typeof(ModelRegistration[]))]
[JsonSerializable(typeof(AuditEvent))]
[JsonSerializable(typeof(OpenAIErrorResponse))]
[JsonSerializable(typeof(EmbeddingApiRequest))]
[JsonSerializable(typeof(ChatCompletionChunk))]
[JsonSerializable(typeof(TextCompletionChunk))]
internal sealed partial class InferenceJsonContext : JsonSerializerContext;
