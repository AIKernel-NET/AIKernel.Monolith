using System.Text.Json.Serialization;
using AIKernel.Dtos.Events;
using AIKernel.Monolith.Models;
using AIKernel.Monolith.OpenAI;

namespace AIKernel.Monolith.Internal;

[JsonSerializable(typeof(ModelRegistration))]
[JsonSerializable(typeof(ChatApiRequest))]
[JsonSerializable(typeof(ModelAdmissionStatus))]
[JsonSerializable(typeof(string[]), TypeInfoPropertyName = "CudaPciBusIds")]
[JsonSerializable(typeof(AuditEvent))]
[JsonSerializable(typeof(OpenAIErrorResponse))]
[JsonSerializable(typeof(EmbeddingApiRequest))]
[JsonSerializable(typeof(ChatCompletionChunk))]
[JsonSerializable(typeof(TextCompletionChunk))]
internal sealed partial class InferenceJsonContext : JsonSerializerContext;
