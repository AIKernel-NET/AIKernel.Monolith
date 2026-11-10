using System.Text.Json.Serialization;
using AIKernel.Monolith.Rag;

namespace AIKernel.Monolith.Internal;

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
[JsonSerializable(typeof(RagIndexRequest))]
[JsonSerializable(typeof(RagDocument))]
[JsonSerializable(typeof(RagIndexResponse))]
[JsonSerializable(typeof(RagSearchRequest))]
[JsonSerializable(typeof(RagSearchResponse))]
[JsonSerializable(typeof(RagAnswerRequest))]
[JsonSerializable(typeof(RagAnswerResponse))]
[JsonSerializable(typeof(RagHit[]))]
[JsonSerializable(typeof(RagIndex))]
internal sealed partial class RagJsonContext : JsonSerializerContext;
