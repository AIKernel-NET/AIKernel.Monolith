using System.Text.Json.Serialization;
using AIKernel.Monolith.OpenAI;

namespace AIKernel.Monolith.Rag;

internal sealed record RagIndexRequest(string? EmbeddingModel, RagDocument?[]? Documents);
internal sealed record RagDocument(string? Id, string? Text, string? Title, string? Source);
internal sealed record RagIndexResponse(string IndexId, string EmbeddingModel, int DocumentCount, int ChunkCount);

internal sealed record RagIndexListResponse(string[] Indexes);
internal sealed record RagDocumentListResponse(string IndexId, RagDocumentSummary[] Documents);
internal sealed record RagDocumentSummary(string DocumentId, string? Title, string? Source, int ChunkCount);

internal sealed record RagSearchRequest(string? Query, int? TopK, RagFilter? Filter, string? SearchMode);
internal sealed record RagAnswerRequest(string? Model, string? Question, int? TopK, RagFilter? Filter, bool? Stream, string? SearchMode);
internal sealed record RagAnswerContext(string Model, ChatMessage[] Messages, RagHit[] Sources);
internal sealed record RagAnswerSources(string Id, string Model, RagHit[] Sources);

[JsonUnmappedMemberHandling(JsonUnmappedMemberHandling.Disallow)]
internal sealed record RagFilter(string?[]? DocumentIds, string?[]? Sources);
internal sealed record RagSearchResponse(string IndexId, RagHit[] Hits);
internal sealed record RagAnswerResponse(string Id, string Model, string Answer, RagHit[] Sources);

internal sealed record RagSource(string DocumentId, int ChunkNumber, string? Title, string? Source);
internal sealed record RagHit(string Citation, RagSource Source, string Text, double Score);

internal sealed record RagIndex(RagEmbeddingModel EmbeddingModel, RagChunk[] Chunks);
internal sealed record RagEmbeddingModel(string Id, string Provider, string Endpoint, string UpstreamModel);
internal sealed record RagChunk(RagSource Source, string Text, float[] Vector);
