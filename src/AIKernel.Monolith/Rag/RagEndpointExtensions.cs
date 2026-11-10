using System.Text.Json.Serialization.Metadata;
using AIKernel.Monolith.Internal;
using AIKernel.Monolith.Models;
using AIKernel.Monolith.OpenAI;

namespace AIKernel.Monolith.Rag;

internal static class RagEndpointExtensions
{
    internal static void MapRagApi(this IEndpointRouteBuilder endpoints)
    {
        var collection = endpoints.MapGroup("/v1/rag/indexes");
        collection.MapGet("", (RagService rag, CancellationToken cancellationToken) =>
            ToResult(rag.ListIndexes(cancellationToken), RagJsonContext.Default.RagIndexListResponse));

        var indexes = collection.MapGroup("/{indexId}");
        indexes.MapDelete("", async (string indexId, RagService rag, CancellationToken cancellationToken) =>
            await rag.DeleteIndexAsync(indexId, cancellationToken) is { } error
                ? ErrorResponse(error)
                : Results.NoContent());

        indexes.MapGet("/documents", async (string indexId, RagService rag, CancellationToken cancellationToken) =>
            ToResult(await rag.ListDocumentsAsync(indexId, cancellationToken), RagJsonContext.Default.RagDocumentListResponse));

        indexes.MapPut("", async (string indexId, RagIndexRequest request, RagService rag,
            HttpContext context, CancellationToken cancellationToken) =>
            ToResult(await rag.IndexAsync(indexId, request, context.TraceIdentifier, cancellationToken),
                RagJsonContext.Default.RagIndexResponse));

        indexes.MapPut("/documents", async (string indexId, RagDocument request, RagService rag,
            HttpContext context, CancellationToken cancellationToken) =>
            ToResult(await rag.UpsertDocumentAsync(indexId, request, context.TraceIdentifier, cancellationToken),
                RagJsonContext.Default.RagIndexResponse));

        indexes.MapDelete("/documents", async (string indexId, string? documentId, RagService rag,
            CancellationToken cancellationToken) =>
            ToResult(await rag.DeleteDocumentAsync(indexId, documentId, cancellationToken),
                RagJsonContext.Default.RagIndexResponse));

        indexes.MapPost("/search", async (string indexId, RagSearchRequest request, RagService rag,
            HttpContext context, CancellationToken cancellationToken) =>
            ToResult(await rag.SearchAsync(indexId, request.Query, request.TopK, context.TraceIdentifier, cancellationToken),
                RagJsonContext.Default.RagSearchResponse));

        indexes.MapPost("/answer", async (string indexId, RagAnswerRequest request, RagService rag,
            HttpContext context, CancellationToken cancellationToken) =>
            ToResult(await rag.AnswerAsync(indexId, request, context.TraceIdentifier, cancellationToken),
                RagJsonContext.Default.RagAnswerResponse));
    }

    private static IResult ToResult<T>(InferenceOutcome<T> outcome, JsonTypeInfo<T> typeInfo) where T : class
        => outcome.Error is { } error
            ? ErrorResponse(error)
            : Results.Json(outcome.Value!, typeInfo);

    private static IResult ErrorResponse(InferenceError error)
        => Results.Json(new OpenAIErrorResponse(new OpenAIError(error.Message, error.Code)),
            InferenceJsonContext.Default.OpenAIErrorResponse, statusCode: error.Status);
}
