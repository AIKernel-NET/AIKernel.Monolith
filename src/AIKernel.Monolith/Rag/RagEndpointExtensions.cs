using System.Text.Json.Serialization.Metadata;
using AIKernel.Monolith.Internal;
using AIKernel.Monolith.Models;
using AIKernel.Monolith.OpenAI;

namespace AIKernel.Monolith.Rag;

internal static class RagEndpointExtensions
{
    internal static void MapRagApi(this IEndpointRouteBuilder endpoints)
    {
        var indexes = endpoints.MapGroup("/v1/rag/indexes/{indexId}");

        indexes.MapPut("", async (string indexId, RagIndexRequest request, RagService rag,
            HttpContext context, CancellationToken cancellationToken) =>
            ToResult(await rag.IndexAsync(indexId, request, context.TraceIdentifier, cancellationToken),
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
            ? Results.Json(new OpenAIErrorResponse(new OpenAIError(error.Message, error.Code)),
                InferenceJsonContext.Default.OpenAIErrorResponse, statusCode: error.Status)
            : Results.Json(outcome.Value!, typeInfo);
}
