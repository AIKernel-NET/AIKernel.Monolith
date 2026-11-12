using System.Text.Json;
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
            ToResult(await rag.SearchAsync(indexId, request, context.TraceIdentifier, cancellationToken),
                RagJsonContext.Default.RagSearchResponse));

        indexes.MapPost("/answer", async (string indexId, RagAnswerRequest request, RagService rag,
            GovernedModelInference inference, HttpContext context, CancellationToken cancellationToken) =>
        {
            if (request.Stream == true)
            {
                return await StreamAnswerAsync(indexId, request, rag, inference, context, cancellationToken);
            }

            return ToResult(await rag.AnswerAsync(indexId, request, context.TraceIdentifier, cancellationToken),
                RagJsonContext.Default.RagAnswerResponse);
        });
    }

    private static async Task<IResult> StreamAnswerAsync(
        string indexId, RagAnswerRequest request, RagService rag, GovernedModelInference inference,
        HttpContext context, CancellationToken cancellationToken)
    {
        var prepared = await rag.PrepareAnswerAsync(indexId, request, context.TraceIdentifier, cancellationToken);
        if (prepared.Error is { } prepareError)
        {
            return ErrorResponse(prepareError);
        }

        var answer = prepared.Value!;
        var stream = new OpenAIStreamingResponse(context, answer.Model, chat: true, cancellationToken);
        var sources = new RagAnswerSources(stream.Id, answer.Model, answer.Sources);
        var outcome = await inference.StreamAsync(answer.Model, answer.Messages, context.TraceIdentifier,
            "rag.answer", stream.Id, async text =>
            {
                await WriteSourcesAsync();
                await stream.WriteChunkAsync(text);
            }, cancellationToken);
        if (outcome.Error is { } error)
        {
            if (!context.Response.HasStarted)
            {
                return ErrorResponse(error);
            }

            await stream.WriteErrorAsync(error);
            return Results.Empty;
        }

        await WriteSourcesAsync();
        await stream.CompleteAsync(outcome.Value!);
        return Results.Empty;

        Task WriteSourcesAsync() => context.Response.HasStarted
            ? Task.CompletedTask
            : stream.WriteEventAsync(JsonSerializer.Serialize(sources, RagJsonContext.Default.RagAnswerSources));
    }

    private static IResult ToResult<T>(InferenceOutcome<T> outcome, JsonTypeInfo<T> typeInfo) where T : class
        => outcome.Error is { } error
            ? ErrorResponse(error)
            : Results.Json(outcome.Value!, typeInfo);

    private static IResult ErrorResponse(InferenceError error) => new InferenceErrorResult(error);
}
