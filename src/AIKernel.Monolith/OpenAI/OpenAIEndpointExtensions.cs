using AIKernel.Monolith.Internal;
using AIKernel.Monolith.Models;

namespace AIKernel.Monolith.OpenAI;

/// <summary>
/// EN: Minimal API endpoints for the OpenAI-compatible surface. JA: OpenAI 互換 surface 向け Minimal API endpoint です。
/// </summary>
public static class OpenAIEndpointExtensions
{
    /// <summary>
    /// EN: Maps OpenAI-compatible endpoints. JA: OpenAI 互換 endpoint を map します。
    /// </summary>
    public static IEndpointRouteBuilder MapOpenAICompatibleApi(this IEndpointRouteBuilder endpoints)
    {
        endpoints.MapGet("/v1/models", (ModelProviderRegistry registry) => Results.Json(new
            ModelsResponse(
                "list",
                registry.ListModels().Select(x => new ModelObject(x.Id, "model", x.Provider)).ToArray()),
            MonolithJsonContext.Default.ModelsResponse));

        endpoints.MapPost("/v1/completions", async (CompletionRequest request, HttpContext context,
            GovernedModelInference inference, CancellationToken cancellationToken) =>
        {
            var outcome = await inference.CompleteAsync(request.Model,
                string.IsNullOrWhiteSpace(request.Prompt) ? null : [new ChatMessage("user", request.Prompt)],
                context.TraceIdentifier, "completions", cancellationToken);
            if (outcome.Error is { } error)
            {
                return ErrorResponse(error);
            }

            var completion = outcome.Completion!;
            return Results.Json(
                new CompletionResponse(completion.Id, "text_completion", [new CompletionChoice(completion.Text, 0, "stop")]),
                MonolithJsonContext.Default.CompletionResponse);
        });

        endpoints.MapPost("/v1/chat/completions", async (ChatCompletionRequest request, HttpContext context,
            GovernedModelInference inference, CancellationToken cancellationToken) =>
        {
            var outcome = await inference.CompleteAsync(request.Model, request.Messages,
                context.TraceIdentifier, "chat.completions", cancellationToken);
            if (outcome.Error is { } error)
            {
                return ErrorResponse(error);
            }

            var completion = outcome.Completion!;
            return Results.Json(
                new ChatCompletionResponse(completion.Id, "chat.completion", [new ChatChoice(0, new ChatResponseMessage("assistant", completion.Text), "stop")]),
                MonolithJsonContext.Default.ChatCompletionResponse);
        });

        endpoints.MapPost("/v1/embeddings", (EmbeddingRequest request, ModelProviderRegistry registry) =>
            ErrorResponse(string.IsNullOrWhiteSpace(request.Model)
                ? InferenceErrors.InvalidRequest
                : registry.Find(request.Model) is null ? InferenceErrors.ModelNotFound : InferenceErrors.EmbeddingsUnsupported));

        return endpoints;
    }

    private static IResult ErrorResponse(InferenceError error)
        => Results.Json(
            new OpenAIErrorResponse(new OpenAIError(error.Message, error.Code)),
            InferenceJsonContext.Default.OpenAIErrorResponse,
            statusCode: error.Status);
}
