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

        endpoints.MapGet("/v1/models/admission", async (HttpContext context, ModelProviderRegistry registry,
            ModelAdmission admission, CancellationToken cancellationToken) =>
        {
            context.Response.Headers.CacheControl = "no-store";
            var query = context.Request.Query["model"];
            if (query.Count != 1 || string.IsNullOrWhiteSpace(query[0]))
                return ErrorResponse(InferenceErrors.ModelIdRequired);
            if (registry.Find(query[0]!) is not { } model)
                return ErrorResponse(InferenceErrors.ModelNotFound);

            var status = await admission.ReadAsync(model, context.TraceIdentifier, cancellationToken);
            return Results.Json(status, InferenceJsonContext.Default.ModelAdmissionStatus);
        });

        endpoints.MapPost("/v1/completions", async (CompletionRequest request, HttpContext context,
            GovernedModelInference inference, CancellationToken cancellationToken) =>
        {
            IReadOnlyList<ChatMessage>? messages = string.IsNullOrWhiteSpace(request.Prompt)
                ? null : [new ChatMessage("user", request.Prompt)];
            if (request.Stream == true)
            {
                return await StreamAsync(request.Model, messages, context, inference, "completions", chat: false, cancellationToken);
            }

            var outcome = await inference.CompleteAsync(request.Model, messages,
                context.TraceIdentifier, "completions", cancellationToken);
            if (outcome.Error is { } error)
            {
                return ErrorResponse(error);
            }

            var completion = outcome.Value!;
            return Results.Json(
                new CompletionResponse(completion.Id, "text_completion", [new CompletionChoice(completion.Text, 0, completion.FinishReason)]),
                MonolithJsonContext.Default.CompletionResponse);
        });

        endpoints.MapPost("/v1/chat/completions", async (ChatApiRequest request, HttpContext context,
            GovernedModelInference inference, CancellationToken cancellationToken) =>
        {
            var messages = request.ReadMessages();
            if (request.Stream == true)
            {
                return await StreamAsync(request.Model, messages, context, inference,
                    "chat.completions", chat: true, cancellationToken);
            }

            var outcome = await inference.CompleteAsync(request.Model, messages,
                context.TraceIdentifier, "chat.completions", cancellationToken);
            if (outcome.Error is { } error)
            {
                return ErrorResponse(error);
            }

            var completion = outcome.Value!;
            return Results.Json(
                new ChatCompletionResponse(completion.Id, "chat.completion", [new ChatChoice(0, new ChatResponseMessage("assistant", completion.Text), completion.FinishReason)]),
                MonolithJsonContext.Default.ChatCompletionResponse);
        });

        endpoints.MapPost("/v1/embeddings", async (EmbeddingApiRequest request, HttpContext context,
            GovernedModelInference inference, CancellationToken cancellationToken) =>
        {
            var outcome = await inference.EmbedAsync(request.Model, request.GetTexts(), context.TraceIdentifier, cancellationToken);
            if (outcome.Error is { } error)
            {
                return ErrorResponse(error);
            }

            var data = outcome.Value!.Select((vector, index) => new EmbeddingData("embedding", index, vector)).ToArray();
            return Results.Json(new EmbeddingResponse("list", data, request.Model),
                MonolithJsonContext.Default.EmbeddingResponse);
        });

        return endpoints;
    }

    private static async Task<IResult> StreamAsync(
        string model,
        IReadOnlyList<ChatMessage>? messages,
        HttpContext context,
        GovernedModelInference inference,
        string operation,
        bool chat,
        CancellationToken cancellationToken)
    {
        var stream = new OpenAIStreamingResponse(context, model, chat, cancellationToken);
        var outcome = await inference.StreamAsync(model, messages, context.TraceIdentifier, operation,
            stream.Id, stream.WriteChunkAsync, cancellationToken);
        if (outcome.Error is { } error)
        {
            if (!context.Response.HasStarted)
            {
                return ErrorResponse(error);
            }

            await stream.WriteErrorAsync(error);
            return Results.Empty;
        }

        await stream.CompleteAsync(outcome.Value!);
        return Results.Empty;
    }

    private static IResult ErrorResponse(InferenceError error) => new InferenceErrorResult(error);
}
