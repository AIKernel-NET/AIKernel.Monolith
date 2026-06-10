using AIKernel.Monolith.Models;
using AIKernel.Monolith.Internal;

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

        endpoints.MapPost("/v1/completions", (CompletionRequest request, ModelProviderRegistry registry) =>
        {
            var completion = registry.Complete(new ModelCompletionRequest(request.Model, request.Prompt));
            return Results.Json(
                new CompletionResponse(completion.Id, "text_completion", [new CompletionChoice(completion.Text, 0, "stop")]),
                MonolithJsonContext.Default.CompletionResponse);
        });

        endpoints.MapPost("/v1/chat/completions", (ChatCompletionRequest request, ModelProviderRegistry registry) =>
        {
            var prompt = string.Join("\n", request.Messages.Select(x => $"{x.Role}: {x.Content}"));
            var completion = registry.Complete(new ModelCompletionRequest(request.Model, prompt));
            return Results.Json(
                new ChatCompletionResponse(completion.Id, "chat.completion", [new ChatChoice(0, new ChatResponseMessage("assistant", completion.Text), "stop")]),
                MonolithJsonContext.Default.ChatCompletionResponse);
        });

        endpoints.MapPost("/v1/embeddings", (EmbeddingRequest request) =>
        {
            var vector = request.Input.Select(ch => (float)(ch % 23) / 23f).Take(16).ToArray();
            return Results.Json(
                new EmbeddingResponse("list", [new EmbeddingData("embedding", 0, vector)], request.Model),
                MonolithJsonContext.Default.EmbeddingResponse);
        });

        return endpoints;
    }
}
