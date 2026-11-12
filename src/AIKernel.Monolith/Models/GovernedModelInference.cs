using System.Text;
using System.Text.Json;
using AIKernel.Abstractions.Events;
using AIKernel.Common.Results;
using AIKernel.Abstractions.Governance;
using AIKernel.Dtos.Events;
using AIKernel.Dtos.Governance;
using AIKernel.Enums;
using AIKernel.Enums.Governance;
using AIKernel.Monolith.OpenAI;
using AIKernel.Monolith.Models.Local;
using AIKernel.Providers.ChatOpenAI;

namespace AIKernel.Monolith.Models;

internal sealed class GovernedModelInference(
    ModelProviderRegistry registry,
    IDecisionGate gate,
    IAuditEventWriter audit,
    IHttpClientFactory clients,
    PagedLlamaPool localModels,
    ILogger<GovernedModelInference> logger)
{
    internal const string HttpClientName = "monolith-inference";

    internal Task<InferenceOutcome<ModelCompletion>> CompleteAsync(
        string? modelId,
        IReadOnlyList<ChatMessage>? messages,
        string requestId,
        string operation,
        CancellationToken cancellationToken)
    {
        var id = CreateCompletionId();
        return RunAsync<ModelCompletion>(modelId, requestId, operation, "chat",
            ValidateMessages(messages), id, async (provider, token) =>
            {
                var text = await provider.GenerateAsync(messages!, token);
                return string.IsNullOrEmpty(text) ? null : new ModelCompletion(id, text);
            }, cancellationToken, async (settings, metadata, token) =>
            {
                var text = new StringBuilder();
                var result = await localModels.GenerateAsync(modelId!, settings, messages!, chunk =>
                {
                    text.Append(chunk);
                    return Task.CompletedTask;
                }, metadata, token);
                return result.Bind(finish => text.Length == 0
                    ? Result<ModelCompletion>.Fail(new ErrorContext("Empty local model response.", "provider_error", false))
                    : Result<ModelCompletion>.Success(new ModelCompletion(id, text.ToString()) { FinishReason = finish }));
            });
    }

    internal Task<InferenceOutcome<string>> StreamAsync(
        string? modelId,
        IReadOnlyList<ChatMessage>? messages,
        string requestId,
        string operation,
        string completionId,
        Func<string, Task> onChunk,
        CancellationToken cancellationToken)
        => RunAsync<string>(modelId, requestId, operation, "chat",
            ValidateMessages(messages), completionId,
            async (provider, token) => await provider.StreamCompletionAsync(messages!, onChunk, token),
            cancellationToken, async (settings, metadata, token) =>
                await localModels.GenerateAsync(modelId!, settings, messages!, onChunk, metadata, token));

    internal Task<InferenceOutcome<IReadOnlyList<float[]>>> EmbedAsync(
        string? modelId,
        IReadOnlyList<string>? texts,
        string requestId,
        CancellationToken cancellationToken,
        string operation = "embeddings")
        => RunAsync<IReadOnlyList<float[]>>(modelId, requestId, operation, "embedding",
            texts is null ? InferenceErrors.InvalidEmbeddingRequest : null, null,
            async (provider, token) => await provider.EmbedBatchAsync(texts!, token),
            cancellationToken);

    internal static string CreateCompletionId() => $"cmpl-{Guid.CreateVersion7():N}";

    internal static async Task<(bool Allowed, GateDecisionKind Decision)> EvaluatePolicyAsync(
        IDecisionGate gate, RegisteredModel model, string operation, string requestId, CancellationToken cancellationToken)
    {
        var decision = await gate.EvaluateAsync(new DecisionGateRequest
        {
            OperationId = operation,
            CorrelationId = requestId,
            GateInput = model.Policy
        }, cancellationToken);

        return (decision is { Succeeded: true, Accepted: true, DecisionKind: GateDecisionKind.Allow },
            decision.DecisionKind);
    }

    private static InferenceError? ValidateMessages(IReadOnlyList<ChatMessage>? messages)
        => messages is { Count: > 0 } && messages.All(message => message is
            { Role: "system" or "developer" or "user" or "assistant", Content: not null })
            ? null
            : InferenceErrors.InvalidRequest;

    private async Task<InferenceOutcome<T>> RunAsync<T>(
        string? modelId,
        string requestId,
        string operation,
        string requiredKind,
        InferenceError? inputError,
        string? completionId,
        Func<ChatOpenAIProvider, CancellationToken, Task<T?>> execute,
        CancellationToken cancellationToken,
        Func<PagedLlamaSettings, Dictionary<string, string>, CancellationToken, Task<Result<T>>>? executeLocal = null) where T : class
    {
        var model = modelId is null ? null : registry.Find(modelId);
        var metadata = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["request_id"] = requestId,
            ["operation"] = operation
        };

        if (model is null)
        {
            return await RejectAsync(string.IsNullOrWhiteSpace(modelId)
                ? InferenceErrors.InvalidRequest : InferenceErrors.ModelNotFound);
        }

        metadata["provider_id"] = model.Metadata.Provider;
        if (model.Metadata.Kind != requiredKind)
        {
            return await RejectAsync(requiredKind == "embedding"
                ? InferenceErrors.EmbeddingsUnsupported : InferenceErrors.ChatUnsupported);
        }

        if (inputError is not null)
        {
            return await RejectAsync(inputError);
        }

        var (allowed, decision) = await EvaluatePolicyAsync(gate, model, operation, requestId, cancellationToken);
        metadata["decision"] = decision.ToString();
        metadata["logos"] = model.Policy.Logos.ToString();
        metadata["ethos"] = model.Policy.Ethos.ToString();
        metadata["pathos"] = model.Policy.Pathos.ToString();

        if (!await RecordAsync("inference.policy", allowed ? "allowed" : "denied",
                allowed ? AuditSeverity.Information : AuditSeverity.Warning))
        {
            return new(Error: InferenceErrors.AuditUnavailable);
        }

        if (!allowed)
        {
            return new(Error: InferenceErrors.PolicyDenied);
        }

        if (completionId is not null)
        {
            metadata["completion_id"] = completionId;
        }

        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            T? value;
            if (model.Local is { } local)
            {
                var result = await executeLocal!(local, metadata, cancellationToken);
                if (result.IsFailure)
                {
                    var error = result.Error!;
                    if (error.Code == "context_length_exceeded")
                        return await RejectAsync(InferenceErrors.ContextLengthExceeded);

                    metadata["local_error"] = error.Code;
                    logger.LogWarning("Local inference failed for request {RequestId}: {Failure}", requestId, error.Message);
                    return await ProviderFailedAsync(error.Code);
                }

                value = result.Value;
            }
            else
            {
                var apiKey = model.ApiKeyEnvironmentVariable is { } variable
                    ? Environment.GetEnvironmentVariable(variable)
                    : null;
                if (model.ApiKeyEnvironmentVariable is not null && string.IsNullOrWhiteSpace(apiKey))
                {
                    return await ProviderFailedAsync("credential_unavailable");
                }

                var settings = model.Settings! with { ApiKey = apiKey };
                using var httpClient = clients.CreateClient(HttpClientName);
                var provider = new ChatOpenAIProvider(settings, new ChatOpenAIClient(httpClient, settings));
                await provider.InitializeAsync();
                value = await execute(provider, cancellationToken);
            }

            cancellationToken.ThrowIfCancellationRequested();
            if (value is null)
            {
                return await ProviderFailedAsync("empty_provider_response");
            }

            return await RecordAsync("inference.execution", "succeeded", AuditSeverity.Information)
                ? new(Value: value)
                : new(Error: InferenceErrors.AuditUnavailable);
        }
        catch (Exception ex) when (ex is HttpRequestException or JsonException or InvalidDataException
            or IOException or OperationCanceledException)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                await RecordAsync("inference.execution", "canceled", AuditSeverity.Warning);
                cancellationToken.ThrowIfCancellationRequested();
            }

            return await ProviderFailedAsync("provider_error");
        }

        async Task<InferenceOutcome<T>> RejectAsync(InferenceError error)
            => await RecordAsync("inference.rejected", error.Code, AuditSeverity.Warning)
                ? new(Error: error)
                : new(Error: InferenceErrors.AuditUnavailable);

        async Task<InferenceOutcome<T>> ProviderFailedAsync(string reason)
            => await RecordAsync("inference.execution", reason, AuditSeverity.Error)
                ? new(Error: reason switch
                {
                    "unsupported_model_format" => InferenceErrors.UnsupportedLocalModel,
                    "local_memory_exhausted" => InferenceErrors.LocalMemoryExhausted,
                    "resource_budget_exceeded" or "resource_pcie_budget_exceeded" or "device_queue_full" or
                    "device_temperature_exceeded" or "device_thermal_migration_unavailable" or "device_power_throttled" or
                    "device_telemetry_unavailable" or "device_thermal_migration_failed"
                        => new InferenceError(503, reason, "The configured local inference route is unavailable.",
                            InferenceErrors.GetRetryAfterSeconds(reason)),
                    _ => InferenceErrors.ProviderFailed
                })
                : new(Error: InferenceErrors.AuditUnavailable);

        async ValueTask<bool> RecordAsync(string type, string outcome, AuditSeverity severity)
        {
            try
            {
                await audit.LogAuditEventAsync(new AuditEvent(
                    Guid.CreateVersion7().ToString("N"), type, DateTime.UtcNow, modelId ?? string.Empty,
                    outcome, severity, new Dictionary<string, string>(metadata, StringComparer.Ordinal)));
                return true;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                logger.LogError(ex, "Inference audit could not be written for request {RequestId}", requestId);
                return false;
            }
        }
    }
}

internal sealed record InferenceOutcome<T>(T? Value = null, InferenceError? Error = null) where T : class;
internal sealed record InferenceError(int Status, string Code, string Message, int? RetryAfterSeconds = null);

internal static class InferenceErrors
{
    // Advise clients when to check again without estimating the GPU's cooling time.
    private const int ThermalRetryAfterSeconds = 5;

    internal static int? GetRetryAfterSeconds(string? code) =>
        code is "device_temperature_exceeded" or "device_thermal_migration_unavailable" ? ThermalRetryAfterSeconds : null;

    internal static readonly InferenceError InvalidRequest =
        new(400, "invalid_request", "A model ID and valid text messages are required.");
    internal static readonly InferenceError ContextLengthExceeded =
        new(400, "context_length_exceeded", "The prompt exceeds the configured local model context.");
    internal static readonly InferenceError UnsupportedLocalModel =
        new(422, "unsupported_model_format", "The local model format or encoding is unsupported.");
    internal static readonly InferenceError LocalMemoryExhausted =
        new(503, "local_memory_exhausted", "The local model could not allocate memory.");
    internal static readonly InferenceError InvalidEmbeddingRequest =
        new(400, "invalid_request", "Provide one to 2048 non-empty texts and use float encoding.");
    internal static readonly InferenceError ModelIdRequired =
        new(400, "invalid_request", "Supply exactly one model ID.");
    internal static readonly InferenceError ModelNotFound =
        new(404, "model_not_found", "The requested model is not registered.");
    internal static readonly InferenceError PolicyDenied =
        new(403, "policy_denied", "Inference is not allowed for the requested model.");
    internal static readonly InferenceError ProviderFailed =
        new(502, "provider_error", "The configured provider could not complete the request.");
    internal static readonly InferenceError AuditUnavailable =
        new(503, "audit_unavailable", "Inference audit storage is unavailable.");
    internal static readonly InferenceError EmbeddingsUnsupported =
        new(400, "unsupported_operation", "The requested model does not support embeddings.");
    internal static readonly InferenceError ChatUnsupported =
        new(400, "unsupported_operation", "The requested model does not support chat completions.");
}
