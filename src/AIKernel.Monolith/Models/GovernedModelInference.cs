using System.Text.Json;
using AIKernel.Abstractions.Events;
using AIKernel.Abstractions.Governance;
using AIKernel.Dtos.Events;
using AIKernel.Dtos.Governance;
using AIKernel.Enums;
using AIKernel.Enums.Governance;
using AIKernel.Monolith.OpenAI;
using AIKernel.Providers.ChatOpenAI;

namespace AIKernel.Monolith.Models;

internal sealed class GovernedModelInference(
    ModelProviderRegistry registry,
    IDecisionGate gate,
    IAuditEventWriter audit,
    IHttpClientFactory clients,
    ILogger<GovernedModelInference> logger)
{
    internal const string HttpClientName = "monolith-inference";

    internal async Task<InferenceOutcome> CompleteAsync(
        string? modelId,
        IReadOnlyList<ChatMessage>? messages,
        string requestId,
        string operation,
        CancellationToken cancellationToken)
    {
        var model = modelId is null ? null : registry.Find(modelId);
        var metadata = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["request_id"] = requestId,
            ["operation"] = operation
        };

        if (model is null)
        {
            var error = string.IsNullOrWhiteSpace(modelId) ? InferenceErrors.InvalidRequest : InferenceErrors.ModelNotFound;
            return await RecordAsync("inference.rejected", error.Code, AuditSeverity.Warning)
                ? new(Error: error)
                : new(Error: InferenceErrors.AuditUnavailable);
        }

        metadata["provider_id"] = model.Metadata.Provider;
        if (messages is null || messages.Count == 0 ||
            messages.Any(message => message is not
                { Role: "system" or "developer" or "user" or "assistant", Content: not null }))
        {
            return await RecordAsync("inference.rejected", InferenceErrors.InvalidRequest.Code, AuditSeverity.Warning)
                ? new(Error: InferenceErrors.InvalidRequest)
                : new(Error: InferenceErrors.AuditUnavailable);
        }

        var decision = await gate.EvaluateAsync(new DecisionGateRequest
        {
            OperationId = operation,
            CorrelationId = requestId,
            GateInput = model.Policy
        }, cancellationToken);

        var allowed = decision is { Succeeded: true, Accepted: true, DecisionKind: GateDecisionKind.Allow };
        metadata["decision"] = decision.DecisionKind.ToString();
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

        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            var apiKey = model.ApiKeyEnvironmentVariable is { } variable
                ? Environment.GetEnvironmentVariable(variable)
                : null;
            if (model.ApiKeyEnvironmentVariable is not null && string.IsNullOrWhiteSpace(apiKey))
            {
                return await ProviderFailedAsync("credential_unavailable");
            }

            var settings = model.Settings with { ApiKey = apiKey };
            using var httpClient = clients.CreateClient(HttpClientName);
            var provider = new ChatOpenAIProvider(settings, new ChatOpenAIClient(httpClient, settings));
            await provider.InitializeAsync();
            var text = await provider.GenerateAsync(messages, cancellationToken);
            if (string.IsNullOrEmpty(text))
            {
                return await ProviderFailedAsync("empty_provider_response");
            }

            var completion = new ModelCompletion($"cmpl-{Guid.CreateVersion7():N}", text);
            metadata["completion_id"] = completion.Id;
            return await RecordAsync("inference.execution", "succeeded", AuditSeverity.Information)
                ? new(Completion: completion)
                : new(Error: InferenceErrors.AuditUnavailable);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            await RecordAsync("inference.execution", "canceled", AuditSeverity.Warning);
            throw;
        }
        catch (Exception ex) when (ex is HttpRequestException or JsonException or OperationCanceledException)
        {
            return await ProviderFailedAsync("provider_error");
        }

        async Task<InferenceOutcome> ProviderFailedAsync(string reason)
            => await RecordAsync("inference.execution", reason, AuditSeverity.Error)
                ? new(Error: InferenceErrors.ProviderFailed)
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

internal sealed record InferenceOutcome(ModelCompletion? Completion = null, InferenceError? Error = null);
internal sealed record InferenceError(int Status, string Code, string Message);

internal static class InferenceErrors
{
    internal static readonly InferenceError InvalidRequest =
        new(400, "invalid_request", "A model ID and valid text messages are required.");
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
}
