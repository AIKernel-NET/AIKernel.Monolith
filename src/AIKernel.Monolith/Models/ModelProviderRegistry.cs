using System.Text.Json;
using System.Text.Json.Serialization;
using AIKernel.Dtos.Governance;
using AIKernel.Enums.Governance;
using AIKernel.Monolith.Hosting;
using AIKernel.Monolith.Internal;
using AIKernel.Providers.ChatOpenAI;

namespace AIKernel.Monolith.Models;

/// <summary>
/// Loads executable model registrations from the external model mount.
/// </summary>
public sealed class ModelProviderRegistry
{
    private const string ChatOpenAIProviderId = "providers.openai";
    private readonly Dictionary<string, RegisteredModel> models = new(StringComparer.Ordinal);

    public ModelProviderRegistry(MountOptions mountOptions)
    {
        foreach (var path in Directory.EnumerateFiles(mountOptions.ModelRoot, "*.json", SearchOption.AllDirectories)
                     .Order(StringComparer.Ordinal))
        {
            using var document = JsonDocument.Parse(File.ReadAllText(path));
            var root = document.RootElement;
            var registrations = root.ValueKind == JsonValueKind.Array
                ? root.Deserialize(InferenceJsonContext.Default.ModelRegistrationArray)
                : [root.Deserialize(InferenceJsonContext.Default.ModelRegistration)];

            foreach (var registration in registrations ?? [])
            {
                if (registration?.Provider != ChatOpenAIProviderId)
                {
                    continue;
                }

                var model = CreateModel(registration, path);
                if (!models.TryAdd(model.Metadata.Id, model))
                {
                    throw new InvalidDataException($"Duplicate model ID '{model.Metadata.Id}' in '{path}'.");
                }
            }
        }
    }

    /// <summary>Lists registered models in ordinal ID order.</summary>
    public IReadOnlyList<ModelMetadata> ListModels()
        => models.Values.Select(x => x.Metadata).OrderBy(x => x.Id, StringComparer.Ordinal).ToArray();

    internal RegisteredModel? Find(string id) => models.GetValueOrDefault(id);

    private static RegisteredModel CreateModel(ModelRegistration registration, string path)
    {
        if (string.IsNullOrWhiteSpace(registration.Id) ||
            registration.Kind is not (null or "chat" or "embedding") ||
            !Uri.TryCreate(registration.Endpoint, UriKind.Absolute, out var endpoint) ||
            endpoint.Scheme is not ("http" or "https") ||
            endpoint.UserInfo.Length != 0 || endpoint.Query.Length != 0 || endpoint.Fragment.Length != 0 ||
            (registration.ApiKeyEnvironmentVariable is not null &&
                (string.IsNullOrWhiteSpace(registration.ApiKeyEnvironmentVariable) || endpoint.Scheme != "https")) ||
            (registration.Model is not null && string.IsNullOrWhiteSpace(registration.Model)))
        {
            throw new InvalidDataException($"Invalid ChatOpenAI model registration in '{path}'.");
        }

        var kind = registration.Kind ?? "chat";
        var upstreamModel = registration.Model ?? registration.Id;
        var settings = new ChatOpenAISettings
        {
            Endpoint = new UriBuilder(endpoint) { Path = endpoint.AbsolutePath.TrimEnd('/') + "/" }.Uri,
            Model = upstreamModel,
            EmbeddingModel = kind == "embedding" ? upstreamModel : null
        };

        return new RegisteredModel(
            new ModelMetadata(registration.Id, ChatOpenAIProviderId, kind),
            settings,
            registration.ApiKeyEnvironmentVariable,
            new GateInput
            {
                Logos = ParseVote(registration.Policy?.Logos),
                Ethos = ParseVote(registration.Policy?.Ethos),
                Pathos = ParseVote(registration.Policy?.Pathos)
            });
    }

    private static CouncilVoteValue ParseVote(string? vote) => vote switch
    {
        "approve" => CouncilVoteValue.Approve,
        "abstain" => CouncilVoteValue.Abstain,
        "reject" => CouncilVoteValue.Reject,
        _ => CouncilVoteValue.Unknown
    };
}

internal sealed record RegisteredModel(
    ModelMetadata Metadata,
    ChatOpenAISettings Settings,
    string? ApiKeyEnvironmentVariable,
    GateInput Policy);

internal sealed record ModelRegistration(
    [property: JsonPropertyName("id")] string? Id,
    [property: JsonPropertyName("provider")] string? Provider,
    [property: JsonPropertyName("kind")] string? Kind,
    [property: JsonPropertyName("endpoint")] string? Endpoint,
    [property: JsonPropertyName("model")] string? Model,
    [property: JsonPropertyName("apiKeyEnvironmentVariable")] string? ApiKeyEnvironmentVariable,
    [property: JsonPropertyName("policy")] ModelPolicy? Policy);

internal sealed record ModelPolicy(
    [property: JsonPropertyName("logos")] string? Logos,
    [property: JsonPropertyName("ethos")] string? Ethos,
    [property: JsonPropertyName("pathos")] string? Pathos);
