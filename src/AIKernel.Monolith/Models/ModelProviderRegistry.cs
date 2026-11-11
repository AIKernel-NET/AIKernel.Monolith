using System.Text.Json;
using System.Text.Json.Serialization;
using AIKernel.Dtos.Governance;
using AIKernel.Enums.Governance;
using AIKernel.Monolith.Hosting;
using AIKernel.Monolith.Internal;
using AIKernel.Monolith.Models.Local;
using AIKernel.Providers.ChatOpenAI;

namespace AIKernel.Monolith.Models;

/// <summary>
/// Loads executable model registrations from the external model mount.
/// </summary>
public sealed class ModelProviderRegistry
{
    private const string ChatOpenAIProviderId = "providers.openai";
    private const string PagedLlamaProviderId = "providers.paged-llama";
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
                if (registration?.Provider is not (ChatOpenAIProviderId or PagedLlamaProviderId))
                {
                    continue;
                }

                var model = registration.Provider == PagedLlamaProviderId
                    ? CreateLocalModel(registration, mountOptions, path)
                    : CreateModel(registration, path);
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
            ReadPolicy(registration.Policy));
    }

    private static RegisteredModel CreateLocalModel(ModelRegistration registration, MountOptions mounts, string path)
    {
        var config = registration.Local;
        var blockSize = config?.BlockSize ?? 2 * 1024 * 1024;
        var ramBlocks = config?.RamBlocks ?? 32;
        var vramBlocks = config?.VramBlocks ?? 32;
        var context = config?.ContextTokens ?? 512;
        var maxNew = config?.MaxNewTokens ?? 32;
        var device = config?.Device ?? 0;
        if (string.IsNullOrWhiteSpace(registration.Id) || registration.Kind is not (null or "chat") ||
            registration.Endpoint is not null || registration.ApiKeyEnvironmentVariable is not null ||
            config is null || blockSize is < 4096 or > 32 * 1024 * 1024 || (blockSize & (blockSize - 1)) != 0 ||
            ramBlocks <= 0 || vramBlocks <= 0 || device < 0 || context is < 8 or > 2048 || maxNew is < 1 or > 128)
        {
            throw new InvalidDataException($"Invalid paged Llama model registration in '{path}'.");
        }

        var local = new PagedLlamaSettings(
            ResolveAsset(mounts.ModelRoot, config.Checkpoint, path),
            ResolveAsset(mounts.ModelRoot, config.Tokenizer, path),
            Path.Combine(mounts.ConfigRoot, "model-swap"),
            blockSize, ramBlocks, vramBlocks, device, context, maxNew);
        return new RegisteredModel(new ModelMetadata(registration.Id, PagedLlamaProviderId, "chat"),
            null, null, ReadPolicy(registration.Policy), local);
    }

    private static string ResolveAsset(string modelRoot, string? asset, string registrationPath)
    {
        if (string.IsNullOrWhiteSpace(asset) || Path.IsPathRooted(asset))
        {
            throw new InvalidDataException($"Expected a path relative to the model mount in '{registrationPath}'.");
        }

        var root = Path.GetFullPath(modelRoot);
        var fullPath = Path.GetFullPath(asset, root);
        var relative = Path.GetRelativePath(root, fullPath);
        if (relative is "." or ".." || relative.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal)
            || Path.IsPathRooted(relative))
        {
            throw new InvalidDataException($"A model path escapes the model mount in '{registrationPath}'.");
        }

        return fullPath;
    }

    private static GateInput ReadPolicy(ModelPolicy? policy) => new()
    {
        Logos = ParseVote(policy?.Logos),
        Ethos = ParseVote(policy?.Ethos),
        Pathos = ParseVote(policy?.Pathos)
    };

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
    ChatOpenAISettings? Settings,
    string? ApiKeyEnvironmentVariable,
    GateInput Policy,
    PagedLlamaSettings? Local = null);

internal sealed record ModelRegistration(
    [property: JsonPropertyName("id")] string? Id,
    [property: JsonPropertyName("provider")] string? Provider,
    [property: JsonPropertyName("kind")] string? Kind,
    [property: JsonPropertyName("endpoint")] string? Endpoint,
    [property: JsonPropertyName("model")] string? Model,
    [property: JsonPropertyName("apiKeyEnvironmentVariable")] string? ApiKeyEnvironmentVariable,
    [property: JsonPropertyName("policy")] ModelPolicy? Policy,
    [property: JsonPropertyName("local")] PagedLlamaRegistration? Local);

internal sealed record ModelPolicy(
    [property: JsonPropertyName("logos")] string? Logos,
    [property: JsonPropertyName("ethos")] string? Ethos,
    [property: JsonPropertyName("pathos")] string? Pathos);
