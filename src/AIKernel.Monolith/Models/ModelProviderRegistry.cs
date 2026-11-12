using System.Text.Json;
using System.Text.Json.Serialization;
using AIKernel.Dtos.Governance;
using AIKernel.Enums.Governance;
using AIKernel.Core.Governance;
using AIKernel.Core.Memory;
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
        List<(string Path, JsonException Error)> invalidJson = [];
        foreach (var path in Directory.EnumerateFiles(mountOptions.ModelRoot, "*.json", SearchOption.AllDirectories)
                     .Order(StringComparer.Ordinal))
        {
            using var document = ReadModelMountJson(path, invalidJson);
            if (document is null) continue;
            var root = document.RootElement;
            // Model mounts may also contain tokenizer and architecture metadata.
            IEnumerable<JsonElement> candidates = root.ValueKind == JsonValueKind.Array
                ? root.EnumerateArray()
                : [root];
            foreach (var candidate in candidates)
            {
                if (candidate.ValueKind != JsonValueKind.Object ||
                    !candidate.TryGetProperty("provider", out var provider) ||
                    provider.ValueKind != JsonValueKind.String ||
                    provider.GetString() is not (ChatOpenAIProviderId or PagedLlamaProviderId))
                {
                    continue;
                }

                var registration = candidate.Deserialize(InferenceJsonContext.Default.ModelRegistration)!;
                var model = registration.Provider == PagedLlamaProviderId
                    ? CreateLocalModel(registration, mountOptions, path)
                    : CreateModel(registration, path);
                if (!models.TryAdd(model.Metadata.Id, model))
                {
                    throw new InvalidDataException($"Duplicate model ID '{model.Metadata.Id}' in '{path}'.");
                }
            }
        }

        // Corrupt model assets fail at the provider boundary. Invalid registration files fail at startup.
        if (invalidJson.Count > 0)
        {
            var comparer = OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal;
            var assets = new HashSet<string>(comparer);
            foreach (var model in models.Values)
            {
                if (model.Local?.Tokenizer is { } tokenizer) assets.Add(tokenizer);
                if (model.Local?.Config is { } config) assets.Add(config);
            }

            foreach (var (path, error) in invalidJson)
            {
                if (!assets.Contains(Path.GetFullPath(path)))
                    throw new InvalidDataException($"Invalid model registration JSON in '{path}'.", error);
            }
        }
    }

    private static JsonDocument? ReadModelMountJson(string path, List<(string Path, JsonException Error)> invalidJson)
    {
        using var file = File.OpenRead(path);
        try
        {
            return JsonDocument.Parse(file);
        }
        catch (JsonException error)
        {
            invalidJson.Add((path, error));
            return null;
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
        var kvRam = config?.KvRamBlocks ?? 32;
        var kvVram = config?.KvVramBlocks ?? 16;
        var temperature = config?.Temperature ?? 0;
        var topP = config?.TopP ?? 1;
        var maxPcie = config?.MaxPcieSwapBytes ?? long.MaxValue;
        var maxTemperature = config?.MaxTemperatureCelsius ?? 0;
        var batchPower = config?.BatchPowerLimitPercent ?? 0;
        var priority = config?.Priority switch
        {
            null or "normal" => InferencePriority.Normal,
            "backgroundBatch" => InferencePriority.BackgroundBatch,
            "high" => InferencePriority.High,
            "emergency" => InferencePriority.Emergency,
            _ => (InferencePriority?)null
        };
        if (string.IsNullOrWhiteSpace(registration.Id) || registration.Kind is not (null or "chat") ||
            registration.Endpoint is not null || registration.ApiKeyEnvironmentVariable is not null ||
            config is null || blockSize is < 4096 or > 32 * 1024 * 1024 || (blockSize & (blockSize - 1)) != 0 ||
            ramBlocks <= 0 || vramBlocks <= 0 || device < 0 || context is < 8 or > 2048 || maxNew is < 1 or > 128 ||
            kvRam < 1 || kvVram < 0 || !float.IsFinite(temperature) || temperature < 0 ||
            !float.IsFinite(topP) || topP is <= 0 or > 1 || priority is null ||
            maxPcie < 0 || maxTemperature > 120 || batchPower > 100 ||
            config.ThermalMigrationDevices?.Any(static ordinal => ordinal < 0) == true)
        {
            throw new InvalidDataException($"Invalid paged Llama model registration in '{path}'.");
        }

        Dictionary<int, DeviceThermalLimits>? thermalLimits = null;
        if (config.ThermalDeviceLimits is { Length: > 0 } entries)
        {
            thermalLimits = new(entries.Length);
            foreach (var entry in entries)
            {
                if (entry is null || entry.Device is not int ordinal || ordinal < 0 ||
                    entry.MaxTemperatureCelsius is not uint running || running is 0 or > 120)
                    throw new InvalidDataException($"Invalid thermalDeviceLimits entry in '{path}'.");

                var target = entry.MaxMigrationTargetTemperatureCelsius ?? running;
                if (target == 0 || target > running || !thermalLimits.TryAdd(ordinal, new(running, target)))
                    throw new InvalidDataException($"Invalid or duplicate thermalDeviceLimits entry for device {ordinal} in '{path}'.");
            }
        }

        var local = new PagedLlamaSettings(
            ResolveAsset(mounts.ModelRoot, config.Checkpoint, path),
            config.Tokenizer is null ? null : ResolveAsset(mounts.ModelRoot, config.Tokenizer, path),
            config.Config is null ? null : ResolveAsset(mounts.ModelRoot, config.Config, path),
            Path.Combine(mounts.ConfigRoot, "model-swap"),
            blockSize, ramBlocks, vramBlocks, device, context, maxNew, kvRam, kvVram, temperature, topP,
            priority.Value, maxPcie, new(maxTemperature, batchPower), new(config.EncryptSwap ?? false, config.ZeroWipe ?? false),
            config.ThermalMigrationDevices, thermalLimits);
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
