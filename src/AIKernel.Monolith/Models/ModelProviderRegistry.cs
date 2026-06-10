using AIKernel.Monolith.Hosting;
using System.Text.Json;

namespace AIKernel.Monolith.Models;

/// <summary>
/// EN: Registry for model providers and external model metadata. JA: model provider と外部 model metadata の registry です。
/// </summary>
public sealed class ModelProviderRegistry
{
    private readonly MountOptions mountOptions;
    private readonly IEnumerable<IModelProvider> providers;

    /// <summary>
    /// EN: Creates a model registry. JA: model registry を作成します。
    /// </summary>
    public ModelProviderRegistry(MountOptions mountOptions, IEnumerable<IModelProvider> providers)
    {
        this.mountOptions = mountOptions;
        this.providers = providers;
    }

    /// <summary>
    /// EN: Lists mounted models in deterministic order. JA: マウント済みモデルを決定論的順序で列挙します。
    /// </summary>
    public IReadOnlyList<ModelMetadata> ListModels()
    {
        var fromProviders = providers.SelectMany(x => x.ListModels());
        var fromFiles = Directory.EnumerateFiles(mountOptions.ModelRoot, "*.json", SearchOption.AllDirectories)
            .Order(StringComparer.Ordinal)
            .SelectMany(ReadModels);
        return fromProviders.Concat(fromFiles)
            .OrderBy(x => x.Id, StringComparer.Ordinal)
            .ThenBy(x => x.Provider, StringComparer.Ordinal)
            .ToArray();
    }

    /// <summary>
    /// EN: Completes text using the first deterministic provider. JA: 最初の決定論的 provider でテキストを completion します。
    /// </summary>
    public ModelCompletion Complete(ModelCompletionRequest request)
        => providers.OrderBy(x => x.Id, StringComparer.Ordinal).First().Complete(request);

    private static IEnumerable<ModelMetadata> ReadModels(string path)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(path));
        var root = document.RootElement;
        if (root.ValueKind == JsonValueKind.Array)
        {
            foreach (var item in root.EnumerateArray())
            {
                yield return ReadModel(item);
            }
        }
        else
        {
            yield return ReadModel(root);
        }
    }

    private static ModelMetadata ReadModel(JsonElement item)
    {
        var id = item.TryGetProperty("id", out var idValue) ? idValue.GetString() ?? "unknown" : "unknown";
        var provider = item.TryGetProperty("provider", out var providerValue) ? providerValue.GetString() ?? "external" : "external";
        var kind = item.TryGetProperty("kind", out var kindValue) ? kindValue.GetString() ?? "chat" : "chat";
        return new ModelMetadata(id, provider, kind);
    }
}
