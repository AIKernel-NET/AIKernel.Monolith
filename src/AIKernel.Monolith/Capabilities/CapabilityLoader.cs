using AIKernel.Monolith.Hosting;

namespace AIKernel.Monolith.Capabilities;

/// <summary>
/// EN: Scans external capability roots in deterministic order. JA: 外部 Capability ルートを決定論的順序でスキャンします。
/// </summary>
public sealed class CapabilityLoader
{
    private readonly MountOptions mountOptions;
    private readonly ICapabilityMetadataReader metadataReader;
    private readonly ICapabilityAssemblyLoader assemblyLoader;
    private readonly ICapabilityInvokerBinder invokerBinder;

    /// <summary>
    /// EN: Creates a capability loader. JA: Capability loader を作成します。
    /// </summary>
    public CapabilityLoader(
        MountOptions mountOptions,
        ICapabilityMetadataReader metadataReader,
        ICapabilityAssemblyLoader assemblyLoader,
        ICapabilityInvokerBinder invokerBinder)
    {
        this.mountOptions = mountOptions;
        this.metadataReader = metadataReader;
        this.assemblyLoader = assemblyLoader;
        this.invokerBinder = invokerBinder;
    }

    /// <summary>
    /// EN: Scans capability descriptors without modifying external files. JA: 外部ファイルを変更せず Capability descriptor をスキャンします。
    /// </summary>
    public IReadOnlyList<CapabilityDescriptor> Scan()
        => Directory.EnumerateFiles(mountOptions.CapabilityRoot, "*.*", SearchOption.AllDirectories)
            .Where(IsSupported)
            .Order(StringComparer.Ordinal)
            .Select(metadataReader.Read)
            .OrderBy(x => x.Id, StringComparer.Ordinal)
            .ThenBy(x => x.Version, StringComparer.Ordinal)
            .ToArray();

    /// <summary>
    /// EN: Loads and binds all discovered capabilities. JA: 検出したすべての Capability をロードして束縛します。
    /// </summary>
    public IReadOnlyList<CapabilityInvokerHandle> LoadAll()
        => Scan().Select(assemblyLoader.Load).Select(invokerBinder.Bind).ToArray();

    private static bool IsSupported(string path)
    {
        var extension = Path.GetExtension(path).ToLowerInvariant();
        return extension is ".json" or ".dll" or ".so";
    }
}
