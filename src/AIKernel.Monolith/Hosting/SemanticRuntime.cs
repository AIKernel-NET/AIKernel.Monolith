using AIKernel.Monolith.Capabilities;
using AIKernel.Monolith.Models;
using AIKernel.Monolith.Routing;
using AIKernel.Monolith.VFS;

namespace AIKernel.Monolith.Hosting;

/// <summary>
/// EN: Coordinates the deterministic Semantic OS runtime. JA: 決定論的 Semantic OS ランタイムを統括します。
/// </summary>
public sealed class SemanticRuntime
{
    private readonly MountOptions mountOptions;
    private readonly CapabilityLoader capabilityLoader;
    private readonly ModelProviderRegistry modelProviderRegistry;
    private readonly MountTable mountTable;
    private readonly Routing.IRouter router;

    /// <summary>
    /// EN: Creates a runtime coordinator. JA: ランタイム調整器を作成します。
    /// </summary>
    public SemanticRuntime(
        MountOptions mountOptions,
        CapabilityLoader capabilityLoader,
        ModelProviderRegistry modelProviderRegistry,
        MountTable mountTable,
        Routing.IRouter router)
    {
        this.mountOptions = mountOptions;
        this.capabilityLoader = capabilityLoader;
        this.modelProviderRegistry = modelProviderRegistry;
        this.mountTable = mountTable;
        this.router = router;
    }

    /// <summary>
    /// EN: Returns a safe public runtime summary. JA: 安全な公開ランタイム概要を返します。
    /// </summary>
    public RuntimeDescription Describe()
        => new(
            "AIKernel.Monolith",
            "0.1.1",
            mountOptions.ConfigRoot,
            mountOptions.ModelRoot,
            mountOptions.VfsRoot,
            mountOptions.CapabilityRoot,
            capabilityLoader.Scan().Select(x => x.Id).ToArray(),
            modelProviderRegistry.ListModels().Select(x => x.Id).ToArray(),
            mountTable.Describe(),
            router.GetType().Name);
}

/// <summary>
/// EN: Public runtime description DTO. JA: 公開ランタイム記述 DTO です。
/// </summary>
public sealed record RuntimeDescription(
    string Name,
    string Version,
    string ConfigRoot,
    string ModelRoot,
    string VfsRoot,
    string CapabilityRoot,
    IReadOnlyList<string> Capabilities,
    IReadOnlyList<string> Models,
    VFS.MountDescription Mounts,
    string Router);

/// <summary>
/// EN: Public health response DTO. JA: 公開 health response DTO です。
/// </summary>
public sealed record HealthResponse(string Status, RuntimeDescription Runtime);
