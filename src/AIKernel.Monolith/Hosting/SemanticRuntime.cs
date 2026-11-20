using System.Reflection;
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
    private static readonly string HostVersion = typeof(SemanticRuntime).Assembly
        .GetCustomAttribute<AssemblyInformationalVersionAttribute>()!.InformationalVersion.Split('+', 2)[0];
    private readonly Lazy<IReadOnlyList<string>> capabilityIds;
    private readonly ModelProviderRegistry modelProviderRegistry;
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
        ArgumentNullException.ThrowIfNull(mountOptions);
        capabilityIds = new Lazy<IReadOnlyList<string>>(() => Array.AsReadOnly(capabilityLoader.Scan().Select(x => x.Id).ToArray()));
        this.modelProviderRegistry = modelProviderRegistry;
        ArgumentNullException.ThrowIfNull(mountTable);
        this.router = router;
    }

    /// <summary>
    /// EN: Returns a safe public runtime summary. JA: 安全な公開ランタイム概要を返します。
    /// </summary>
    public RuntimeDescription Describe()
        => new(
            "AIKernel.Monolith",
            HostVersion,
            "config",
            "models",
            "vfs",
            "capabilities",
            capabilityIds.Value,
            modelProviderRegistry.ListModels().Select(x => x.Id).ToArray(),
            new MountDescription("config", "models", "vfs", "capabilities"),
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
