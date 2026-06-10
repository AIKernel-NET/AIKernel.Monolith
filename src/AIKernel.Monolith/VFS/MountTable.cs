using AIKernel.Monolith.Hosting;

namespace AIKernel.Monolith.VFS;

/// <summary>
/// EN: Describes configured external mount points. JA: 設定済み外部マウントポイントを記述します。
/// </summary>
public sealed class MountTable
{
    private readonly MountOptions mountOptions;

    /// <summary>
    /// EN: Creates a mount table. JA: mount table を作成します。
    /// </summary>
    public MountTable(MountOptions mountOptions) => this.mountOptions = mountOptions;

    /// <summary>
    /// EN: Returns a safe mount summary. JA: 安全な mount 概要を返します。
    /// </summary>
    public MountDescription Describe() => new(
        mountOptions.ConfigRoot,
        mountOptions.ModelRoot,
        mountOptions.VfsRoot,
        mountOptions.CapabilityRoot);
}

/// <summary>
/// EN: Public mount description. JA: 公開マウント記述です。
/// </summary>
public sealed record MountDescription(string Config, string Models, string Vfs, string Capabilities);
