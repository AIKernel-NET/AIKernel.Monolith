using AIKernel.Monolith.Hosting;

namespace AIKernel.Monolith.VFS;

/// <summary>
/// EN: Read-only local file system provider rooted at the external VFS mount. JA: 外部 VFS マウントをルートとする読み取り専用ローカル FS provider です。
/// </summary>
public sealed class LocalVirtualFileSystem : IVirtualFileSystem
{
    private readonly MountOptions mountOptions;

    /// <summary>
    /// EN: Creates a local VFS provider. JA: ローカル VFS provider を作成します。
    /// </summary>
    public LocalVirtualFileSystem(MountOptions mountOptions) => this.mountOptions = mountOptions;

    /// <inheritdoc />
    public bool Exists(string path) => File.Exists(Resolve(path)) || Directory.Exists(Resolve(path));

    /// <inheritdoc />
    public string ReadText(string path) => File.ReadAllText(Resolve(path));

    /// <inheritdoc />
    public IReadOnlyList<string> List(string path)
    {
        var fullPath = Resolve(path);
        return Directory.Exists(fullPath)
            ? Directory.EnumerateFileSystemEntries(fullPath).Order(StringComparer.Ordinal).Select(Path.GetFileName).Where(x => x is not null).Select(x => x!).ToArray()
            : Array.Empty<string>();
    }

    private string Resolve(string path)
    {
        var relative = path.TrimStart('/', '\\');
        var full = Path.GetFullPath(Path.Combine(mountOptions.VfsRoot, relative));
        var root = Path.GetFullPath(mountOptions.VfsRoot);
        if (!full.StartsWith(root, StringComparison.Ordinal))
        {
            throw new InvalidOperationException("VFS path escapes the configured root.");
        }

        return full;
    }
}
