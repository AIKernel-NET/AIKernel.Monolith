namespace AIKernel.Monolith.VFS;

/// <summary>
/// EN: In-memory read-only VFS used for deterministic tests and ephemeral mounts. JA: 決定論的テストと一時マウント向けのインメモリ読み取り専用 VFS です。
/// </summary>
public sealed class MemoryVirtualFileSystem : IVirtualFileSystem
{
    private readonly SortedDictionary<string, string> files = new(StringComparer.Ordinal);

    /// <summary>
    /// EN: Adds an ephemeral file to the memory VFS. JA: memory VFS に一時ファイルを追加します。
    /// </summary>
    public void Add(string path, string content) => files[Normalize(path)] = content;

    /// <inheritdoc />
    public bool Exists(string path) => files.ContainsKey(Normalize(path));

    /// <inheritdoc />
    public string ReadText(string path) => files[Normalize(path)];

    /// <inheritdoc />
    public IReadOnlyList<string> List(string path)
    {
        var prefix = Normalize(path).TrimEnd('/') + "/";
        return files.Keys.Where(x => x.StartsWith(prefix, StringComparison.Ordinal)).Order(StringComparer.Ordinal).ToArray();
    }

    private static string Normalize(string path) => "/" + path.Trim('/', '\\').Replace('\\', '/');
}
