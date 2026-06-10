namespace AIKernel.Monolith.VFS;

/// <summary>
/// EN: Read-only virtual file system contract. JA: 読み取り専用の仮想ファイルシステム契約です。
/// </summary>
public interface IVirtualFileSystem
{
    /// <summary>
    /// EN: Checks whether a path exists. JA: パスが存在するか確認します。
    /// </summary>
    bool Exists(string path);

    /// <summary>
    /// EN: Reads text from a path. JA: パスからテキストを読み取ります。
    /// </summary>
    string ReadText(string path);

    /// <summary>
    /// EN: Lists entries under a path. JA: パス配下のエントリを列挙します。
    /// </summary>
    IReadOnlyList<string> List(string path);
}
