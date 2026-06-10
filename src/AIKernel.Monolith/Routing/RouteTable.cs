namespace AIKernel.Monolith.Routing;

/// <summary>
/// EN: In-memory deterministic route table. JA: インメモリの決定論的ルートテーブルです。
/// </summary>
public sealed class RouteTable
{
    private readonly SortedDictionary<string, string> routes = new(StringComparer.Ordinal);

    /// <summary>
    /// EN: Adds or replaces a route. JA: ルートを追加または置換します。
    /// </summary>
    public void Add(string key, string target) => routes[key] = target;

    /// <summary>
    /// EN: Resolves a route key. JA: ルートキーを解決します。
    /// </summary>
    public string? Resolve(string key) => routes.TryGetValue(key, out var target) ? target : null;

    /// <summary>
    /// EN: Returns a stable route snapshot. JA: 安定したルートスナップショットを返します。
    /// </summary>
    public IReadOnlyDictionary<string, string> Snapshot() => routes;
}
