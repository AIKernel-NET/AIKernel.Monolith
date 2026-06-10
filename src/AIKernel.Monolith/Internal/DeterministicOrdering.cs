namespace AIKernel.Monolith.Internal;

/// <summary>
/// EN: Shared deterministic ordering helpers. JA: 共有の決定論的順序付け helper です。
/// </summary>
public static class DeterministicOrdering
{
    /// <summary>
    /// EN: Orders strings with ordinal comparison. JA: 文字列を ordinal 比較で順序付けます。
    /// </summary>
    public static IEnumerable<string> Ordinal(IEnumerable<string> values) => values.Order(StringComparer.Ordinal);
}
