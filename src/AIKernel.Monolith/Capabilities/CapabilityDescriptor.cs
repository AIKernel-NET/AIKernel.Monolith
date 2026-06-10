namespace AIKernel.Monolith.Capabilities;

/// <summary>
/// EN: Public metadata for a loadable capability module. JA: 読み込み可能な Capability module の公開メタデータです。
/// </summary>
public sealed record CapabilityDescriptor(
    string Id,
    string Name,
    string Version,
    string Kind,
    string Path,
    IReadOnlyList<string> Operations);
