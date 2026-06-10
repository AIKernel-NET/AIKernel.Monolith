namespace AIKernel.Monolith.Capabilities;

/// <summary>
/// EN: Reads deterministic capability metadata from external files. JA: 外部ファイルから決定論的な Capability メタデータを読み取ります。
/// </summary>
public interface ICapabilityMetadataReader
{
    /// <summary>
    /// EN: Reads metadata for a capability file or manifest. JA: Capability ファイルまたは manifest のメタデータを読み取ります。
    /// </summary>
    CapabilityDescriptor Read(string path);
}
