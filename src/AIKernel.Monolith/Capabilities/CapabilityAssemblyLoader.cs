namespace AIKernel.Monolith.Capabilities;

/// <summary>
/// EN: Produces safe load records for capability artifacts. JA: Capability 成果物に対する安全なロード記録を生成します。
/// </summary>
public sealed class CapabilityAssemblyLoader : ICapabilityAssemblyLoader
{
    /// <inheritdoc />
    public CapabilityLoadRecord Load(CapabilityDescriptor descriptor)
    {
        if (!File.Exists(descriptor.Path))
        {
            return new CapabilityLoadRecord(descriptor, false, $"Capability artifact does not exist: {descriptor.Path}");
        }

        var extension = Path.GetExtension(descriptor.Path).ToLowerInvariant();
        var supported = extension is ".json" or ".dll" or ".so";
        return supported
            ? new CapabilityLoadRecord(descriptor, true, "loaded")
            : new CapabilityLoadRecord(descriptor, false, $"Unsupported capability artifact: {extension}");
    }
}

/// <summary>
/// EN: Result of a deterministic capability load attempt. JA: 決定論的 Capability ロード試行の結果です。
/// </summary>
public sealed record CapabilityLoadRecord(CapabilityDescriptor Descriptor, bool Loaded, string Message);
