namespace AIKernel.Monolith.Capabilities;

/// <summary>
/// EN: Creates deterministic invoker handles for loaded capabilities. JA: ロード済み Capability の決定論的 invoker handle を作成します。
/// </summary>
public sealed class CapabilityInvokerBinder : ICapabilityInvokerBinder
{
    /// <inheritdoc />
    public CapabilityInvokerHandle Bind(CapabilityLoadRecord record)
        => new(record.Descriptor.Id, record.Loaded, record.Message, record.Descriptor.Operations);
}

/// <summary>
/// EN: Runtime handle for an invokable capability. JA: 呼び出し可能な Capability のランタイム handle です。
/// </summary>
public sealed record CapabilityInvokerHandle(string CapabilityId, bool Available, string Message, IReadOnlyList<string> Operations);
