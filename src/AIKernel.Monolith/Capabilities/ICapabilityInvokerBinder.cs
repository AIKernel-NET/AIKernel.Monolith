namespace AIKernel.Monolith.Capabilities;

/// <summary>
/// EN: Binds loaded capabilities to invokable runtime stubs. JA: ロード済み Capability を呼び出し可能なランタイムスタブへ束縛します。
/// </summary>
public interface ICapabilityInvokerBinder
{
    /// <summary>
    /// EN: Binds a load record to an invoker handle. JA: ロード記録を invoker handle に束縛します。
    /// </summary>
    CapabilityInvokerHandle Bind(CapabilityLoadRecord record);
}
