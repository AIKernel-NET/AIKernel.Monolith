namespace AIKernel.Monolith.Capabilities;

/// <summary>
/// EN: Loads managed or native capability artifacts without mutating user files. JA: ユーザーファイルを変更せず managed/native Capability 成果物を読み込みます。
/// </summary>
public interface ICapabilityAssemblyLoader
{
    /// <summary>
    /// EN: Creates a deterministic load record for a descriptor. JA: Descriptor に対する決定論的ロード記録を作成します。
    /// </summary>
    CapabilityLoadRecord Load(CapabilityDescriptor descriptor);
}
