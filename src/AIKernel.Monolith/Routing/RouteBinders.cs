namespace AIKernel.Monolith.Routing;

/// <summary>
/// EN: Binds generic input to a capability route. JA: 汎用入力を Capability route に束縛します。
/// </summary>
public sealed class CapabilityRouteBinder
{
    /// <summary>
    /// EN: Produces a deterministic capability decision. JA: 決定論的 Capability 判断を生成します。
    /// </summary>
    public RouteDecision Bind(string input) => new("capability", "null.capability", $"No explicit route matched input length {input.Length}.");
}

/// <summary>
/// EN: Binds model names to model routes. JA: モデル名を model route に束縛します。
/// </summary>
public sealed class ModelRouteBinder
{
    /// <summary>
    /// EN: Produces a deterministic model decision. JA: 決定論的 model 判断を生成します。
    /// </summary>
    public RouteDecision Bind(string model) => new("model", model, "Explicit model requested.");
}

/// <summary>
/// EN: Binds tool names to tool routes. JA: ツール名を tool route に束縛します。
/// </summary>
public sealed class ToolRouteBinder
{
    /// <summary>
    /// EN: Produces a deterministic tool decision. JA: 決定論的 tool 判断を生成します。
    /// </summary>
    public RouteDecision Bind(string tool) => new("tool", tool, "Explicit tool requested.");
}
