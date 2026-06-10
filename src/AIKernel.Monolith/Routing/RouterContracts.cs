namespace AIKernel.Monolith.Routing;

/// <summary>
/// EN: Deterministic router abstraction. JA: 決定論的ルーター抽象です。
/// </summary>
public interface IRouter
{
    /// <summary>
    /// EN: Resolves an input into a route decision. JA: 入力を route decision に解決します。
    /// </summary>
    RouteDecision Resolve(RouteRequest request);
}

/// <summary>
/// EN: Input for route resolution. JA: ルート解決への入力です。
/// </summary>
public sealed record RouteRequest(string Input, string? Model = null, string? Tool = null);

/// <summary>
/// EN: Deterministic route decision. JA: 決定論的なルート判断です。
/// </summary>
public sealed record RouteDecision(string TargetKind, string TargetId, string Reason);
