namespace AIKernel.Monolith.Routing;

/// <summary>
/// EN: Deterministic router for model, tool, and capability requests. JA: model/tool/capability 要求向けの決定論的ルーターです。
/// </summary>
public sealed class DeterministicRouter : IRouter
{
    private readonly SemanticRouteResolver semanticRouteResolver;

    /// <summary>
    /// EN: Creates a deterministic router. JA: 決定論的ルーターを作成します。
    /// </summary>
    public DeterministicRouter(SemanticRouteResolver semanticRouteResolver) => this.semanticRouteResolver = semanticRouteResolver;

    /// <inheritdoc />
    public RouteDecision Resolve(RouteRequest request) => semanticRouteResolver.Resolve(request);
}
