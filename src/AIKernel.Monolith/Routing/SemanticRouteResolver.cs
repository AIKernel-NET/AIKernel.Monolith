namespace AIKernel.Monolith.Routing;

/// <summary>
/// EN: Resolves semantic routing using deterministic binders. JA: 決定論的 binder を使って semantic routing を解決します。
/// </summary>
public sealed class SemanticRouteResolver
{
    private readonly CapabilityRouteBinder capabilityRouteBinder;
    private readonly ModelRouteBinder modelRouteBinder;
    private readonly ToolRouteBinder toolRouteBinder;

    /// <summary>
    /// EN: Creates a route resolver. JA: ルート resolver を作成します。
    /// </summary>
    public SemanticRouteResolver(CapabilityRouteBinder capabilityRouteBinder, ModelRouteBinder modelRouteBinder, ToolRouteBinder toolRouteBinder)
    {
        this.capabilityRouteBinder = capabilityRouteBinder;
        this.modelRouteBinder = modelRouteBinder;
        this.toolRouteBinder = toolRouteBinder;
    }

    /// <summary>
    /// EN: Resolves request routing in model, tool, then capability order. JA: model、tool、capability の順にリクエストルーティングを解決します。
    /// </summary>
    public RouteDecision Resolve(RouteRequest request)
    {
        if (!string.IsNullOrWhiteSpace(request.Model))
        {
            return modelRouteBinder.Bind(request.Model);
        }

        if (!string.IsNullOrWhiteSpace(request.Tool))
        {
            return toolRouteBinder.Bind(request.Tool);
        }

        return capabilityRouteBinder.Bind(request.Input);
    }
}
