namespace AIKernel.Monolith.Models;

/// <summary>
/// EN: Semantic model provider contract. JA: Semantic model provider の契約です。
/// </summary>
public interface IModelProvider
{
    /// <summary>
    /// EN: Provider identifier. JA: Provider 識別子です。
    /// </summary>
    string Id { get; }

    /// <summary>
    /// EN: Lists models exposed by the provider. JA: Provider が公開するモデルを列挙します。
    /// </summary>
    IReadOnlyList<ModelMetadata> ListModels();

    /// <summary>
    /// EN: Produces a deterministic completion. JA: 決定論的な completion を生成します。
    /// </summary>
    ModelCompletion Complete(ModelCompletionRequest request);
}

/// <summary>
/// EN: Low-level model driver contract. JA: 低レベル model driver 契約です。
/// </summary>
public interface IModelDriver : IModelProvider;

/// <summary>
/// EN: Public model metadata. JA: 公開モデルメタデータです。
/// </summary>
public sealed record ModelMetadata(string Id, string Provider, string Kind);

/// <summary>
/// EN: Completion request. JA: Completion 要求です。
/// </summary>
public sealed record ModelCompletionRequest(string Model, string Prompt);

/// <summary>
/// EN: Completion response. JA: Completion 応答です。
/// </summary>
public sealed record ModelCompletion(string Id, string Text);
