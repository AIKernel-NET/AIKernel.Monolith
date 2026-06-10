namespace AIKernel.Monolith.Models;

/// <summary>
/// EN: Deterministic null model provider used before real model drivers are mounted. JA: 実モデル driver がマウントされる前に使う決定論的 null model provider です。
/// </summary>
public sealed class NullModelProvider : IModelProvider
{
    /// <inheritdoc />
    public string Id => "null";

    /// <inheritdoc />
    public IReadOnlyList<ModelMetadata> ListModels() => [new("null-model", Id, "deterministic-null")];

    /// <inheritdoc />
    public ModelCompletion Complete(ModelCompletionRequest request)
        => new(Guid.CreateVersion7().ToString("N"), $"[null-model:{request.Model}] {request.Prompt}");
}
