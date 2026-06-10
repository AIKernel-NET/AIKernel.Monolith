using System.Text.Json;

namespace AIKernel.Monolith.Capabilities;

/// <summary>
/// EN: JSON-aware capability metadata reader. JA: JSON に対応した Capability メタデータリーダーです。
/// </summary>
public sealed class CapabilityMetadataReader : ICapabilityMetadataReader
{
    /// <inheritdoc />
    public CapabilityDescriptor Read(string path)
    {
        if (Path.GetExtension(path).Equals(".json", StringComparison.OrdinalIgnoreCase))
        {
            using var document = JsonDocument.Parse(File.ReadAllText(path));
            var root = document.RootElement;
            var id = GetString(root, "id") ?? GetString(root, "capabilityId") ?? Path.GetFileNameWithoutExtension(path);
            var name = GetString(root, "name") ?? id;
            var version = GetString(root, "version") ?? "0.0.0";
            var kind = GetString(root, "kind") ?? "managed";
            var operations = root.TryGetProperty("operations", out var op) && op.ValueKind == JsonValueKind.Array
                ? op.EnumerateArray().Select(x => x.GetString()).Where(x => !string.IsNullOrWhiteSpace(x)).Select(x => x!).Order(StringComparer.Ordinal).ToArray()
                : Array.Empty<string>();

            return new CapabilityDescriptor(id, name, version, kind, path, operations);
        }

        var fileName = Path.GetFileNameWithoutExtension(path);
        var extension = Path.GetExtension(path).ToLowerInvariant();
        var inferredKind = extension is ".dll" ? "managed" : extension is ".so" ? "native-linux" : "native-windows";
        return new CapabilityDescriptor(fileName, fileName, "0.0.0", inferredKind, path, Array.Empty<string>());
    }

    private static string? GetString(JsonElement element, string name)
        => element.TryGetProperty(name, out var property) && property.ValueKind == JsonValueKind.String
            ? property.GetString()
            : null;
}
