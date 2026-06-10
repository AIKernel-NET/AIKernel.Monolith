using System.Collections;

namespace AIKernel.Monolith.Hosting;

/// <summary>
/// EN: External mount roots required by the monolith. JA: Monolith が必要とする外部マウントルートです。
/// </summary>
public sealed record MountOptions(
    string ConfigRoot,
    string ModelRoot,
    string VfsRoot,
    string CapabilityRoot)
{
    /// <summary>
    /// EN: Creates mount options from command-line arguments and environment variables. JA: コマンドライン引数と環境変数からマウント設定を作成します。
    /// </summary>
    public static MountOptions From(IReadOnlyList<string> args, IDictionary environment)
    {
        static string? ArgValue(IReadOnlyList<string> args, string name)
        {
            var prefix = name + "=";
            return args.FirstOrDefault(x => x.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)) is { } value
                ? value[prefix.Length..].Trim()
                : null;
        }

        static string EnvValue(IDictionary environment, string name)
            => environment[name]?.ToString()?.Trim() ?? string.Empty;

        return new MountOptions(
            ArgValue(args, "--config-root") ?? EnvValue(environment, "AIKERNEL_CONFIG_ROOT"),
            ArgValue(args, "--model-root") ?? EnvValue(environment, "AIKERNEL_MODEL_ROOT"),
            ArgValue(args, "--vfs-root") ?? EnvValue(environment, "AIKERNEL_VFS_ROOT"),
            ArgValue(args, "--capability-root") ?? EnvValue(environment, "AIKERNEL_CAPABILITY_ROOT"));
    }

    /// <summary>
    /// EN: Validates that all external roots exist. JA: すべての外部ルートが存在することを検証します。
    /// </summary>
    public MountValidationResult Validate()
    {
        var errors = new List<string>();
        ValidateDirectory(ConfigRoot, "config-root", errors);
        ValidateDirectory(ModelRoot, "model-root", errors);
        ValidateDirectory(VfsRoot, "vfs-root", errors);
        ValidateDirectory(CapabilityRoot, "capability-root", errors);
        return new MountValidationResult(errors.Count == 0, errors);
    }

    private static void ValidateDirectory(string value, string label, List<string> errors)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            errors.Add($"{label} is not configured.");
            return;
        }

        if (!Directory.Exists(value))
        {
            errors.Add($"{label} does not exist: {value}");
        }
    }
}

/// <summary>
/// EN: Validation result for external mount roots. JA: 外部マウントルートの検証結果です。
/// </summary>
public sealed record MountValidationResult(bool IsValid, IReadOnlyList<string> Errors);
