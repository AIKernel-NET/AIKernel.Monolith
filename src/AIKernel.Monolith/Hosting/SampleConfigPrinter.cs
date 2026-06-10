namespace AIKernel.Monolith.Hosting;

/// <summary>
/// EN: Prints embedded sample configuration templates. JA: 同梱サンプル設定テンプレートを出力します。
/// </summary>
public static class SampleConfigPrinter
{
    private static readonly string[] SampleFiles =
    [
        "sample-config.yaml",
        "sample-routing.yaml",
        "sample-vfs.yaml",
        "sample-models.json",
        "sample-capability-manifest.json"
    ];

    /// <summary>
    /// EN: Prints all sample files from the local samples directory. JA: ローカル samples ディレクトリからすべてのサンプルを出力します。
    /// </summary>
    public static void PrintAll(TextWriter writer)
    {
        var sampleRoot = Path.Combine(AppContext.BaseDirectory, "samples");
        foreach (var file in SampleFiles)
        {
            writer.WriteLine($"--- {file} ---");
            var path = Path.Combine(sampleRoot, file);
            writer.WriteLine(File.Exists(path)
                ? File.ReadAllText(path)
                : $"Sample file not found in this build: {file}");
            writer.WriteLine();
        }
    }
}
