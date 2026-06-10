using AIKernel.Monolith.Hosting;
using AIKernel.Monolith.Internal;
using AIKernel.Monolith.OpenAI;

if (args.Any(static x => string.Equals(x, "--print-sample-config", StringComparison.OrdinalIgnoreCase)))
{
    SampleConfigPrinter.PrintAll(Console.Out);
    return 0;
}

var mountOptions = MountOptions.From(args, Environment.GetEnvironmentVariables());
var validation = mountOptions.Validate();
if (!validation.IsValid)
{
    Console.Error.WriteLine("AIKernel.Monolith failed to start because required external mount roots are missing.");
    foreach (var error in validation.Errors)
    {
        Console.Error.WriteLine($"- {error}");
    }

    Console.Error.WriteLine("Required: --config-root, --model-root, --vfs-root, --capability-root or matching AIKERNEL_*_ROOT variables.");
    return 2;
}

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddAIKernelMonolith(mountOptions);

var app = builder.Build();

app.MapGet("/", (SemanticRuntime runtime) => Results.Json(runtime.Describe(), MonolithJsonContext.Default.RuntimeDescription));
app.MapGet("/health", (SemanticRuntime runtime) => Results.Json(new HealthResponse("ok", runtime.Describe()), MonolithJsonContext.Default.HealthResponse));
app.MapOpenAICompatibleApi();

app.Run();
return 0;

/// <summary>
/// EN: Program marker used by integration tests. JA: 統合テストで参照する Program マーカーです。
/// </summary>
public partial class Program;
