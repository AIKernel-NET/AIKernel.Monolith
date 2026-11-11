using System.Globalization;
using System.Text;
using System.Text.Json;
using AIKernel.Common.Results;
using AIKernel.Core.Governance;
using AIKernel.Kernel.Governance;
using AIKernel.Kernel.Memory;
using AIKernel.Monolith.Models.Local;
using AIKernel.Monolith.OpenAI;

if (args.Length != 4)
    throw new ArgumentException("Provide the checkpoint, tokenizer, source ordinal, and destination ordinal.");
var source = int.Parse(args[2], CultureInfo.InvariantCulture);
var destination = int.Parse(args[3], CultureInfo.InvariantCulture);
var directory = Directory.CreateTempSubdirectory("aikernel-thermal-");
try
{
    var sourcePci = ReadPci(source);
    var destinationPci = ReadPci(destination);
    Check(!string.Equals(sourcePci, destinationPci, StringComparison.OrdinalIgnoreCase), "Expected two distinct GPUs.");
    Check(CudaVirtualMemoryDevice.GetDeviceCount().Value >= 2, "CUDA enumeration did not find both GPUs.");
    using var nvml = NvmlDeviceTelemetry.Open().Value ?? throw new InvalidOperationException("NVML is required.");
    var initialSource = nvml.Read(sourcePci);
    var initialDestination = nvml.Read(destinationPci);
    Check(initialSource.IsSuccess && initialDestination.IsSuccess, "Both PCI identities must resolve in NVML.");

    var settings = new PagedLlamaSettings(Path.GetFullPath(args[0]), Path.GetFullPath(args[1]), null, directory.FullName,
        4096, 4, 2, source, 64, 8, 1, 1, 0, 1, InferencePriority.Normal, long.MaxValue,
        new DeviceHealthLimits(120, 0), new(Encrypt: true, ZeroWipe: true), [destination]);
    using var timeout = new CancellationTokenSource(TimeSpan.FromMinutes(3));
    ChatMessage[] prompt = [new("user", "Once upon a time")];

    var baseline = await Generate(new ThermalSample(nvml, sourcePci), heatAfterFirstPiece: false);
    Check(baseline.Text == ", there was a little girl", "Unexpected baseline output.");
    var migrating = new ThermalSample(nvml, sourcePci);
    var resumed = await Generate(migrating, heatAfterFirstPiece: true);
    Check(resumed.Error is null && resumed.Text == baseline.Text, "Generation changed after migration.");
    Check(resumed.Audit["gpu_migrations"] == "1" && resumed.Audit["gpu_migration_to_pci"] == destinationPci &&
        resumed.Audit["gpu_migration_from_pci"] == sourcePci &&
        resumed.Audit["cuda_device_ordinal"] == destination.ToString(CultureInfo.InvariantCulture),
        "The request did not continue on the second physical device.");
    Check(int.Parse(resumed.Audit["gpu_migration_after_generated_tokens"], CultureInfo.InvariantCulture) > 0,
        "Migration must happen after the first generated piece.");
    Check(resumed.Audit["local_prefill_tokens"] == baseline.Audit["local_prefill_tokens"] &&
        resumed.Audit["local_forward_layers"] == baseline.Audit["local_forward_layers"],
        "Prefill or a completed layer was repeated.");

    using var checkpoint = new BinaryReader(File.OpenRead(settings.Checkpoint));
    checkpoint.BaseStream.Position = 2 * sizeof(int);
    var layers = checkpoint.ReadInt32();
    var prefill = int.Parse(baseline.Audit["local_prefill_tokens"], CultureInfo.InvariantCulture);
    var midLayer = new ThermalSample(nvml, sourcePci) { HeatOnSourceRead = 2 + prefill * layers + 1 };
    var logits = await Generate(midLayer, heatAfterFirstPiece: false);
    Check(logits.Error is null && logits.Text == baseline.Text && logits.Audit["gpu_migration_layer"] == "0" &&
        logits.Audit["gpu_migration_position"] == prefill.ToString(CultureInfo.InvariantCulture) &&
        logits.Audit["local_forward_layers"] == baseline.Audit["local_forward_layers"],
        "The first decode token did not resume at the next layer.");
    var maximumLogitDifference = logits.Logits.Zip(baseline.Logits, (actual, expected) => Math.Abs(actual - expected)).Max();
    Check(float.IsFinite(maximumLogitDifference) && maximumLogitDifference <= 0.0002f, "Logits changed after migration.");

    var unavailable = await Generate(new ThermalSample(nvml, sourcePci) { HeatAllWhenArmed = true },
        heatAfterFirstPiece: true);
    Check(unavailable.Error == "device_thermal_migration_unavailable" && unavailable.Pieces == 1 &&
        unavailable.Audit.GetValueOrDefault("cuda_device_ordinal") == source.ToString(CultureInfo.InvariantCulture),
        "A request continued after every alternative became ineligible.");

    using var output = new Utf8JsonWriter(Console.OpenStandardOutput(), new() { Indented = true });
    output.WriteStartObject();
    output.WriteString("source_pci", sourcePci);
    output.WriteString("destination_pci", destinationPci);
    output.WriteNumber("source_nvml_temperature", initialSource.Value.TemperatureCelsius);
    output.WriteNumber("destination_nvml_temperature", initialDestination.Value.TemperatureCelsius);
    output.WriteString("thermal_trigger", "controlled over-limit sample over live NVML");
    output.WriteNumber("prefill_tokens", prefill);
    output.WriteString("migration_after_generated_tokens", resumed.Audit["gpu_migration_after_generated_tokens"]);
    output.WriteString("forward_layers", resumed.Audit["local_forward_layers"]);
    output.WriteNumber("maximum_logit_difference", maximumLogitDifference);
    output.WriteString("no_destination_error", unavailable.Error);
    output.WriteNumber("pieces_before_rejection", unavailable.Pieces);
    output.WriteEndObject();
    output.Flush();

    async Task<Run> Generate(ThermalSample telemetry, bool heatAfterFirstPiece)
    {
        var schedulers = new Dictionary<int, LayerQosScheduler> { [source] = new(), [destination] = new() };
        using var model = new PagedLlamaModel(settings, timeout.Token, telemetry);
        using var file = new BinaryReader(File.OpenRead(settings.Checkpoint));
        file.BaseStream.Position = 5 * sizeof(int);
        var values = new float[Math.Abs(file.ReadInt32())];
        var audit = new Dictionary<string, string>();
        var text = new StringBuilder();
        var pieces = 0;
        string? error = null;
        try
        {
            await model.GenerateAsync(prompt, piece =>
            {
                text.Append(piece);
                if (++pieces == 1 && heatAfterFirstPiece) telemetry.Armed = true;
                return Task.CompletedTask;
            }, audit, null, timeout.Token, new(null, null, values), ordinal => schedulers[ordinal]);
        }
        catch (InferenceGovernanceException ex)
        {
            error = ex.Error.Code;
        }
        return new Run(text.ToString(), audit, values, pieces, error);
    }
}
finally
{
    directory.Delete(recursive: true);
}

static string ReadPci(int ordinal)
{
    using var gpu = new CudaVirtualMemoryDevice(ordinal);
    return gpu.GetPciBusId() ?? throw new InvalidOperationException("CUDA PCI identity is unavailable.");
}

static void Check(bool condition, string message)
{
    if (!condition) throw new InvalidOperationException(message);
}

sealed record Run(string Text, Dictionary<string, string> Audit, float[] Logits, int Pieces, string? Error);

sealed class ThermalSample(IDeviceHealthTelemetry nvml, string source) : IDeviceHealthTelemetry
{
    private int sourceReads;
    internal bool Armed { get; set; }
    internal bool HeatAllWhenArmed { get; init; }
    internal int HeatOnSourceRead { get; init; } = int.MaxValue;

    public Result<DeviceHealthSnapshot> Read(string pciBusId)
    {
        var sample = nvml.Read(pciBusId);
        var isSource = string.Equals(pciBusId, source, StringComparison.OrdinalIgnoreCase);
        if (sample.IsSuccess && (isSource && (++sourceReads >= HeatOnSourceRead || Armed) ||
            Armed && HeatAllWhenArmed))
            return Result<DeviceHealthSnapshot>.Success(sample.Value with { TemperatureCelsius = 121 });
        return sample;
    }
}
