using System.ComponentModel;
using System.Diagnostics;
using System.Text.Json;
using AIKernel.Kernel.Memory;
using AIKernel.Monolith.Internal;

namespace AIKernel.Monolith.Models;

internal sealed class CudaTopology(ILogger<CudaTopology> logger)
{
    internal const string ProbeArgument = "--probe-cuda-topology";

    // A null inventory means enumeration failed. An empty inventory means no CUDA devices.
    internal string?[]? PciBusIds { get; private set; }

    internal static string?[]? Probe()
    {
        var count = CudaVirtualMemoryDevice.GetDeviceCount();
        if (!count.IsSuccess)
        {
            var first = CudaVirtualMemoryDevice.GetDevicePciBusId(0);
            return first.Error?.Code == "cuda_device_not_found" ? [] : null;
        }

        var ids = new string?[count.Value];
        for (var ordinal = 0; ordinal < ids.Length; ordinal++)
            ids[ordinal] = CudaVirtualMemoryDevice.GetDevicePciBusId(ordinal).Value;
        return ids;
    }

    internal async Task InitializeAsync()
    {
        // A short-lived process releases the CUDA driver's per-process GPU memory before the host accepts requests.
        if (Environment.ProcessPath is not { } executable)
            return;

        var start = new ProcessStartInfo(executable)
        {
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true
        };
        if (string.Equals(Path.GetFileNameWithoutExtension(executable), "dotnet", StringComparison.OrdinalIgnoreCase))
            start.ArgumentList.Add(Environment.GetCommandLineArgs()[0]);
        start.ArgumentList.Add(ProbeArgument);

        try
        {
            using var process = Process.Start(start);
            if (process is null)
                return;

            using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(15));
            var output = process.StandardOutput.ReadToEndAsync(timeout.Token);
            try
            {
                await process.WaitForExitAsync(timeout.Token);
                var json = await output;
                if (process.ExitCode == 0)
                    PciBusIds = JsonSerializer.Deserialize(json, InferenceJsonContext.Default.CudaPciBusIds);
            }
            catch (OperationCanceledException)
            {
                if (!process.HasExited)
                {
                    try { process.Kill(); }
                    catch (InvalidOperationException) when (process.HasExited) { }
                    await process.WaitForExitAsync();
                }
                logger.LogWarning("CUDA device enumeration timed out.");
            }
        }
        catch (Exception error) when (error is Win32Exception or InvalidOperationException or IOException or JsonException)
        {
            logger.LogWarning(error, "CUDA device enumeration failed.");
        }
    }
}
