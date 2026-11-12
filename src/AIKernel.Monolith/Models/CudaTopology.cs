using System.ComponentModel;
using System.Diagnostics;
using System.Text;
using System.Text.Json;
using AIKernel.Kernel.Memory;
using AIKernel.Monolith.Internal;

namespace AIKernel.Monolith.Models;

internal sealed class CudaTopology(ILogger<CudaTopology> logger)
{
    internal const string ProbeArgument = "--probe-cuda-topology";
    internal const string SupervisedProbeArgument = "--supervised";
    private const byte StartProbe = 1;

    private static readonly TimeSpan RetryDelay = TimeSpan.FromSeconds(5);
    private string?[]? pciBusIds;

    // A null inventory means enumeration is unavailable. An empty inventory means no CUDA devices.
    internal string?[]? PciBusIds => Volatile.Read(ref pciBusIds);

    internal static bool WaitForParent()
    {
        var lifetime = Console.OpenStandardInput();
        if (lifetime.ReadByte() != StartProbe)
            return false;

        if (!OperatingSystem.IsWindows())
        {
            // The parent keeps the pipe open for the lifetime of this probe.
            new Thread(() =>
            {
                try { lifetime.ReadByte(); }
                catch (IOException) { }
                using var self = Process.GetCurrentProcess();
                self.Kill();
            }) { IsBackground = true }.Start();
        }

        return true;
    }

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

    internal async Task RetryUntilAvailableAsync(CancellationToken stoppingToken)
    {
        while (PciBusIds is null)
        {
            await Task.Delay(RetryDelay, stoppingToken);
            await InitializeAsync(stoppingToken);
            if (PciBusIds is not null)
                logger.LogInformation("CUDA device enumeration recovered.");
        }
    }

    internal async Task InitializeAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        // A short-lived process releases the CUDA driver's per-process GPU memory before the host accepts requests.
        if (Environment.ProcessPath is not { } executable)
            return;

        var start = new ProcessStartInfo(executable)
        {
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardInput = true,
            StandardInputEncoding = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false)
        };
        if (string.Equals(Path.GetFileNameWithoutExtension(executable), "dotnet", StringComparison.OrdinalIgnoreCase))
            start.ArgumentList.Add(Environment.GetCommandLineArgs()[0]);
        start.ArgumentList.Add(ProbeArgument);
        start.ArgumentList.Add(SupervisedProbeArgument);

        try
        {
            using var job = OperatingSystem.IsWindows() ? WindowsCudaProbeJob.Create() : null;
            using var process = Process.Start(start);
            if (process is null)
                return;

            try
            {
                if (job is not null)
                    WindowsCudaProbeJob.Assign(job, process);

                // The child can initialize CUDA once it is covered by the parent's lifetime.
                var lifetime = process.StandardInput.BaseStream;
                lifetime.WriteByte(StartProbe);
                lifetime.Flush();

                using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
                timeout.CancelAfter(TimeSpan.FromSeconds(15));
                var output = process.StandardOutput.ReadToEndAsync(timeout.Token);
                await Task.WhenAll(process.WaitForExitAsync(timeout.Token), output);
                logger.LogDebug("CUDA topology probe process {ProcessId} exited with code {ExitCode}.",
                    process.Id, process.ExitCode);
                if (process.ExitCode != 0)
                    return;

                var inventory = JsonSerializer.Deserialize(await output, InferenceJsonContext.Default.CudaPciBusIds);
                if (inventory is not null && inventory.All(static pci => !string.IsNullOrWhiteSpace(pci)))
                    Volatile.Write(ref pciBusIds, inventory);
            }
            finally
            {
                if (!process.HasExited)
                {
                    try { process.Kill(); }
                    catch (InvalidOperationException) when (process.HasExited) { }
                    await process.WaitForExitAsync();
                }
            }
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            logger.LogWarning("CUDA device enumeration timed out.");
        }
        catch (Exception error) when (error is Win32Exception or InvalidOperationException or IOException or JsonException)
        {
            logger.LogWarning(error, "CUDA device enumeration failed.");
        }
    }
}

internal sealed class CudaTopologyRetryService(CudaTopology topology, ModelProviderRegistry registry) : BackgroundService
{
    protected override Task ExecuteAsync(CancellationToken stoppingToken)
        => registry.HasLocalModels ? topology.RetryUntilAvailableAsync(stoppingToken) : Task.CompletedTask;
}
