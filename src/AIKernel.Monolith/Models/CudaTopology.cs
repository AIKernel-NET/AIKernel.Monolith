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

    internal const int RetryDelaySeconds = 5;
    private CudaTopologySnapshot snapshot = new(null);

    internal CudaTopologySnapshot Snapshot => Volatile.Read(ref snapshot);

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
        while (Snapshot is { PciBusIds: null } waiting)
        {
            await Task.Delay(waiting.RemainingRetryDelay ?? TimeSpan.Zero, stoppingToken);
            await InitializeAsync(stoppingToken);
            if (Snapshot.PciBusIds is not null)
                logger.LogInformation("CUDA device enumeration recovered.");
        }
    }

    internal async Task InitializeAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var previousFailure = Snapshot.LastFailure;
        string?[]? pciBusIds = null;
        var failure = "probe_start_failed";
        Volatile.Write(ref snapshot, new(null, IsProbing: true, LastFailure: previousFailure));
        try
        {
            // The probe process releases the driver's per-process GPU memory before the inventory is published.
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

            using var job = OperatingSystem.IsWindows() ? WindowsCudaProbeJob.Create() : null;
            using var process = Process.Start(start);
            if (process is null)
                return;

            try
            {
                if (job is not null)
                    WindowsCudaProbeJob.Assign(job, process);

                failure = "probe_failed";
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
                {
                    failure = "probe_nonzero_exit";
                    return;
                }

                failure = "probe_invalid_inventory";
                var inventory = JsonSerializer.Deserialize(await output, InferenceJsonContext.Default.CudaPciBusIds);
                if (inventory is not null && inventory.All(static pci => !string.IsNullOrWhiteSpace(pci)))
                    pciBusIds = inventory;
            }
            finally
            {
                if (!process.HasExited)
                {
                    try { process.Kill(); }
                    catch (InvalidOperationException) when (process.HasExited) { }
                }

                // Windows can expose an exit code while termination is still in progress.
                // Wait on the process handle before accepting requests or publishing the inventory.
                if (OperatingSystem.IsWindows())
                    process.WaitForExit();
                else
                    await process.WaitForExitAsync();
            }
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            failure = "probe_timeout";
            logger.LogWarning("CUDA device enumeration timed out.");
        }
        catch (Exception error) when (error is Win32Exception or InvalidOperationException or IOException or JsonException)
        {
            logger.LogWarning(error, "CUDA device enumeration failed.");
        }
        finally
        {
            CudaTopologySnapshot completed = pciBusIds is not null ? new(pciBusIds)
                : cancellationToken.IsCancellationRequested ? new(null, LastFailure: previousFailure)
                : new(null, LastFailure: failure, RetryWaitStarted: Stopwatch.GetTimestamp());
            Volatile.Write(ref snapshot, completed);
        }
    }
}

internal sealed record CudaTopologySnapshot(string?[]? PciBusIds, bool IsProbing = false,
    string? LastFailure = null, long? RetryWaitStarted = null)
{
    private static readonly TimeSpan RetryDelay = TimeSpan.FromSeconds(CudaTopology.RetryDelaySeconds);

    internal TimeSpan? RemainingRetryDelay
    {
        get
        {
            if (RetryWaitStarted is not long started)
                return null;
            var remaining = RetryDelay - Stopwatch.GetElapsedTime(started);
            return remaining > TimeSpan.Zero ? remaining : TimeSpan.Zero;
        }
    }

    internal string State => PciBusIds switch
    {
        { Length: 0 } => "no_devices",
        not null => "ready",
        _ => IsProbing ? "probing" : "retry_wait"
    };
}

internal sealed class CudaTopologyRetryService(CudaTopology topology, ModelProviderRegistry registry) : BackgroundService
{
    protected override Task ExecuteAsync(CancellationToken stoppingToken)
        => registry.HasPagedLlamaModels ? topology.RetryUntilAvailableAsync(stoppingToken) : Task.CompletedTask;
}
