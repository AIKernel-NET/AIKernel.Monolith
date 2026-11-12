using System.Text.Json.Serialization;
using AIKernel.Abstractions.Governance;
using AIKernel.Common.Results;
using AIKernel.Core.Governance;
using AIKernel.Kernel.Governance;
using AIKernel.Kernel.Memory;
using AIKernel.Monolith.Models.Local;

namespace AIKernel.Monolith.Models;

internal sealed class ModelAdmission(IDecisionGate gate, PagedLlamaPool localModels)
{
    private readonly Lock topologyLock = new();
    private readonly Dictionary<int, string> pciByOrdinal = [];

    internal async Task<ModelAdmissionStatus> ReadAsync(RegisteredModel model, string requestId, CancellationToken token)
    {
        var operation = model.Metadata.Kind == "embedding" ? "embeddings" : "chat.completions";
        var (policyAllowed, _) = await GovernedModelInference.EvaluatePolicyAsync(gate, model, operation, requestId, token);
        (string backend, int? device, int? admissionDevice, string? reason, GpuAdmissionStatus[] devices) = model.Local is { } settings
            ? ReadLocal(settings, localModels.GetPlacement(model.Metadata.Id), token)
            : ("external", null, null, CredentialError(model), Array.Empty<GpuAdmissionStatus>());
        if (!policyAllowed)
            reason = InferenceErrors.PolicyDenied.Code;

        return new(model.Metadata.Id, reason is null, reason, backend, device,
            reason is null ? admissionDevice : null, InferenceErrors.GetRetryAfterSeconds(reason), devices);
    }

    private (string Backend, int? Device, int? AdmissionDevice, string? Reason, GpuAdmissionStatus[] Devices)
        ReadLocal(PagedLlamaSettings settings, PagedLlamaPlacement? placement, CancellationToken token)
    {
        using var telemetry = NvmlDeviceTelemetry.Open().Value;
        var count = CudaVirtualMemoryDevice.GetDeviceCount();
        var ordinal = placement is null ? settings.Device : placement.Device;
        var sampled = new Dictionary<int, DeviceAdmission>();
        for (var index = 0; index < count.Value; index++)
        {
            token.ThrowIfCancellationRequested();
            sampled.Add(index, ReadDevice(index, settings, telemetry, placement));
        }

        if (ordinal is int current && !sampled.ContainsKey(current) && (placement is not null || !count.IsSuccess))
            sampled.Add(current, ReadDevice(current, settings, telemetry, placement));

        var devices = sampled.Values.Select(device => device.Status).ToArray();
        if (ordinal is null || !sampled.TryGetValue(ordinal.Value, out var source) || source.Available == false)
            return ("cpu", null, null, null, devices);
        if (source.Available is null)
            return ("unknown", ordinal, null, "cuda_device_unavailable", devices);

        var limits = settings.GetHealthLimits(ordinal.Value);
        if (limits == default)
            return ("cuda", ordinal, ordinal, null, devices);
        if (placement is null && source.Status.TelemetryError is not null)
            return ("cpu", null, null, null, devices);

        if (source.ExecutionError == "device_temperature_exceeded" && placement?.RecoveryPending == true)
        {
            var candidates = (IEnumerable<int>?)settings.ThermalMigrationDevices ?? sampled.Keys;
            foreach (var candidate in candidates)
            {
                if (candidate != ordinal && sampled.TryGetValue(candidate, out var target) &&
                    target.Status.MigrationTargetEligible &&
                    !string.Equals(target.Status.PciBusId, source.Status.PciBusId, StringComparison.OrdinalIgnoreCase))
                    return ("cuda", ordinal, candidate, null, devices);
            }
            return ("cuda", ordinal, null, "device_thermal_migration_unavailable", devices);
        }

        return ("cuda", ordinal, ordinal, source.ExecutionError, devices);
    }

    private DeviceAdmission ReadDevice(int ordinal, PagedLlamaSettings settings, NvmlDeviceTelemetry? telemetry,
        PagedLlamaPlacement? placement)
    {
        var active = placement?.Device == ordinal;
        var (available, pci) = active && placement!.PciBusId is { } knownPci ? (true, knownPci) : ProbeDevice(ordinal);
        var limits = settings.GetHealthLimits(ordinal);
        var targetLimits = settings.GetHealthLimits(ordinal, migrationTarget: true);
        var allowed = settings.ThermalMigrationDevices?.Contains(ordinal) ?? true;
        var health = pci is not null && telemetry is not null ? telemetry.Read(pci)
            : Result<DeviceHealthSnapshot>.Fail(new ErrorContext("Required device telemetry is unavailable.",
                "device_telemetry_unavailable", true));
        var execution = DeviceHealthPolicy.Evaluate(health, limits, settings.Priority);
        var target = DeviceHealthPolicy.Evaluate(health, targetLimits, settings.Priority);
        var status = new GpuAdmissionStatus(ordinal, pci, health.IsSuccess ? health.Value.TemperatureCelsius : null,
            limits.MaximumTemperatureCelsius, targetLimits.MaximumTemperatureCelsius, allowed,
            allowed && target.IsSuccess, health.Error?.Code);
        return new(active ? true : available, execution.Error?.Code, status);
    }

    private (bool? Available, string? Pci) ProbeDevice(int ordinal)
    {
        lock (topologyLock)
        {
            if (pciByOrdinal.TryGetValue(ordinal, out var cached))
                return (true, cached);

            try
            {
                using var device = CudaVirtualMemoryDevice.TryCreate(ordinal, out _);
                if (device is null)
                    return (false, null);

                var pci = device.GetPciBusId();
                if (pci is not null)
                    pciByOrdinal.Add(ordinal, pci);
                return (true, pci);
            }
            catch (Exception error) when (error is DllNotFoundException or EntryPointNotFoundException or
                BadImageFormatException or InvalidOperationException or NotSupportedException or OutOfMemoryException)
            {
                return (null, null);
            }
        }
    }

    private static string? CredentialError(RegisteredModel model) =>
        model.ApiKeyEnvironmentVariable is { } variable && string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable(variable))
            ? "credential_unavailable" : null;

    private sealed record DeviceAdmission(bool? Available, string? ExecutionError, GpuAdmissionStatus Status);
}

internal sealed record ModelAdmissionStatus(
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("admissible")] bool Admissible,
    [property: JsonPropertyName("reason")] string? Reason,
    [property: JsonPropertyName("backend")] string Backend,
    [property: JsonPropertyName("device")] int? Device,
    [property: JsonPropertyName("admission_device")] int? AdmissionDevice,
    [property: JsonPropertyName("retry_after_seconds")] int? RetryAfterSeconds,
    [property: JsonPropertyName("devices")] GpuAdmissionStatus[] Devices);

internal sealed record GpuAdmissionStatus(
    [property: JsonPropertyName("device")] int Device,
    [property: JsonPropertyName("pci_bus_id")] string? PciBusId,
    [property: JsonPropertyName("temperature_celsius")] uint? TemperatureCelsius,
    [property: JsonPropertyName("max_temperature_celsius")] uint MaxTemperatureCelsius,
    [property: JsonPropertyName("max_migration_target_temperature_celsius")] uint MaxMigrationTargetTemperatureCelsius,
    [property: JsonPropertyName("migration_allowed")] bool MigrationAllowed,
    [property: JsonPropertyName("migration_target_eligible")] bool MigrationTargetEligible,
    [property: JsonPropertyName("telemetry_error")] string? TelemetryError);
