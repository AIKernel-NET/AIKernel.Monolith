using System.Text.Json.Serialization;
using AIKernel.Abstractions.Governance;
using AIKernel.Common.Results;
using AIKernel.Core.Governance;
using AIKernel.Kernel.Governance;
using AIKernel.Monolith.Models.Local;

namespace AIKernel.Monolith.Models;

internal sealed class ModelAdmission(IDecisionGate gate, PagedLlamaPool localModels, CudaTopology topology)
{
    internal async Task<ModelAdmissionStatus> ReadAsync(RegisteredModel model, string requestId, CancellationToken token)
    {
        var operation = model.Metadata.Kind == "embedding" ? "embeddings" : "chat.completions";
        var (policyAllowed, _) = await GovernedModelInference.EvaluatePolicyAsync(gate, model, operation, requestId, token);
        var inventory = model.Local is null ? null : topology.Snapshot;
        (string backend, int? device, int? admissionDevice, string? reason, GpuAdmissionStatus[] devices) = model.Local is { } settings
            ? ReadLocal(settings, localModels.GetPlacement(model.Metadata.Id), inventory!.PciBusIds, token)
            : model.Gguf is { } cpu ? ("cpu",null,null,File.Exists(cpu.Checkpoint)?null:"checkpoint_unavailable",Array.Empty<GpuAdmissionStatus>())
            : model.Cosmos is { } cosmos ? ("cpu",null,null,CosmosChatInput.HasCheckpoint(cosmos)?null:"checkpoint_unavailable",Array.Empty<GpuAdmissionStatus>())
            : ("external", null, null, CredentialError(model), Array.Empty<GpuAdmissionStatus>());
        if (!policyAllowed)
            reason = InferenceErrors.PolicyDenied.Code;

        var retryAfter = inventory is { PciBusIds: null } && reason == InferenceErrors.CudaDeviceUnavailable.Code
            ? CudaTopology.RetryDelaySeconds
            : InferenceErrors.GetRetryAfterSeconds(reason);
        var nextRetry = inventory?.RemainingRetryDelay;
        return new(model.Metadata.Id, reason is null, reason, backend, device,
            reason is null ? admissionDevice : null, retryAfter, devices, inventory?.State,
            nextRetry.HasValue ? (int)Math.Ceiling(nextRetry.Value.TotalSeconds) : null, inventory?.LastFailure);
    }

    private (string Backend, int? Device, int? AdmissionDevice, string? Reason, GpuAdmissionStatus[] Devices)
        ReadLocal(PagedLlamaSettings settings, PagedLlamaPlacement? placement, string?[]? visibleDevices, CancellationToken token)
    {
        var ordinal = placement is null ? settings.Device : placement.Device;
        if (visibleDevices is null)
        {
            GpuAdmissionStatus[] unavailable = ordinal is int configured
                ? [ReadDevice(configured, null, settings, null, null).Status]
                : [];
            return ("unknown", ordinal, null, InferenceErrors.CudaDeviceUnavailable.Code, unavailable);
        }

        using var telemetry = visibleDevices.Length == 0 ? null : NvmlDeviceTelemetry.Open().Value;
        var sampled = new Dictionary<int, DeviceAdmission>();
        for (var index = 0; index < visibleDevices.Length; index++)
        {
            token.ThrowIfCancellationRequested();
            sampled.Add(index, ReadDevice(index, visibleDevices[index], settings, telemetry, placement));
        }

        if (ordinal is int current && !sampled.ContainsKey(current) && placement is not null)
            sampled.Add(current, ReadDevice(current, null, settings, telemetry, placement));

        var devices = sampled.Values.Select(device => device.Status).ToArray();
        if (ordinal is null || !sampled.TryGetValue(ordinal.Value, out var source))
            return ("cpu", null, null, null, devices);
        if (!source.Available)
            return ("unknown", ordinal, null, InferenceErrors.CudaDeviceUnavailable.Code, devices);

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

    private static DeviceAdmission ReadDevice(int ordinal, string? pci, PagedLlamaSettings settings,
        NvmlDeviceTelemetry? telemetry, PagedLlamaPlacement? placement)
    {
        var active = placement?.Device == ordinal;
        if (active && placement!.PciBusId is { } knownPci)
            pci = knownPci;
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
            allowed && pci is not null && target.IsSuccess, health.Error?.Code);
        return new(active || pci is not null, execution.Error?.Code, status);
    }

    private static string? CredentialError(RegisteredModel model) =>
        model.ApiKeyEnvironmentVariable is { } variable && string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable(variable))
            ? "credential_unavailable" : null;

    private sealed record DeviceAdmission(bool Available, string? ExecutionError, GpuAdmissionStatus Status);
}

internal sealed record ModelAdmissionStatus(
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("admissible")] bool Admissible,
    [property: JsonPropertyName("reason")] string? Reason,
    [property: JsonPropertyName("backend")] string Backend,
    [property: JsonPropertyName("device")] int? Device,
    [property: JsonPropertyName("admission_device")] int? AdmissionDevice,
    [property: JsonPropertyName("retry_after_seconds")] int? RetryAfterSeconds,
    [property: JsonPropertyName("devices")] GpuAdmissionStatus[] Devices,
    [property: JsonPropertyName("cuda_topology_state")]
    [property: JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] string? CudaTopologyState,
    [property: JsonPropertyName("cuda_next_retry_in_seconds")]
    [property: JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] int? CudaNextRetryInSeconds,
    [property: JsonPropertyName("cuda_last_failure_reason")]
    [property: JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] string? CudaLastFailureReason);

internal sealed record GpuAdmissionStatus(
    [property: JsonPropertyName("device")] int Device,
    [property: JsonPropertyName("pci_bus_id")] string? PciBusId,
    [property: JsonPropertyName("temperature_celsius")] uint? TemperatureCelsius,
    [property: JsonPropertyName("max_temperature_celsius")] uint MaxTemperatureCelsius,
    [property: JsonPropertyName("max_migration_target_temperature_celsius")] uint MaxMigrationTargetTemperatureCelsius,
    [property: JsonPropertyName("migration_allowed")] bool MigrationAllowed,
    [property: JsonPropertyName("migration_target_eligible")] bool MigrationTargetEligible,
    [property: JsonPropertyName("telemetry_error")] string? TelemetryError);
