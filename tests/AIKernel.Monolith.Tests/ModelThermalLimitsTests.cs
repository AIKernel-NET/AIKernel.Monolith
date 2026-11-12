using AIKernel.Common.Results;
using AIKernel.Core.Governance;
using AIKernel.Kernel.Governance;
using AIKernel.Monolith.Hosting;
using AIKernel.Monolith.Models;
using Xunit;

namespace AIKernel.Monolith.Tests;

public sealed class ModelThermalLimitsTests : IDisposable
{
    private readonly DirectoryInfo root = Directory.CreateTempSubdirectory("monolith-thermal-limits-");

    [Fact]
    public void ADeviceCanKeepRunningAtATemperatureThatDisqualifiesItAsAMigrationTarget()
    {
        var settings = ReadRegistry("""
            [
              {"device":0,"maxTemperatureCelsius":68,"maxMigrationTargetTemperatureCelsius":62},
              {"device":1,"maxTemperatureCelsius":77,"maxMigrationTargetTemperatureCelsius":75},
              {"device":2,"maxTemperatureCelsius":70}
            ]
            """).Find("thermal")!.Local!;
        var sample = new DeviceHealthSnapshot(65, null, null, 0, 1024, 1024, null, null, null, null);

        Assert.True(Evaluate(sample, settings.GetHealthLimits(0)).IsSuccess);
        Assert.Equal("device_temperature_exceeded",
            Evaluate(sample, settings.GetHealthLimits(0, migrationTarget: true)).Error?.Code);
        Assert.True(Evaluate(sample with { TemperatureCelsius = 62 },
            settings.GetHealthLimits(0, migrationTarget: true)).IsSuccess);
        Assert.Equal(new DeviceHealthLimits(77, 90), settings.GetHealthLimits(1));
        Assert.Equal(new DeviceHealthLimits(75, 90), settings.GetHealthLimits(1, migrationTarget: true));
        Assert.Equal(new DeviceHealthLimits(70, 90), settings.GetHealthLimits(2, migrationTarget: true));
        Assert.Equal(new DeviceHealthLimits(83, 90), settings.GetHealthLimits(3, migrationTarget: true));
    }

    [Theory]
    [InlineData("[null]")]
    [InlineData("""[{"maxTemperatureCelsius":68}]""")]
    [InlineData("""[{"device":-1,"maxTemperatureCelsius":68}]""")]
    [InlineData("""[{"device":0,"maxTemperatureCelsius":0}]""")]
    [InlineData("""[{"device":0,"maxTemperatureCelsius":121}]""")]
    [InlineData("""[{"device":0,"maxTemperatureCelsius":68,"maxMigrationTargetTemperatureCelsius":0}]""")]
    [InlineData("""[{"device":0,"maxTemperatureCelsius":68,"maxMigrationTargetTemperatureCelsius":69}]""")]
    [InlineData("""[{"device":0,"maxTemperatureCelsius":68},{"device":0,"maxTemperatureCelsius":77}]""")]
    public void InvalidDeviceThermalLimitsFailAtRegistration(string limits)
    {
        var error = Assert.Throws<InvalidDataException>(() => ReadRegistry(limits));
        Assert.Contains("thermalDeviceLimits", error.Message);
    }

    private ModelProviderRegistry ReadRegistry(string limits)
    {
        File.WriteAllText(Path.Combine(root.FullName, "models.json"), $$"""
            {
              "id": "thermal",
              "provider": "providers.paged-llama",
              "local": {
                "checkpoint": "model.gguf",
                "maxTemperatureCelsius": 83,
                "batchPowerLimitPercent": 90,
                "thermalDeviceLimits": {{limits}}
              }
            }
            """);
        return new ModelProviderRegistry(new MountOptions(root.FullName, root.FullName, root.FullName, root.FullName));
    }

    private static Result<DeviceHealthSnapshot> Evaluate(DeviceHealthSnapshot sample, DeviceHealthLimits limits)
        => DeviceHealthPolicy.Evaluate(Result<DeviceHealthSnapshot>.Success(sample), limits, InferencePriority.Normal);

    public void Dispose() => root.Delete(recursive: true);
}
