using System.Text.Json;
using AIKernel.Kernel.Inference;
using AIKernel.Monolith.Hosting;
using AIKernel.Monolith.Models;
using Xunit;

namespace AIKernel.Monolith.Tests;

public sealed class CosmosRegistrationTests : IDisposable
{
    private readonly DirectoryInfo root = Directory.CreateTempSubdirectory("cosmos-registration-");

    private CosmosCpuSettings Read(string access, string? memory = null, bool profile = false)
    {
        var options = memory is null ? "" : ",\"memory\":" + memory;
        if (profile) options += ",\"profilePrefill\":true";
        File.WriteAllText(Path.Combine(root.FullName, "models.json"), $$"""
            {
              "id": "edge", "provider": "providers.cosmos-cpu", "model": "cosmos3-edge",
              "cosmos": { "directory": "edge", "manifestSha256": "{{new string('a',64)}}",
                          "weightAccess": "{{access}}" {{options}} }
            }
            """);
        return new ModelProviderRegistry(new MountOptions(root.FullName, root.FullName, root.FullName, root.FullName))
            .Find("edge")!.Cosmos!;
    }

    [Fact]
    public void PagedSettingsUseCoreDefaultsAndDoNotAllocateDuringRegistration()
    {
        Assert.False(Read("paged").ProfilePrefill);
        Assert.True(Read("paged", profile:true).ProfilePrefill);
        var expected = new CosmosMemoryOptions { SwapDirectory = Path.Combine(root.FullName, "model-swap") };
        Assert.Equal(expected, Read("paged").Memory);
        Assert.Equal(expected with { KvCacheBytes = 8388608 }, Read("paged", """{"kvCacheBytes":8388608}""").Memory);
        var memory = Read("paged", """{"maximumResidentBytes":7000000000,"weightCacheBytes":3221225472,"kvCacheBytes":128000000,"runtimeReserveBytes":900000000}""").Memory;
        Assert.Equal(expected with { MaximumResidentBytes = 7000000000, WeightCacheBytes = 3221225472,
            KvCacheBytes = 128000000, RuntimeReserveBytes = 900000000 }, memory);
        Assert.False(Directory.Exists(expected.SwapDirectory));
    }

    [Theory]
    [InlineData("""{"kvCacheBytes":0}""")]
    [InlineData("""{"weightCacheBytes":1024}""")]
    [InlineData("""{"maximumResidentBytes":300000000,"runtimeReserveBytes":280000000}""")]
    public void InvalidBudgetsFailAtRegistration(string memory) =>
        Assert.Throws<InvalidDataException>(() => Read("paged", memory));

    [Fact]
    public void MappedModeAndUnknownOptionsCannotSilentlyIgnoreOperatorSettings()
    {
        Assert.Null(Read("mapped").Memory);
        Assert.Throws<InvalidDataException>(() => Read("mapped", "{}"));
        Assert.Throws<JsonException>(() => Read("paged", """{"kvBytes":8388608}"""));
    }

    public void Dispose() => root.Delete(true);
}
