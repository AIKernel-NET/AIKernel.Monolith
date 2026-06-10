using AIKernel.Monolith.Capabilities;
using AIKernel.Monolith.Hosting;
using Xunit;

namespace AIKernel.Monolith.Tests;

/// <summary>
/// EN: Tests for deterministic capability loading. JA: 決定論的 Capability loading のテストです。
/// </summary>
public sealed class CapabilityLoaderTests
{
    /// <summary>
    /// EN: Capability manifests are returned in deterministic order. JA: Capability manifest は決定論的順序で返されます。
    /// </summary>
    [Fact]
    public void ScanReturnsCapabilitiesInOrdinalOrder()
    {
        var root = Directory.CreateTempSubdirectory("aikernel-monolith-capability-test");
        try
        {
            var config = Directory.CreateDirectory(Path.Combine(root.FullName, "config"));
            var models = Directory.CreateDirectory(Path.Combine(root.FullName, "models"));
            var vfs = Directory.CreateDirectory(Path.Combine(root.FullName, "vfs"));
            var capabilities = Directory.CreateDirectory(Path.Combine(root.FullName, "capabilities"));
            File.WriteAllText(Path.Combine(capabilities.FullName, "b.json"), """{"id":"b.capability","name":"B","version":"0.1.1","operations":["b"]}""");
            File.WriteAllText(Path.Combine(capabilities.FullName, "a.json"), """{"id":"a.capability","name":"A","version":"0.1.1","operations":["a"]}""");

            var options = new MountOptions(config.FullName, models.FullName, vfs.FullName, capabilities.FullName);
            var loader = new CapabilityLoader(options, new CapabilityMetadataReader(), new CapabilityAssemblyLoader(), new CapabilityInvokerBinder());

            var ids = loader.Scan().Select(x => x.Id).ToArray();

            Assert.Equal(["a.capability", "b.capability"], ids);
        }
        finally
        {
            root.Delete(recursive: true);
        }
    }
}
