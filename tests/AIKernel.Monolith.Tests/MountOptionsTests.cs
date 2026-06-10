using AIKernel.Monolith.Hosting;
using Xunit;

namespace AIKernel.Monolith.Tests;

/// <summary>
/// EN: Tests for external mount configuration. JA: 外部マウント設定のテストです。
/// </summary>
public sealed class MountOptionsTests
{
    /// <summary>
    /// EN: Empty mount options fail closed. JA: 空のマウント設定は fail closed します。
    /// </summary>
    [Fact]
    public void ValidateFailsWhenRootsAreMissing()
    {
        var options = new MountOptions("", "", "", "");
        var result = options.Validate();
        Assert.False(result.IsValid);
        Assert.Equal(4, result.Errors.Count);
    }

    /// <summary>
    /// EN: Existing roots pass validation. JA: 存在するルートは検証に成功します。
    /// </summary>
    [Fact]
    public void ValidatePassesWhenRootsExist()
    {
        var root = Directory.CreateTempSubdirectory("aikernel-monolith-test");
        try
        {
            var config = Directory.CreateDirectory(Path.Combine(root.FullName, "config"));
            var models = Directory.CreateDirectory(Path.Combine(root.FullName, "models"));
            var vfs = Directory.CreateDirectory(Path.Combine(root.FullName, "vfs"));
            var capabilities = Directory.CreateDirectory(Path.Combine(root.FullName, "capabilities"));

            var options = new MountOptions(config.FullName, models.FullName, vfs.FullName, capabilities.FullName);
            Assert.True(options.Validate().IsValid);
        }
        finally
        {
            root.Delete(recursive: true);
        }
    }
}
