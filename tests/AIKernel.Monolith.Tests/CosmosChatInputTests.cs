using System.Text.Json;
using AIKernel.Kernel.Inference;
using AIKernel.Monolith.Models;
using AIKernel.Monolith.OpenAI;
using Xunit;

namespace AIKernel.Monolith.Tests;

public sealed class CosmosChatInputTests
{
    [Theory]
    [InlineData("cosmos3-edge")]
    [InlineData("cosmos3-nano")]
    public void FollowUpRetainsRolesAndPreviousImageAndClearsPixels(string architecture)
    {
        var pixels = Enumerable.Repeat((byte)42, 32 * 32 * 3).ToArray();
        var parts = JsonSerializer.SerializeToElement(new object[] {
            new { type = "image_rgb24", width = 32, height = 32, data = Convert.ToBase64String(pixels) },
            new { type = "text", text = "Left?" }
        });
        using var input = new CosmosChatInput();
        Assert.True(input.Read([new("system", "Test"), new("user", "") { Parts = parts },
            new("assistant", "Red."), new("user", "And on the right?")], architecture));
        Assert.Equal("Test", input.System);
        var turn = Assert.Single(input.History);
        Assert.Equal("Red.", turn.AssistantReply);
        var rgb = Assert.IsType<CosmosContent.Image>(turn.UserContent[0]).Frame.Pixels;
        Assert.Equal(pixels, rgb.ToArray());
        Assert.Equal("And on the right?", Assert.IsType<CosmosContent.Text>(Assert.Single(input.User)).Value);
        input.Dispose();
        Assert.All(rgb.ToArray(), b => Assert.Equal((byte)0, b));
    }

    [Theory]
    [InlineData("""[{"type":"image_url","url":"https://example.test/picture"}]""")]
    [InlineData("""[{"type":"video_rgb24","frames":[]}]""")]
    [InlineData("""[{"type":"image_rgb24","width":33,"height":32,"data":""}]""")]
    public void UnsupportedMediaIsRejected(string json)
    {
        using var document = JsonDocument.Parse(json);
        using var input = new CosmosChatInput();
        Assert.False(input.Read([new("user", "") { Parts = document.RootElement }], "cosmos3-edge"));
    }
}
