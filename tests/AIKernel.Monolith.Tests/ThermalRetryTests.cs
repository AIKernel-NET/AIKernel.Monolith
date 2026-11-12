using System.Text;
using System.Text.Json;
using AIKernel.Monolith.Models;
using AIKernel.Monolith.OpenAI;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Xunit;

namespace AIKernel.Monolith.Tests;

public sealed class ThermalRetryTests
{
    [Fact]
    public async Task MigrationFailureAfterAChunkIncludesRetryAdviceInTheSseError()
    {
        using var body = new MemoryStream();
        using var services = new ServiceCollection().AddLogging().BuildServiceProvider();
        var context = new DefaultHttpContext { RequestServices = services };
        context.Response.Body = body;
        var stream = new OpenAIStreamingResponse(context, "local", chat: false, CancellationToken.None);

        await stream.WriteChunkAsync("partial output");
        await stream.WriteErrorAsync(new InferenceError(StatusCodes.Status503ServiceUnavailable,
            "device_thermal_migration_unavailable", "No eligible GPU", RetryAfterSeconds: 5));

        var events = Encoding.UTF8.GetString(body.ToArray()).Split("\n\n", StringSplitOptions.RemoveEmptyEntries);
        Assert.Equal("text/event-stream", context.Response.ContentType);
        Assert.False(context.Response.Headers.ContainsKey("Retry-After"));
        Assert.Collection(events, chunk =>
        {
            Assert.StartsWith("data: ", chunk);
            using var document = JsonDocument.Parse(chunk[6..]);
            Assert.Equal("partial output", document.RootElement.GetProperty("choices")[0].GetProperty("text").GetString());
        }, failure =>
        {
            Assert.StartsWith("data: ", failure);
            using var document = JsonDocument.Parse(failure[6..]);
            var error = document.RootElement.GetProperty("error");
            Assert.Equal("device_thermal_migration_unavailable", error.GetProperty("code").GetString());
            Assert.Equal(5, error.GetProperty("retry_after_seconds").GetInt32());
        });
    }
}
