using System.Net;
using System.Text;
using System.Text.Json;
using AIKernel.Monolith.Hosting;
using AIKernel.Monolith.OpenAI;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.DependencyInjection;
using Xunit;

namespace AIKernel.Monolith.Tests;

public sealed class GovernedInferenceTests
{
    [Fact]
    public async Task RuntimeReportsTheBuiltHostVersion()
    {
        await using var host = new InferenceHost();
        var reported = host.Describe().Version.Split('-')[0];
        Assert.Equal(typeof(SemanticRuntime).Assembly.GetName().Version!.ToString(3), reported);
    }

    [Theory]
    [InlineData("/v1/chat/completions")]
    [InlineData("/v1/completions")]
    public async Task InferenceRoutesToTheRequestedProviderAfterRecordingThePolicyDecision(string route)
    {
        await using var host = new InferenceHost();
        var (status, body) = await host.SendAsync(route, "second");

        Assert.Equal(StatusCodes.Status200OK, status);
        Assert.Contains("upstream answer", body);
        var call = Assert.Single(host.Upstream.Calls);
        Assert.Equal("https://second.example/v1/chat/completions", call.Uri);
        Assert.Equal("upstream-second", call.Model);
        Assert.Equal("test prompt", call.Input);
        Assert.Contains("\"Description\":\"allowed\"", call.AuditBeforeSend);

        var audit = host.ReadAudit();
        Assert.Collection(audit,
            entry => AssertAudit(entry, "inference.policy", "allowed", "second"),
            entry => AssertAudit(entry, "inference.execution", "succeeded", "second"));
        using var response = JsonDocument.Parse(body);
        Assert.Equal(response.RootElement.GetProperty("id").GetString(),
            audit[1].GetProperty("Metadata").GetProperty("completion_id").GetString());
        Assert.DoesNotContain("test prompt", File.ReadAllText(host.AuditPath));
        Assert.DoesNotContain("upstream answer", File.ReadAllText(host.AuditPath));
    }

    [Theory]
    [InlineData("/v1/chat/completions")]
    [InlineData("/v1/completions")]
    public async Task UnregisteredModelIsRejectedAndAudited(string route)
    {
        await using var host = new InferenceHost();
        var (status, body) = await host.SendAsync(route, "missing");

        Assert.Equal(StatusCodes.Status404NotFound, status);
        Assert.Contains("model_not_found", body);
        Assert.Empty(host.Upstream.Calls);
        AssertAudit(Assert.Single(host.ReadAudit()), "inference.rejected", "model_not_found", "missing");
    }

    [Theory]
    [InlineData("vetoed")]
    [InlineData("no-policy")]
    [InlineData("cpu-no-policy")]
    [InlineData("cosmos-no-policy")]
    public async Task CorePolicyDenialPreventsTheProviderCall(string model)
    {
        await using var host = new InferenceHost();
        var (status, body) = await host.SendAsync("/v1/chat/completions", model);

        Assert.Equal(StatusCodes.Status403Forbidden, status);
        Assert.Contains("policy_denied", body);
        Assert.Empty(host.Upstream.Calls);
        var audit = Assert.Single(host.ReadAudit());
        AssertAudit(audit, "inference.policy", "denied", model);
        Assert.Equal("Deny", audit.GetProperty("Metadata").GetProperty("decision").GetString());
    }

    [Theory]
    [InlineData("gpt-missing")]
    [InlineData("cosmos-missing")]
    public async Task MissingCheckpointReturnsServiceUnavailableAndRecordsTheFailure(string model)
    {
        await using var host = new InferenceHost();
        var (status, body) = await host.SendAsync("/v1/chat/completions", model);
        Assert.Equal(StatusCodes.Status503ServiceUnavailable, status);
        Assert.Contains("checkpoint_unavailable", body);
        Assert.Collection(host.ReadAudit(),
            entry => AssertAudit(entry, "inference.policy", "allowed", model),
            entry => AssertAudit(entry, "inference.execution", "checkpoint_unavailable", model));
        Assert.Empty(host.Upstream.Calls);
    }

    [Fact]
    public async Task AuditWriteFailurePreventsTheProviderCall()
    {
        await using var host = new InferenceHost();
        Directory.CreateDirectory(host.AuditPath);

        var (status, body) = await host.SendAsync("/v1/chat/completions", "second");

        Assert.Equal(StatusCodes.Status503ServiceUnavailable, status);
        Assert.Contains("audit_unavailable", body);
        Assert.Empty(host.Upstream.Calls);
    }

    private static void AssertAudit(JsonElement entry, string type, string outcome, string model)
    {
        Assert.Equal(type, entry.GetProperty("EventType").GetString());
        Assert.Equal(outcome, entry.GetProperty("Description").GetString());
        Assert.Equal(model, entry.GetProperty("Subject").GetString());
        Assert.Equal("test-request", entry.GetProperty("Metadata").GetProperty("request_id").GetString());
    }

    private sealed class InferenceHost : IAsyncDisposable
    {
        private readonly DirectoryInfo root = Directory.CreateTempSubdirectory("monolith-inference-");
        private readonly WebApplication app;

        public InferenceHost()
        {
            var config = root.CreateSubdirectory("config").FullName;
            var models = root.CreateSubdirectory("models").FullName;
            AuditPath = Path.Combine(config, "inference-audit.jsonl");
            File.WriteAllText(Path.Combine(models, "models.json"), """
                [
                  { "id": "gpt-missing", "provider":"providers.gguf-cpu", "model":"gpt-oss-20b",
                    "policy":{"logos":"approve","ethos":"approve","pathos":"approve"},
                    "gguf":{"checkpoint":"not-installed.gguf"} },
                  { "id": "cosmos-missing", "provider":"providers.cosmos-cpu", "model":"cosmos3-edge",
                    "policy":{"logos":"approve","ethos":"approve","pathos":"approve"},
                    "cosmos":{"directory":"not-installed","manifestSha256":"0000000000000000000000000000000000000000000000000000000000000000"} },
                  { "id": "cosmos-no-policy", "provider":"providers.cosmos-cpu", "model":"cosmos3-edge",
                    "cosmos":{"directory":"edge","manifestSha256":"0000000000000000000000000000000000000000000000000000000000000000"} },
                  { "id": "cpu-no-policy", "provider":"providers.gguf-cpu", "model":"qwen3",
                    "gguf":{"checkpoint":"not-installed.gguf"} },
                  {
                    "id": "first", "provider": "providers.openai",
                    "endpoint": "https://first.example/v1/",
                    "policy": { "logos": "approve", "ethos": "approve", "pathos": "abstain" }
                  },
                  {
                    "id": "second", "provider": "providers.openai", "model": "upstream-second",
                    "endpoint": "https://second.example/v1",
                    "policy": { "logos": "approve", "ethos": "approve", "pathos": "abstain" }
                  },
                  {
                    "id": "vetoed", "provider": "providers.openai",
                    "endpoint": "https://second.example/v1/",
                    "policy": { "logos": "approve", "ethos": "reject", "pathos": "approve" }
                  },
                  {
                    "id": "no-policy", "provider": "providers.openai",
                    "endpoint": "https://second.example/v1/"
                  }
                ]
                """);

            Upstream = new FakeUpstream(AuditPath);
            var builder = WebApplication.CreateBuilder([]);
            builder.Services.AddAIKernelMonolith(new MountOptions(config, models,
                root.CreateSubdirectory("vfs").FullName, root.CreateSubdirectory("capabilities").FullName));
            builder.Services.AddHttpClient("monolith-inference")
                .ConfigurePrimaryHttpMessageHandler(() => Upstream);
            app = builder.Build();
            app.MapOpenAICompatibleApi();
        }

        public string AuditPath { get; }
        public FakeUpstream Upstream { get; }
        public RuntimeDescription Describe() => app.Services.GetRequiredService<SemanticRuntime>().Describe();

        public async Task<(int Status, string Body)> SendAsync(string route, string model)
        {
            var endpoint = ((IEndpointRouteBuilder)app).DataSources.SelectMany(x => x.Endpoints)
                .OfType<RouteEndpoint>()
                .Single(x => x.RoutePattern.RawText == route);

            using var scope = app.Services.CreateScope();
            var context = new DefaultHttpContext { RequestServices = scope.ServiceProvider, TraceIdentifier = "test-request" };
            context.Features.Set<IHttpRequestBodyDetectionFeature>(new RequestBodyFeature());
            var json = route == "/v1/chat/completions"
                ? $$"""{"model":"{{model}}","messages":[{"role":"user","content":"test prompt"}]}"""
                : $$"""{"model":"{{model}}","prompt":"test prompt"}""";
            using var request = new MemoryStream(Encoding.UTF8.GetBytes(json));
            using var response = new MemoryStream();
            context.Request.Method = HttpMethods.Post;
            context.Request.Path = route;
            context.Request.ContentType = "application/json";
            context.Request.ContentLength = request.Length;
            context.Request.Body = request;
            context.Response.Body = response;

            await endpoint.RequestDelegate!(context);
            return (context.Response.StatusCode, Encoding.UTF8.GetString(response.ToArray()));
        }

        public JsonElement[] ReadAudit() => File.ReadAllLines(AuditPath).Select(line =>
        {
            using var document = JsonDocument.Parse(line);
            return document.RootElement.Clone();
        }).ToArray();

        public async ValueTask DisposeAsync()
        {
            await app.DisposeAsync();
            root.Delete(recursive: true);
        }
    }

    private sealed class RequestBodyFeature : IHttpRequestBodyDetectionFeature
    {
        public bool CanHaveBody => true;
    }

    private sealed class FakeUpstream(string auditPath) : HttpMessageHandler
    {
        public List<(string Uri, string? Model, string? Input, string AuditBeforeSend)> Calls { get; } = [];

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var audit = File.ReadAllText(auditPath);
            using var body = JsonDocument.Parse(await request.Content!.ReadAsStringAsync(cancellationToken));
            Calls.Add((request.RequestUri!.AbsoluteUri,
                body.RootElement.GetProperty("model").GetString(),
                body.RootElement.GetProperty("messages")[0].GetProperty("content").GetString(),
                audit));

            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(
                    """{"choices":[{"message":{"role":"assistant","content":"upstream answer"}}]}""",
                    Encoding.UTF8, "application/json")
            };
        }
    }
}
