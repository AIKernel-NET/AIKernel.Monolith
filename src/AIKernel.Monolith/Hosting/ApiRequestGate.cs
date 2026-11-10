using System.Globalization;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text;
using System.Threading.RateLimiting;
using AIKernel.Monolith.Internal;
using AIKernel.Monolith.OpenAI;

namespace AIKernel.Monolith.Hosting;

internal sealed class ApiRequestGate(byte[] apiKeyHash, FixedWindowRateLimiter limiter) : IDisposable
{
    private const string ApiKeyVariable = "AIKERNEL_API_KEY";
    private const string RequestLimitVariable = "AIKERNEL_API_REQUEST_LIMIT";
    private const string WindowSecondsVariable = "AIKERNEL_API_WINDOW_SECONDS";

    internal static ApiRequestGate? FromEnvironment(Func<string, string?> readVariable, out string? error)
    {
        var apiKey = readVariable(ApiKeyVariable);
        if (!IsBearerToken(apiKey))
        {
            error = $"Set {ApiKeyVariable} to a non-empty bearer token.";
            return null;
        }

        if (!ReadPositiveInt(RequestLimitVariable, out var permitLimit))
        {
            error = $"{RequestLimitVariable} must be a positive integer.";
            return null;
        }

        if (!ReadPositiveInt(WindowSecondsVariable, out var windowSeconds))
        {
            error = $"{WindowSecondsVariable} must be a positive integer.";
            return null;
        }

        error = null;
        return new ApiRequestGate(HashToken(apiKey!), new FixedWindowRateLimiter(new FixedWindowRateLimiterOptions
        {
            PermitLimit = permitLimit,
            Window = TimeSpan.FromSeconds(windowSeconds),
            AutoReplenishment = false,
            QueueLimit = 0
        }));

        bool ReadPositiveInt(string name, out int value)
            => int.TryParse(readVariable(name) ?? "60", NumberStyles.None, CultureInfo.InvariantCulture, out value)
               && value > 0;
    }

    internal async Task InvokeAsync(HttpContext context, RequestDelegate next)
    {
        if (!IsAuthenticated(context.Request))
        {
            context.Response.Headers.WWWAuthenticate = "Bearer realm=\"AIKernel.Monolith\"";
            await RejectAsync(context, StatusCodes.Status401Unauthorized,
                "unauthorized", "A valid Monolith bearer token is required.");
            return;
        }

        limiter.TryReplenish();
        using var lease = limiter.AttemptAcquire();
        if (!lease.IsAcquired)
        {
            var wait = lease.TryGetMetadata(MetadataName.RetryAfter, out var retryAfter)
                ? retryAfter : limiter.ReplenishmentPeriod;
            context.Response.Headers.RetryAfter =
                Math.Max(1, (long)Math.Ceiling(wait.TotalSeconds)).ToString(CultureInfo.InvariantCulture);
            await RejectAsync(context, StatusCodes.Status429TooManyRequests,
                "rate_limit_exceeded", "The Monolith request limit has been reached.");
            return;
        }

        await next(context);
    }

    private bool IsAuthenticated(HttpRequest request)
    {
        var authorization = request.Headers.Authorization;
        return authorization.Count == 1 &&
               AuthenticationHeaderValue.TryParse(authorization[0], out var header) &&
               string.Equals(header.Scheme, "Bearer", StringComparison.OrdinalIgnoreCase) &&
               header.Parameter is { Length: > 0 } token &&
               CryptographicOperations.FixedTimeEquals(apiKeyHash, HashToken(token));
    }

    private static bool IsBearerToken(string? token)
    {
        var characters = token?.TrimEnd('=');
        return characters is { Length: > 0 } &&
               characters.All(c => char.IsAsciiLetterOrDigit(c) || c is '-' or '.' or '_' or '~' or '+' or '/');
    }

    private static byte[] HashToken(string token) => SHA256.HashData(Encoding.UTF8.GetBytes(token));

    private static Task RejectAsync(HttpContext context, int status, string code, string message)
    {
        context.Response.Headers.CacheControl = "no-store";
        return Results.Json(new OpenAIErrorResponse(new OpenAIError(message, code)),
            InferenceJsonContext.Default.OpenAIErrorResponse, statusCode: status).ExecuteAsync(context);
    }

    public void Dispose() => limiter.Dispose();
}
