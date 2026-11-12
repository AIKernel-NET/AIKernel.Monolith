using System.Globalization;
using AIKernel.Monolith.Internal;
using AIKernel.Monolith.Models;

namespace AIKernel.Monolith.OpenAI;

internal sealed class InferenceErrorResult(InferenceError error) : IResult
{
    internal static OpenAIErrorResponse CreatePayload(InferenceError error) =>
        new(new OpenAIError(error.Message, error.Code, error.RetryAfterSeconds));

    public Task ExecuteAsync(HttpContext context)
    {
        if (error.RetryAfterSeconds is int seconds)
            context.Response.Headers.RetryAfter = seconds.ToString(CultureInfo.InvariantCulture);

        return Results.Json(CreatePayload(error), InferenceJsonContext.Default.OpenAIErrorResponse,
            statusCode: error.Status).ExecuteAsync(context);
    }
}
