using System.Text.Json;
using AIKernel.Abstractions.Events;
using AIKernel.Dtos.Events;
using AIKernel.Monolith.Internal;

namespace AIKernel.Monolith.Hosting;

internal sealed class FileInferenceAuditWriter(MountOptions options) : IAuditEventWriter, IDisposable
{
    private readonly string path = Path.Combine(options.ConfigRoot, "inference-audit.jsonl");
    private readonly SemaphoreSlim appendLock = new(1);

    public async ValueTask LogAuditEventAsync(AuditEvent auditEvent, CancellationToken cancellationToken = default)
    {
        var line = JsonSerializer.Serialize(auditEvent, InferenceJsonContext.Default.AuditEvent);
        await appendLock.WaitAsync(cancellationToken);
        try
        {
            await File.AppendAllTextAsync(path, line + "\n", cancellationToken);
        }
        finally
        {
            appendLock.Release();
        }
    }

    public void Dispose() => appendLock.Dispose();
}
