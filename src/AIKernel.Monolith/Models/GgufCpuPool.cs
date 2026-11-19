using AIKernel.Common.Results;
using AIKernel.Kernel.Inference;
using AIKernel.Monolith.OpenAI;

namespace AIKernel.Monolith.Models;

// One model and one generation at a time. Hold the lease through generation so a
// concurrent request for another checkpoint cannot dispose a live mapping.
internal sealed class GgufCpuPool : IDisposable
{
    private readonly SemaphoreSlim lease = new(1);
    private GgufCpuSettings? current;
    private IDisposable? model;
    private Func<string,string,int,CancellationToken,Task<CpuGeneration>>? generate;
    private bool disposed;

    internal async Task<Result<string>> GenerateAsync(GgufCpuSettings settings, IReadOnlyList<ChatMessage> messages,
        Func<string,Task> onChunk, Dictionary<string,string> metadata, CancellationToken token)
    {
        if (messages.Count is < 1 or > 2 || messages[^1].Role != "user" ||
            string.IsNullOrWhiteSpace(messages[^1].Content) || messages.Count == 2 && messages[0].Role is not ("system" or "developer"))
            return Failure("invalid_messages");
        string system = messages.Count == 2 ? messages[0].Content! : "";
        string user = messages[^1].Content!;
        if (system.Length > 8192 || user.Length > 4096) return Failure("context_length_exceeded");
        await lease.WaitAsync(token);
        try
        {
            ObjectDisposedException.ThrowIf(disposed,this);
            if (current != settings)
            {
                model?.Dispose();
                current=null; model=null; generate=null;
                switch (settings.Architecture)
                {
                    case "qwen3":
                        var bonsai=new Qwen3CpuModel(settings.Checkpoint,settings.Threads,settings.Context);
                        model=bonsai; generate=bonsai.GenerateAsync; break;
                    case "gemma4-e2b":
                        var gemma=new Gemma4E2BCpuModel(settings.Checkpoint,settings.Threads,settings.Context,4L<<30,
                            new() { Memory=new() { SwapDirectory=settings.SwapDirectory } });
                        model=gemma; generate=gemma.GenerateAsync; break;
                    case "gpt-oss-20b":
                        var gpt=new GptOss20BCpuModel(settings.Checkpoint,settings.Threads,settings.Context,
                            new GgufMemoryOptions { SwapDirectory=settings.SwapDirectory });
                        model=gpt; generate=gpt.GenerateAsync; break;
                    default: return Failure("unsupported_model_format");
                }
                current=settings;
            }
            CpuGeneration reply;
            try { reply=await generate!(system,user,settings.MaxNewTokens,token); }
            catch (ArgumentException) { return Failure("context_length_exceeded"); }
            if (string.IsNullOrEmpty(reply.Text)) return Failure("empty_provider_response");
            token.ThrowIfCancellationRequested();
            metadata["backend"]=reply.CpuBackend;
            metadata["activation_precision"]=reply.ActivationPrecision;
            await onChunk(reply.Text);
            return Result<string>.Success(reply.FinishReason);
        }
        catch (NotSupportedException) { return Failure("unsupported_model_format"); }
        catch (OutOfMemoryException) { return Failure("local_memory_exhausted"); }
        finally { lease.Release(); }
    }

    private static Result<string> Failure(string code) => Result<string>.Fail(new ErrorContext("CPU GGUF inference failed.",code,false));
    public void Dispose()
    {
        lease.Wait();
        try { disposed=true; model?.Dispose(); model=null; generate=null; current=null; }
        finally { lease.Release(); }
    }
}
