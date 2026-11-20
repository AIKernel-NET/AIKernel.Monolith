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
    private CosmosCpuSettings? currentCosmos;
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
                if (!File.Exists(settings.Checkpoint)) return Failure("checkpoint_unavailable");
                model?.Dispose();
                current=null; currentCosmos=null; model=null; generate=null;
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
        catch (FileNotFoundException) { return Failure("checkpoint_unavailable"); }
        catch (DirectoryNotFoundException) { return Failure("checkpoint_unavailable"); }
        catch (OutOfMemoryException) { return Failure("local_memory_exhausted"); }
        finally { lease.Release(); }
    }

    internal async Task<Result<string>> GenerateCosmosAsync(CosmosCpuSettings settings,IReadOnlyList<ChatMessage> messages,
        Func<string,Task> onChunk,Dictionary<string,string> metadata,CancellationToken token)
    {
        await lease.WaitAsync(token);
        try
        {
            ObjectDisposedException.ThrowIf(disposed,this);
            using var input = new CosmosChatInput();
            if (!input.Read(messages,settings.Architecture)) return Failure("invalid_messages");
            if (currentCosmos!=settings)
            {
                if (!settings.HasCheckpoint()) return Failure("checkpoint_unavailable");
                model?.Dispose(); model=null; generate=null; current=null; currentCosmos=null;
                model=new Cosmos3Model(settings.Directory,settings.Variant,new CosmosExecutionOptions {
                    CpuThreads=settings.Threads,ContextTokens=settings.Context,EnableThinking=false,
                    ExpectedImportManifestSha256=settings.ManifestSha256,
                    Memory=settings.Memory
                });
                currentCosmos=settings;
            }
            var cosmos=(Cosmos3Model)model!;
            var before=cosmos.MemoryStatistics;
            CosmosGeneration reply;
            try { reply=await cosmos.GenerateChatAsync(input.System,input.History,input.User,settings.MaxNewTokens,token); }
            catch (ArgumentException) { return Failure("context_length_exceeded"); }
            token.ThrowIfCancellationRequested();
            metadata["backend"]="cpu";
            metadata["model_manifest_sha256"]=reply.Artifact!.ManifestSha256;
            metadata["input_tokens"]=reply.InputTokens.ToString(System.Globalization.CultureInfo.InvariantCulture);
            if (before is not null && cosmos.MemoryStatistics is { } after)
            {
                var format=System.Globalization.CultureInfo.InvariantCulture;
                metadata["weight_backing_reads"]=(after.WeightBackingReads-before.WeightBackingReads).ToString(format);
                metadata["kv_swap_reads"]=(after.KvSwapReads-before.KvSwapReads).ToString(format);
            }
            if (reply.Text.Length>0) await onChunk(reply.Text);
            return Result<string>.Success(reply.FinishReason);
        }
        catch (InvalidDataException) { return Failure("unsupported_model_format"); }
        catch (NotSupportedException) { return Failure("unsupported_model_format"); }
        catch (IOException) { return Failure("checkpoint_unavailable"); }
        catch (OutOfMemoryException) { return Failure("local_memory_exhausted"); }
        finally { lease.Release(); }
    }

    private static Result<string> Failure(string code) => Result<string>.Fail(new ErrorContext("Local CPU inference failed.",code,false));
    public void Dispose()
    {
        lease.Wait();
        try { disposed=true; model?.Dispose(); model=null; generate=null; current=null; }
        finally { lease.Release(); }
    }
}
