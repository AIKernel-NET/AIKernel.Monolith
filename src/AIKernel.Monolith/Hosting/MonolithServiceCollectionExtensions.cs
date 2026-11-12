using AIKernel.Abstractions.Events;
using AIKernel.Abstractions.Governance;
using AIKernel.Core.Governance;
using AIKernel.Monolith.Capabilities;
using AIKernel.Monolith.Internal;
using AIKernel.Monolith.Models;
using AIKernel.Monolith.Models.Local;
using AIKernel.Monolith.Rag;
using AIKernel.Monolith.Routing;
using AIKernel.Monolith.VFS;

namespace AIKernel.Monolith.Hosting;

/// <summary>
/// EN: Dependency injection setup for the AIKernel.Monolith runtime. JA: AIKernel.Monolith ランタイム向け DI 設定です。
/// </summary>
public static class MonolithServiceCollectionExtensions
{
    /// <summary>
    /// EN: Registers the Semantic OS monolith services. JA: Semantic OS Monolith のサービス群を登録します。
    /// </summary>
    public static IServiceCollection AddAIKernelMonolith(this IServiceCollection services, MountOptions mountOptions)
    {
        services.AddSingleton(mountOptions);
        services.AddSingleton<IVirtualFileSystem, LocalVirtualFileSystem>();
        services.AddSingleton<MemoryVirtualFileSystem>();
        services.AddSingleton<MountTable>();
        services.AddSingleton<ICapabilityMetadataReader, CapabilityMetadataReader>();
        services.AddSingleton<ICapabilityAssemblyLoader, CapabilityAssemblyLoader>();
        services.AddSingleton<ICapabilityInvokerBinder, CapabilityInvokerBinder>();
        services.AddSingleton<CapabilityLoader>();
        services.AddSingleton<ModelProviderRegistry>();
        services.AddSingleton<IDecisionGate>(_ => new CtgDecisionGateEvaluator());
        services.AddSingleton<IAuditEventWriter, FileInferenceAuditWriter>();
        services.AddSingleton<PagedLlamaPool>();
        services.AddSingleton<GovernedModelInference>();
        services.AddSingleton<CudaTopology>();
        services.AddSingleton<ModelAdmission>();
        services.AddSingleton<RagService>();
        services.AddHttpClient(GovernedModelInference.HttpClientName)
            .ConfigurePrimaryHttpMessageHandler(() => new HttpClientHandler { AllowAutoRedirect = false });
        services.ConfigureHttpJsonOptions(options =>
        {
            options.SerializerOptions.TypeInfoResolverChain.Insert(0, MonolithJsonContext.Default);
            options.SerializerOptions.TypeInfoResolverChain.Insert(0, InferenceJsonContext.Default);
            options.SerializerOptions.TypeInfoResolverChain.Insert(0, RagJsonContext.Default);
        });
        services.AddSingleton<Routing.IRouter, DeterministicRouter>();
        services.AddSingleton<RouteTable>();
        services.AddSingleton<SemanticRouteResolver>();
        services.AddSingleton<CapabilityRouteBinder>();
        services.AddSingleton<ModelRouteBinder>();
        services.AddSingleton<ToolRouteBinder>();
        services.AddSingleton<SemanticRuntime>();
        return services;
    }
}
