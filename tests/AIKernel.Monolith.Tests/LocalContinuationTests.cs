using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using AIKernel.Core.Governance;
using AIKernel.Kernel.Memory;
using AIKernel.Monolith.Models.Local;
using AIKernel.Monolith.OpenAI;
using Xunit;

namespace AIKernel.Monolith.Tests;

public sealed class LocalContinuationTests
{
    private static bool configured;
    [NativeComputeFact]
    public async Task RestoredKvCacheProducesIdenticalLogitsWithoutPrefill()
    {
        ConfigureInterop();
        using var workspace = new TestWorkspace();
        var checkpoint = Path.Combine(workspace.Path, "model.bin");
        var tokenizer = Path.Combine(workspace.Path, "tokenizer.bin");
        var snapshot = Path.Combine(workspace.Path, "cache.snapshot");
        WriteModel(checkpoint, tokenizer);
        using var model = new PagedLlamaModel(new(checkpoint, tokenizer, null, workspace.Path,
            4096, 2, 2, 0, 8, 1, 1, 1, 0, 1, InferencePriority.Normal, long.MaxValue, default, default),
            CancellationToken.None);
        ChatMessage[] messages = [new("user", "ab")];
        float[] expected = new float[5];
        float[] restored = new float[5];
        Dictionary<string, string> originalAudit = new();
        Dictionary<string, string> resumedAudit = new();
        await model.GenerateAsync(messages, static _ => Task.CompletedTask, originalAudit, null,
            CancellationToken.None, new(snapshot, null, expected));
        await model.GenerateAsync(messages, static _ => Task.CompletedTask, resumedAudit, null,
            CancellationToken.None, new(null, snapshot, restored));

        Assert.Equal("2", originalAudit["local_prefill_tokens"]);
        Assert.Equal("0", resumedAudit["local_prefill_tokens"]);
        Assert.Equal(expected.Select(BitConverter.SingleToInt32Bits), restored.Select(BitConverter.SingleToInt32Bits));
        Assert.Contains(expected, value => value != 0);
        if (Environment.GetEnvironmentVariable("AIKERNEL_TEST_REQUIRE_GPU") == "1")
            Assert.Equal("cuda", resumedAudit["backend"]);
    }

    [NativeComputeFact(windowsGpu: true)]
    public void SharedDmaWritesTheVramTierWithNoManagedAllocationOnReuse()
    {
        ConfigureInterop();
        const int capacity = 4096;
        var name = $"aikernel-ipc-{Guid.NewGuid():N}";
        using var producer = MemoryMappedFile.CreateNew(name, capacity);
        using var writer = producer.CreateViewAccessor();
        writer.Write(0, 0x784512ab);
        using var region = new Win32MemoryMapper().OpenShared(name, capacity).Value!;
        using var device = new CudaVirtualMemoryDevice();
        using var workspace = new TestWorkspace();
        using var space = new VirtualMemorySpace(1, capacity, 1, workspace.Path, 1, device);
        Assert.True(space.ImportSharedBlock(0, region, 0).IsSuccess);
        Assert.Equal(VirtualMemoryTier.Vram, space.GetBlockTier(0));
        Span<byte> output = stackalloc byte[capacity];
        space.ReadBlock(0, output);
        Assert.Equal(0x784512ab, MemoryMarshal.Read<int>(output));

        using var target = device.UploadShared(region, 0, capacity).Value!;
        for (var i = 0; i < 100; i++) _ = device.UploadShared(region, 0, capacity, target);
        var before = GC.GetAllocatedBytesForCurrentThread();
        var success = true;
        for (var i = 0; i < 64; i++) success &= device.UploadShared(region, 0, capacity, target).IsSuccess;
        var after = GC.GetAllocatedBytesForCurrentThread();
        Assert.True(success);
        Assert.Equal(before, after);
    }

    private static void ConfigureInterop()
    {
        if (configured) return;
        var library = Environment.GetEnvironmentVariable("AIKERNEL_TEST_NATIVE_LIBRARY")!;
        NativeLibrary.SetDllImportResolver(typeof(CudaVirtualMemoryDevice).Assembly,
            (name, _, _) => name == "aikernel_cuda_driver" ? NativeLibrary.Load(library) : 0);
        configured = true;
    }

    private static void WriteModel(string modelPath, string tokenizerPath)
    {
        using (var output = new BinaryWriter(File.Create(modelPath)))
        {
            const int width = 4, hidden = 4, layers = 2, vocabulary = 5, context = 8;
            foreach (var value in new[] { width, hidden, layers, 1, 1, vocabulary, context }) output.Write(value);
            Weights(vocabulary * width);
            Weights(layers * width, scale: true);
            for (var matrix = 0; matrix < 4; matrix++) Weights(layers * width * width);
            Weights(layers * width, scale: true);
            for (var matrix = 0; matrix < 3; matrix++) Weights(layers * width * hidden);
            Weights(width, scale: true);
            Weights(context * width);

            void Weights(int length, bool scale = false)
            {
                for (var i = 0; i < length; i++) output.Write(scale ? 1f : (i % 11 - 5) * 0.06f);
            }
        }

        using var vocabularyWriter = new BinaryWriter(File.Create(tokenizerPath));
        vocabularyWriter.Write(8);
        foreach (var piece in new[] { "<unk>", "<s>", "</s>", "a", "b" })
        {
            var bytes = System.Text.Encoding.UTF8.GetBytes(piece);
            vocabularyWriter.Write(1f);
            vocabularyWriter.Write(bytes.Length);
            vocabularyWriter.Write(bytes);
        }
    }

    private sealed class TestWorkspace : IDisposable
    {
        private readonly DirectoryInfo directory = Directory.CreateTempSubdirectory("aikernel-continuation-");
        internal string Path => directory.FullName;
        public void Dispose() => directory.Delete(recursive: true);
    }

    public sealed class NativeComputeFactAttribute : FactAttribute
    {
        public NativeComputeFactAttribute(bool windowsGpu = false)
        {
            if (!File.Exists(Environment.GetEnvironmentVariable("AIKERNEL_TEST_NATIVE_LIBRARY")))
                Skip = "Set AIKERNEL_TEST_NATIVE_LIBRARY to the CUDA Driver ABI shared library.";
            else if (windowsGpu && (!OperatingSystem.IsWindows() ||
                Environment.GetEnvironmentVariable("AIKERNEL_TEST_REQUIRE_GPU") != "1"))
                Skip = "This allocation test requires Windows and AIKERNEL_TEST_REQUIRE_GPU=1.";
        }
    }
}
