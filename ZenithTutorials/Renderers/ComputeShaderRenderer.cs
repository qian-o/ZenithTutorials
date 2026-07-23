namespace ZenithTutorials.Renderers;

internal class ComputeShaderRenderer : IRenderer
{
    private const uint ThreadGroupSize = 16;

    private readonly Texture inputTexture;
    private readonly Texture outputTexture;
    private readonly Buffer constantBuffer;
    private readonly ComputePipeline computePipeline;

    private bool processed;

    public ComputeShaderRenderer()
    {
        inputTexture = App.LoadTexture("shoko.png", false);

        outputTexture = App.Context.CreateTexture(new()
        {
            Type = TextureType.Texture2D,
            Format = PixelFormat.R32G32B32A32Float,
            Width = inputTexture.Desc.Width,
            Height = inputTexture.Desc.Height,
            Depth = 1,
            MipLevels = 1,
            ArrayLayers = 1,
            SampleCount = SampleCount.Count1,
            Usages = TextureUsages.Sampled | TextureUsages.Storage
        });

        Constants constants = new()
        {
            Width = inputTexture.Desc.Width,
            Height = inputTexture.Desc.Height,
            Input = inputTexture.SampledHandle,
            Output = outputTexture.StorageHandle
        };

        constantBuffer = App.LoadBuffer([constants], BufferUsages.Constant);

        using Shader computeShader = App.LoadShader("ComputeShader.slang", "CSMain");

        computePipeline = App.Context.CreateComputePipeline(new() { ComputeShader = computeShader });
    }

    public void Update(double deltaTime)
    {
    }

    public void Render(CommandBuffer commandBuffer, Texture drawable)
    {
        if (!processed)
        {
            commandBuffer.Transition(outputTexture, default, TextureLayout.Undefined, TextureLayout.Storage);

            commandBuffer.SetPipeline(computePipeline);
            commandBuffer.SetConstantBuffer(constantBuffer, 0);
            commandBuffer.Dispatch((inputTexture.Desc.Width + ThreadGroupSize - 1) / ThreadGroupSize, (inputTexture.Desc.Height + ThreadGroupSize - 1) / ThreadGroupSize, 1);

            commandBuffer.Transition(outputTexture, default, TextureLayout.Storage, TextureLayout.Sampled);

            processed = true;
        }

        App.PresentTexture(commandBuffer, drawable, outputTexture, false);
    }

    public void Resize(uint width, uint height)
    {
    }

    public void Dispose()
    {
        computePipeline.Dispose();
        constantBuffer.Dispose();
        outputTexture.Dispose();
        inputTexture.Dispose();
    }
}

[StructLayout(LayoutKind.Explicit, Size = 256)]
file struct Constants
{
    [FieldOffset(0)]
    public uint Width;

    [FieldOffset(4)]
    public uint Height;

    [FieldOffset(8)]
    public ResourceHandle Input;

    [FieldOffset(16)]
    public ResourceHandle Output;
}