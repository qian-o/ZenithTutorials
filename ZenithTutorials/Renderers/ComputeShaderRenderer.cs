namespace ZenithTutorials.Renderers;

internal unsafe class ComputeShaderRenderer : IRenderer
{
    private const uint ThreadGroupSize = 16;

    private const string ComputeShaderSource = """
        Texture2D inputTexture;
        RWTexture2D outputTexture;

        [numthreads(16, 16, 1)]
        void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID)
        {
            uint width, height;
            outputTexture.GetDimensions(width, height);

            // Bounds check
            if (dispatchThreadID.x >= width || dispatchThreadID.y >= height)
            {
                return;
            }

            // Read input pixel
            float4 color = inputTexture[dispatchThreadID.xy];

            // Convert to grayscale using luminance weights
            float gray = dot(color.rgb, float3(0.299, 0.587, 0.114));

            // Write to output
            outputTexture[dispatchThreadID.xy] = float4(gray, gray, gray, color.a);
        }
        """;

    private readonly Texture inputTexture;
    private readonly Texture outputTexture;
    private readonly ResourceLayout resourceLayout;
    private readonly ResourceSet resourceSet;
    private readonly ComputePipeline pipeline;

    private bool processed;

    public ComputeShaderRenderer()
    {
        inputTexture = App.Context.LoadTextureFromFile(Path.Combine(AppContext.BaseDirectory, "Assets", "shoko.png"), generateMipMaps: false);

        outputTexture = App.Context.CreateTexture(new()
        {
            Type = TextureType.Texture2D,
            Format = PixelFormat.R8G8B8A8UNorm,
            Width = inputTexture.Desc.Width,
            Height = inputTexture.Desc.Height,
            Depth = 1,
            MipLevels = 1,
            ArrayLayers = 1,
            SampleCount = SampleCount.Count1,
            Flags = TextureUsageFlags.ShaderResource | TextureUsageFlags.UnorderedAccess
        });

        resourceLayout = App.Context.CreateResourceLayout(new()
        {
            Bindings = BindingHelper.Bindings
            (
                new() { Type = ResourceType.Texture, Count = 1, StageFlags = ShaderStageFlags.Compute },
                new() { Type = ResourceType.TextureReadWrite, Count = 1, StageFlags = ShaderStageFlags.Compute }
            )
        });

        resourceSet = App.Context.CreateResourceSet(new()
        {
            Layout = resourceLayout,
            Resources = [inputTexture, outputTexture]
        });

        using Shader computeShader = App.Context.LoadShaderFromSource(ComputeShaderSource, "CSMain", ShaderStageFlags.Compute);

        pipeline = App.Context.CreateComputePipeline(new()
        {
            Compute = computeShader,
            ResourceLayouts = [resourceLayout],
            ThreadGroupSizeX = ThreadGroupSize,
            ThreadGroupSizeY = ThreadGroupSize,
            ThreadGroupSizeZ = 1
        });
    }

    public void Update(double deltaTime)
    {
    }

    public void Render()
    {
        CommandBuffer commandBuffer = App.Context.Graphics.CommandBuffer();

        if (!processed)
        {
            uint dispatchX = (inputTexture.Desc.Width + ThreadGroupSize - 1) / ThreadGroupSize;
            uint dispatchY = (inputTexture.Desc.Height + ThreadGroupSize - 1) / ThreadGroupSize;

            commandBuffer.SetPipeline(pipeline);
            commandBuffer.SetResourceSet(resourceSet, 0);
            commandBuffer.Dispatch(dispatchX, dispatchY, 1);

            processed = true;
        }

        // Copy the processed texture to the swap chain's color target (centered)
        Texture colorTarget = App.SwapChain.FrameBuffer.Desc.ColorAttachments[0].Target;

        // Clamp copy region to fit within both textures
        uint copyWidth = Math.Min(outputTexture.Desc.Width, App.Width);
        uint copyHeight = Math.Min(outputTexture.Desc.Height, App.Height);

        // Center the copy region
        uint srcX = (outputTexture.Desc.Width - copyWidth) / 2;
        uint srcY = (outputTexture.Desc.Height - copyHeight) / 2;
        uint destX = (App.Width - copyWidth) / 2;
        uint destY = (App.Height - copyHeight) / 2;

        commandBuffer.CopyTexture(outputTexture,
                                  default,
                                  new() { X = srcX, Y = srcY, Z = 0 },
                                  colorTarget,
                                  default,
                                  new() { X = destX, Y = destY, Z = 0 },
                                  new() { Width = copyWidth, Height = copyHeight, Depth = 1 });

        commandBuffer.Submit(waitForCompletion: true);
    }

    public void Resize(uint width, uint height)
    {
    }

    public void Dispose()
    {
        pipeline.Dispose();
        resourceSet.Dispose();
        resourceLayout.Dispose();
        outputTexture.Dispose();
        inputTexture.Dispose();
    }
}
