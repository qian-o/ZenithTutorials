namespace ZenithTutorials.Renderers;

internal unsafe sealed class ComputeShaderRenderer : IRenderer
{
    private const uint ThreadGroupSize = 16;

    private readonly Texture inputTexture;
    private readonly Texture outputTexture;
    private readonly Sampler sampler;
    private readonly Buffer constantBuffer;
    private readonly ComputePipeline computePipeline;
    private readonly GraphicsPipeline displayPipeline;

    private bool processed;

    public ComputeShaderRenderer()
    {
        string texturePath = Path.Combine(AppContext.BaseDirectory, "Assets", "Textures", "shoko.png");
        inputTexture = App.Context.LoadTextureFromFile(texturePath, generateMipMaps: false);

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

        sampler = App.Context.CreateSampler(new()
        {
            MinFilter = FilterMode.Linear,
            MagFilter = FilterMode.Linear,
            MipFilter = FilterMode.Linear,
            AddressU = AddressMode.Clamp,
            AddressV = AddressMode.Clamp,
            AddressW = AddressMode.Clamp,
            CompareOp = CompareOp.Never,
            MaxAnisotropy = 1,
            LodBias = 0.0f,
            MinLod = 0.0f,
            MaxLod = float.MaxValue,
            BorderColor = BorderColor.TransparentBlack
        });

        constantBuffer = App.Context.CreateBuffer(new()
        {
            SizeInBytes = (uint)sizeof(ComputeConstants),
            Usages = BufferUsages.Constant,
            Residency = MemoryResidency.CpuWriteOnly
        });

        ComputeConstants constants = new()
        {
            Width = inputTexture.Desc.Width,
            Height = inputTexture.Desc.Height,
            Input = inputTexture.SampledHandle,
            Output = outputTexture.StorageHandle,
            Image = outputTexture.SampledHandle,
            Sampler = sampler.Handle
        };

        constantBuffer.Upload(0, new()
        {
            Pointer = (nint)(&constants),
            SizeInBytes = (uint)sizeof(ComputeConstants)
        });

        string shaderPath = App.ShaderPath("ComputeShader.slang");
        ShaderDesc computeDesc = ZenithCompiler.CompileFromFile(App.Context.GraphicsApi, shaderPath, "CSMain");
        ShaderDesc vertexDesc = ZenithCompiler.CompileFromFile(App.Context.GraphicsApi, shaderPath, "VSMain");
        ShaderDesc fragmentDesc = ZenithCompiler.CompileFromFile(App.Context.GraphicsApi, shaderPath, "FSMain");

        using Shader computeShader = App.Context.CreateShader(computeDesc);
        using Shader vertexShader = App.Context.CreateShader(vertexDesc);
        using Shader fragmentShader = App.Context.CreateShader(fragmentDesc);

        computePipeline = App.Context.CreateComputePipeline(new() { ComputeShader = computeShader });
        displayPipeline = App.Context.CreateGraphicsPipeline(new()
        {
            VertexShader = vertexShader,
            FragmentShader = fragmentShader,
            InputLayouts = [],
            PrimitiveTopology = PrimitiveTopology.TriangleList,
            AttachmentFormats = new()
            {
                ColorFormats = [App.ColorFormat],
                SampleCount = SampleCount.Count1
            },
            RenderState = new()
            {
                Rasterizer = RasterizerState.CullNone(),
                DepthStencil = DepthStencilState.DepthNone(),
                Blend = BlendState.Opaque()
            }
        });
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
            commandBuffer.Dispatch((inputTexture.Desc.Width + ThreadGroupSize - 1) / ThreadGroupSize,
                                   (inputTexture.Desc.Height + ThreadGroupSize - 1) / ThreadGroupSize,
                                   1);

            commandBuffer.Transition(outputTexture, default, TextureLayout.Storage, TextureLayout.Sampled);

            processed = true;
        }

        uint width = Math.Min(outputTexture.Desc.Width, App.Width);
        uint height = Math.Min(outputTexture.Desc.Height, App.Height);
        int x = (int)((App.Width - width) / 2);
        int y = (int)((App.Height - height) / 2);

        commandBuffer.Transition(drawable, default, TextureLayout.Undefined, TextureLayout.ColorAttachment);

        commandBuffer.BeginRenderPass([ColorAttachment.Clear(drawable, new(0.04f, 0.055f, 0.075f, 1.0f))],
                                      null);

        commandBuffer.SetPipeline(displayPipeline);
        commandBuffer.SetViewports([new()
        {
            X = x,
            Y = y,
            Width = width,
            Height = height,
            MaxDepth = 1.0f
        }]);
        commandBuffer.SetScissors([new()
        {
            X = x,
            Y = y,
            Width = width,
            Height = height
        }]);
        commandBuffer.SetConstantBuffer(constantBuffer, 0);
        commandBuffer.Draw(3, 1, 0, 0);

        commandBuffer.EndRenderPass();
    }

    public void Resize(uint width, uint height)
    {
    }

    public void Dispose()
    {
        displayPipeline.Dispose();
        computePipeline.Dispose();
        constantBuffer.Dispose();
        sampler.Dispose();
        outputTexture.Dispose();
        inputTexture.Dispose();
    }
}

[StructLayout(LayoutKind.Explicit, Size = 256)]
file struct ComputeConstants
{
    [FieldOffset(0)]
    public uint Width;

    [FieldOffset(4)]
    public uint Height;

    [FieldOffset(8)]
    public ResourceHandle Input;

    [FieldOffset(16)]
    public ResourceHandle Output;

    [FieldOffset(24)]
    public ResourceHandle Image;

    [FieldOffset(32)]
    public ResourceHandle Sampler;
}