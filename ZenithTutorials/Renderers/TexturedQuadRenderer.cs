namespace ZenithTutorials.Renderers;

internal unsafe sealed class TexturedQuadRenderer : IRenderer
{
    private readonly Buffer vertexBuffer;
    private readonly Buffer indexBuffer;
    private readonly Texture texture;
    private readonly Sampler sampler;
    private readonly Buffer constantBuffer;
    private readonly GraphicsPipeline pipeline;

    public TexturedQuadRenderer()
    {
        Vertex[] vertices =
        [
            new()
            {
                Position = new(-0.5f, 0.5f, 0.0f),
                TexCoord = new(0.0f, 0.0f)
            },
            new()
            {
                Position = new(0.5f, 0.5f, 0.0f),
                TexCoord = new(1.0f, 0.0f)
            },
            new()
            {
                Position = new(0.5f, -0.5f, 0.0f),
                TexCoord = new(1.0f, 1.0f)
            },
            new()
            {
                Position = new(-0.5f, -0.5f, 0.0f),
                TexCoord = new(0.0f, 1.0f)
            }
        ];

        uint[] indices = [0, 1, 2, 0, 2, 3];

        vertexBuffer = App.Context.CreateBuffer(BufferDesc.Vertex((uint)(sizeof(Vertex) * vertices.Length)));

        fixed (Vertex* pointer = vertices)
        {
            vertexBuffer.Upload(0, new()
            {
                Pointer = (nint)pointer,
                SizeInBytes = (uint)(sizeof(Vertex) * vertices.Length)
            });
        }

        indexBuffer = App.Context.CreateBuffer(BufferDesc.Index((uint)(sizeof(uint) * indices.Length)));

        fixed (uint* pointer = indices)
        {
            indexBuffer.Upload(0, new()
            {
                Pointer = (nint)pointer,
                SizeInBytes = (uint)(sizeof(uint) * indices.Length)
            });
        }

        string texturePath = Path.Combine(AppContext.BaseDirectory, "Assets", "Textures", "shoko.png");
        texture = App.Context.LoadTextureFromFile(texturePath, generateMipMaps: true);
        sampler = App.Context.CreateSampler(SamplerDesc.LinearClamp());

        constantBuffer = App.Context.CreateBuffer(new()
        {
            SizeInBytes = (uint)sizeof(Constants),
            Usages = BufferUsages.Constant,
            Residency = MemoryResidency.CpuWriteOnly
        });

        Constants constants = new()
        {
            Texture = texture.SampledHandle,
            Sampler = sampler.Handle
        };

        constantBuffer.Upload(0, new()
        {
            Pointer = (nint)(&constants),
            SizeInBytes = (uint)sizeof(Constants)
        });

        InputLayout inputLayout = new();

        inputLayout.Add(new()
        {
            Format = ElementFormat.Float3,
            Semantic = ElementSemantic.Position
        });
        inputLayout.Add(new()
        {
            Format = ElementFormat.Float2,
            Semantic = ElementSemantic.TexCoord
        });

        string shaderPath = App.ShaderPath("TexturedQuad.slang");
        ShaderDesc vertexDesc = ZenithCompiler.CompileFromFile(App.Context.GraphicsApi, shaderPath, "VSMain");
        ShaderDesc fragmentDesc = ZenithCompiler.CompileFromFile(App.Context.GraphicsApi, shaderPath, "FSMain");

        using Shader vertexShader = App.Context.CreateShader(vertexDesc);
        using Shader fragmentShader = App.Context.CreateShader(fragmentDesc);

        pipeline = App.Context.CreateGraphicsPipeline(new()
        {
            VertexShader = vertexShader,
            FragmentShader = fragmentShader,
            InputLayouts = [inputLayout],
            PrimitiveTopology = PrimitiveTopology.TriangleList,
            AttachmentFormats = new()
            {
                ColorFormats = [App.ColorFormat],
                DepthStencilFormat = null,
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
        commandBuffer.Transition(drawable, default, TextureLayout.Undefined, TextureLayout.ColorAttachment);

        commandBuffer.BeginRenderPass([ColorAttachment.Clear(drawable, new(0.04f, 0.055f, 0.075f, 1.0f))],
                                      null);

        commandBuffer.SetPipeline(pipeline);
        commandBuffer.SetVertexBuffer(vertexBuffer, 0, 0);
        commandBuffer.SetIndexBuffer(indexBuffer, 0, IndexFormat.UInt32);
        commandBuffer.SetConstantBuffer(constantBuffer, 0);
        commandBuffer.DrawIndexed(6, 1, 0, 0, 0);

        commandBuffer.EndRenderPass();
    }

    public void Resize(uint width, uint height)
    {
    }

    public void Dispose()
    {
        pipeline.Dispose();
        constantBuffer.Dispose();
        sampler.Dispose();
        texture.Dispose();
        indexBuffer.Dispose();
        vertexBuffer.Dispose();
    }
}

[StructLayout(LayoutKind.Explicit, Size = 20)]
file struct Vertex
{
    [FieldOffset(0)]
    public Vector3 Position;

    [FieldOffset(12)]
    public Vector2 TexCoord;
}

[StructLayout(LayoutKind.Explicit, Size = 16)]
file struct Constants
{
    [FieldOffset(0)]
    public ResourceHandle Texture;

    [FieldOffset(8)]
    public ResourceHandle Sampler;
}