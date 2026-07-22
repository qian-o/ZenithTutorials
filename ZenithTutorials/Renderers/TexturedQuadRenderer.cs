namespace ZenithTutorials.Renderers;

internal unsafe sealed class TexturedQuadRenderer : IRenderer
{
    private readonly Buffer vertexBuffer;
    private readonly Buffer indexBuffer;
    private readonly Texture texture;
    private readonly Buffer constantBuffer;
    private readonly GraphicsPipeline pipeline;

    // tutorial:begin initialize-renderer
    public TexturedQuadRenderer()
    {
        string texturePath = Path.Combine(AppContext.BaseDirectory, "Assets", "Textures", "shoko.png");
        string shaderPath = App.ShaderPath("TexturedQuad.slang");

        Vertex[] vertices =
        [
            new(new(-0.5f,  0.5f, 0.0f), new(0.0f, 0.0f)),
            new(new( 0.5f,  0.5f, 0.0f), new(1.0f, 0.0f)),
            new(new( 0.5f, -0.5f, 0.0f), new(1.0f, 1.0f)),
            new(new(-0.5f, -0.5f, 0.0f), new(0.0f, 1.0f))
        ];

        uint[] indices = [0, 1, 2, 0, 2, 3];

        fixed (Vertex* pointer = vertices)
        {
            vertexBuffer = App.Context.CreateBuffer(BufferDesc.Vertex((uint)(sizeof(Vertex) * vertices.Length)));
            vertexBuffer.Upload(0, new()
            {
                Pointer = (nint)pointer,
                SizeInBytes = (uint)(sizeof(Vertex) * vertices.Length)
            });
        }

        fixed (uint* pointer = indices)
        {
            indexBuffer = App.Context.CreateBuffer(BufferDesc.Index((uint)(sizeof(uint) * indices.Length)));
            indexBuffer.Upload(0, new()
            {
                Pointer = (nint)pointer,
                SizeInBytes = (uint)(sizeof(uint) * indices.Length)
            });
        }

        texture = App.Context.LoadTextureFromFile(texturePath, generateMipMaps: true);
        constantBuffer = App.Context.CreateBuffer(BufferDesc.Constant((uint)sizeof(Constants)));

        Constants constants = new()
        {
            Texture = texture.SampledHandle,
            Sampler = App.LinearClampSampler.Handle
        };

        constantBuffer.Upload(0, new()
        {
            Pointer = (nint)(&constants),
            SizeInBytes = (uint)sizeof(Constants)
        });

        InputLayout inputLayout = new();
        inputLayout.Add(new() { Format = ElementFormat.Float3, Semantic = ElementSemantic.Position });
        inputLayout.Add(new() { Format = ElementFormat.Float2, Semantic = ElementSemantic.TexCoord });

        using Shader vertexShader = App.Context.CreateShader(ZenithCompiler.CompileFromFile(App.Context.GraphicsApi, shaderPath, "VSMain"));
        using Shader fragmentShader = App.Context.CreateShader(ZenithCompiler.CompileFromFile(App.Context.GraphicsApi, shaderPath, "FSMain"));

        pipeline = App.Context.CreateGraphicsPipeline(new()
        {
            VertexShader = vertexShader,
            FragmentShader = fragmentShader,
            InputLayouts = [inputLayout],
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
    // tutorial:end initialize-renderer

    public TextureLayout RequiredLayout => TextureLayout.ColorAttachment;

    public void Update(double deltaTime)
    {
    }

    // tutorial:begin render-textured-quad
    public void Render(CommandBuffer commandBuffer, Texture drawable)
    {
        commandBuffer.BeginRenderPass([ColorAttachment.Clear(drawable, new(0.04f, 0.055f, 0.075f, 1.0f))], null);

        commandBuffer.SetPipeline(pipeline);
        commandBuffer.SetVertexBuffer(vertexBuffer, 0, 0);
        commandBuffer.SetIndexBuffer(indexBuffer, 0, IndexFormat.UInt32);
        commandBuffer.SetConstantBuffer(constantBuffer, 0);

        commandBuffer.DrawIndexed(6, 1, 0, 0, 0);

        commandBuffer.EndRenderPass();
    }
    // tutorial:end render-textured-quad

    public void Resize(uint width, uint height)
    {
    }

    public void Dispose()
    {
        pipeline.Dispose();
        constantBuffer.Dispose();
        texture.Dispose();
        indexBuffer.Dispose();
        vertexBuffer.Dispose();
    }
}

[StructLayout(LayoutKind.Sequential)]
file struct Vertex(Vector3 position, Vector2 texCoord)
{
    public Vector3 Position = position;

    public Vector2 TexCoord = texCoord;
}

[StructLayout(LayoutKind.Explicit, Size = 16)]
file struct Constants
{
    [FieldOffset(0)]
    public ResourceHandle Texture;

    [FieldOffset(8)]
    public ResourceHandle Sampler;
}