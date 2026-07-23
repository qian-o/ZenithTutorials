namespace ZenithTutorials.Renderers;

internal unsafe sealed class TexturedQuadRenderer : IRenderer
{
    private readonly Buffer vertexBuffer;
    private readonly Buffer indexBuffer;
    private readonly Texture texture;
    private readonly Buffer constantBuffer;
    private readonly GraphicsPipeline pipeline;

    public TexturedQuadRenderer()
    {
        Vertex[] vertices =
        [
            new(new(-0.5f,  0.5f, 0.0f), new(0.0f, 0.0f)),
            new(new( 0.5f,  0.5f, 0.0f), new(1.0f, 0.0f)),
            new(new( 0.5f, -0.5f, 0.0f), new(1.0f, 1.0f)),
            new(new(-0.5f, -0.5f, 0.0f), new(0.0f, 1.0f))
        ];

        uint[] indices = [0, 1, 2, 0, 2, 3];

        vertexBuffer = App.LoadBuffer(vertices, BufferUsages.Vertex);
        indexBuffer = App.LoadBuffer(indices, BufferUsages.Index);
        texture = App.LoadTexture("shoko.png", true);

        Constants constants = new()
        {
            Texture = texture.SampledHandle,
            Sampler = App.LinearClampSampler.Handle
        };

        constantBuffer = App.LoadBuffer([constants], BufferUsages.Constant);

        InputLayout inputLayout = new();
        inputLayout.Add(new() { Format = ElementFormat.Float3, Semantic = ElementSemantic.Position });
        inputLayout.Add(new() { Format = ElementFormat.Float2, Semantic = ElementSemantic.TexCoord });

        using Shader vertexShader = App.LoadShader("TexturedQuad.slang", "VSMain");
        using Shader fragmentShader = App.LoadShader("TexturedQuad.slang", "FSMain");

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

    public TextureLayout RequiredLayout => TextureLayout.ColorAttachment;

    public void Update(double deltaTime)
    {
    }

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