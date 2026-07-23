namespace ZenithTutorials.Renderers;

internal class HelloTriangleRenderer : IRenderer
{
    private readonly Buffer vertexBuffer;
    private readonly GraphicsPipeline pipeline;

    public HelloTriangleRenderer()
    {
        Vertex[] vertices =
        [
            new(new(0.0f, 0.6f, 0.0f), new(1.0f, 0.2f, 0.15f, 1.0f)),
            new(new(0.6f, -0.5f, 0.0f), new(0.15f, 0.85f, 0.35f, 1.0f)),
            new(new(-0.6f, -0.5f, 0.0f), new(0.2f, 0.45f, 1.0f, 1.0f))
        ];

        vertexBuffer = App.LoadBuffer(vertices, BufferUsages.Vertex);

        InputLayout inputLayout = new();
        inputLayout.Add(new() { Format = ElementFormat.Float3, Semantic = ElementSemantic.Position });
        inputLayout.Add(new() { Format = ElementFormat.Float4, Semantic = ElementSemantic.Color });

        using Shader vertexShader = App.LoadShader("HelloTriangle.slang", "VSMain");
        using Shader fragmentShader = App.LoadShader("HelloTriangle.slang", "FSMain");

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

    public void Update(double deltaTime)
    {
    }

    public void Render(CommandBuffer commandBuffer, Texture drawable)
    {
        commandBuffer.BeginRenderPass([ColorAttachment.Clear(drawable, new(0.04f, 0.055f, 0.075f, 1.0f))], null);

        commandBuffer.SetPipeline(pipeline);
        commandBuffer.SetVertexBuffer(vertexBuffer, 0, 0);

        commandBuffer.Draw(3, 1, 0, 0);

        commandBuffer.EndRenderPass();
    }

    public void Resize(uint width, uint height)
    {
    }

    public void Dispose()
    {
        pipeline.Dispose();
        vertexBuffer.Dispose();
    }
}

[StructLayout(LayoutKind.Sequential)]
file struct Vertex(Vector3 position, Vector4 color)
{
    public Vector3 Position = position;

    public Vector4 Color = color;
}