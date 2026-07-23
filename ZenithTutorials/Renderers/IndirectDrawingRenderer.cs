namespace ZenithTutorials.Renderers;

internal unsafe class IndirectDrawingRenderer : IRenderer
{
    private const uint InstanceCount = 25;
    private const uint GridWidth = 5;
    private const PixelFormat DepthFormat = PixelFormat.D32FloatS8UInt;

    private readonly Buffer vertexBuffer;
    private readonly Buffer indexBuffer;
    private readonly Buffer indirectBuffer;
    private readonly Buffer instanceBuffer;
    private readonly Buffer constantBuffer;
    private readonly GraphicsPipeline pipeline;

    private Texture? depthTexture;
    private float rotationAngle;

    public IndirectDrawingRenderer()
    {
        Vertex[] vertices =
        [
            new(new(-0.5f, -0.5f,  0.5f), Vector4.One),
            new(new( 0.5f, -0.5f,  0.5f), Vector4.One),
            new(new( 0.5f,  0.5f,  0.5f), Vector4.One),
            new(new(-0.5f,  0.5f,  0.5f), Vector4.One),
            new(new(-0.5f, -0.5f, -0.5f), Vector4.One),
            new(new( 0.5f, -0.5f, -0.5f), Vector4.One),
            new(new( 0.5f,  0.5f, -0.5f), Vector4.One),
            new(new(-0.5f,  0.5f, -0.5f), Vector4.One)
        ];

        uint[] indices =
        [
            0, 1, 2, 0, 2, 3,
            5, 4, 7, 5, 7, 6,
            4, 0, 3, 4, 3, 7,
            1, 5, 6, 1, 6, 2,
            3, 2, 6, 3, 6, 7,
            4, 5, 1, 4, 1, 0
        ];

        IndirectDrawIndexedArgs arguments = new()
        {
            IndexCount = (uint)indices.Length,
            InstanceCount = InstanceCount
        };

        vertexBuffer = App.LoadBuffer(vertices, BufferUsages.Vertex);
        indexBuffer = App.LoadBuffer(indices, BufferUsages.Index);
        indirectBuffer = App.LoadBuffer([arguments], BufferUsages.Indirect);
        instanceBuffer = App.LoadBuffer(new Instance[InstanceCount], BufferUsages.StorageReadOnly);
        constantBuffer = App.LoadBuffer([new Constants()], BufferUsages.Constant);

        InputLayout inputLayout = new();
        inputLayout.Add(new() { Format = ElementFormat.Float3, Semantic = ElementSemantic.Position });
        inputLayout.Add(new() { Format = ElementFormat.Float4, Semantic = ElementSemantic.Color });

        using Shader vertexShader = App.LoadShader("IndirectDrawing.slang", "VSMain");
        using Shader fragmentShader = App.LoadShader("IndirectDrawing.slang", "FSMain");

        pipeline = App.Context.CreateGraphicsPipeline(new()
        {
            VertexShader = vertexShader,
            FragmentShader = fragmentShader,
            InputLayouts = [inputLayout],
            PrimitiveTopology = PrimitiveTopology.TriangleList,
            AttachmentFormats = new()
            {
                ColorFormats = [App.ColorFormat],
                DepthStencilFormat = DepthFormat,
                SampleCount = SampleCount.Count1
            },
            RenderState = new()
            {
                Rasterizer = RasterizerState.CullBack(),
                DepthStencil = DepthStencilState.DepthReadWrite(),
                Blend = BlendState.Opaque()
            }
        });

        Resize(App.Width, App.Height);
        Update(0.0);
    }

    public TextureLayout RequiredLayout => TextureLayout.ColorAttachment;

    public void Update(double deltaTime)
    {
        rotationAngle += (float)deltaTime;

        Instance[] instances = new Instance[InstanceCount];

        fixed (Instance* pointer = instances)
        {
            for (uint index = 0; index < InstanceCount; index++)
            {
                uint x = index % GridWidth;
                uint y = index / GridWidth;
                float offsetX = (x - ((GridWidth - 1) * 0.5f)) * 1.5f;
                float offsetY = (y - ((GridWidth - 1) * 0.5f)) * 1.5f;
                float rotation = rotationAngle * (1.0f + (index * 0.1f));

                pointer[index] = new()
                {
                    Model = Matrix4x4.CreateScale(0.4f) * Matrix4x4.CreateRotationY(rotation) * Matrix4x4.CreateRotationX(rotation * 0.5f) * Matrix4x4.CreateTranslation(offsetX, offsetY, 0.0f),
                    Color = new((float)x / GridWidth, (float)y / GridWidth, 1.0f - ((float)x / GridWidth), 1.0f)
                };
            }

            instanceBuffer.Upload(0, new()
            {
                Pointer = (nint)pointer,
                SizeInBytes = (uint)(sizeof(Instance) * instances.Length)
            });
        }
    }

    public void Render(CommandBuffer commandBuffer, Texture drawable)
    {
        if (depthTexture is null)
        {
            depthTexture = App.Context.CreateTexture(TextureDesc.DepthStencilAttachment(DepthFormat, App.Width, App.Height, SampleCount.Count1));

            commandBuffer.Transition(depthTexture, default, TextureLayout.Undefined, TextureLayout.DepthStencilAttachment);
        }

        commandBuffer.BeginRenderPass([ColorAttachment.Clear(drawable, new(0.04f, 0.055f, 0.075f, 1.0f))], DepthStencilAttachment.Clear(depthTexture, 1.0f, 0));

        commandBuffer.SetPipeline(pipeline);
        commandBuffer.SetVertexBuffer(vertexBuffer, 0, 0);
        commandBuffer.SetIndexBuffer(indexBuffer, 0, IndexFormat.UInt32);
        commandBuffer.SetConstantBuffer(constantBuffer, 0);

        commandBuffer.DrawIndexedIndirect(indirectBuffer, 0, 1);

        commandBuffer.EndRenderPass();
    }

    public void Resize(uint width, uint height)
    {
        depthTexture?.Dispose();
        depthTexture = null;

        Constants constants = new()
        {
            View = Matrix4x4.CreateLookAt(new(0.0f, 0.0f, 8.0f), Vector3.Zero, Vector3.UnitY),
            Projection = Matrix4x4.CreatePerspectiveFieldOfView(float.DegreesToRadians(45.0f), (float)width / height, 0.1f, 100.0f),
            Instances = instanceBuffer.StorageReadOnlyHandle
        };

        constantBuffer.Upload(0, new()
        {
            Pointer = (nint)(&constants),
            SizeInBytes = (uint)sizeof(Constants)
        });
    }

    public void Dispose()
    {
        depthTexture?.Dispose();

        pipeline.Dispose();
        constantBuffer.Dispose();
        instanceBuffer.Dispose();
        indirectBuffer.Dispose();
        indexBuffer.Dispose();
        vertexBuffer.Dispose();
    }
}

[StructLayout(LayoutKind.Sequential)]
file struct Vertex(Vector3 position, Vector4 color)
{
    public Vector3 Position = position;

    public Vector4 Color = color;
}

[StructLayout(LayoutKind.Explicit, Size = 80)]
file struct Instance
{
    [FieldOffset(0)]
    public Matrix4x4 Model;

    [FieldOffset(64)]
    public Vector4 Color;
}

[StructLayout(LayoutKind.Explicit, Size = 256)]
file struct Constants
{
    [FieldOffset(0)]
    public Matrix4x4 View;

    [FieldOffset(64)]
    public Matrix4x4 Projection;

    [FieldOffset(128)]
    public ResourceHandle Instances;
}