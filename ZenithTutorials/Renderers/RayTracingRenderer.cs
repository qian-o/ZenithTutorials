namespace ZenithTutorials.Renderers;

internal unsafe sealed class RayTracingRenderer : IRenderer
{
    private const uint ThreadGroupSize = 16;

    private readonly Buffer floorVertexBuffer;
    private readonly Buffer floorIndexBuffer;
    private readonly Buffer aabbBuffer;
    private readonly Buffer sphereBuffer;
    private readonly Buffer constantBuffer;
    private readonly BottomLevelAccelerationStructure floorBlas;
    private readonly BottomLevelAccelerationStructure sphereBlas;
    private readonly TopLevelAccelerationStructure tlas;
    private readonly ComputePipeline rayTracingPipeline;
    private readonly GraphicsPipeline displayPipeline;

    private Texture? outputTexture;
    private float totalTime;

    public RayTracingRenderer()
    {
        if (!App.Context.Capabilities.RayTracingSupported)
        {
            throw new PlatformNotSupportedException("Ray Tracing is not supported by the selected device.");
        }

        Vector3[] floorVertices =
        [
            new(-50.0f, 0.0f, -50.0f),
            new(50.0f, 0.0f, -50.0f),
            new(50.0f, 0.0f, 50.0f),
            new(-50.0f, 0.0f, 50.0f)
        ];

        uint[] floorIndices = [0, 1, 2, 0, 2, 3];

        Sphere[] spheres =
        [
            new()
            {
                Center = new(-2.0f, 1.0f, 1.0f),
                Radius = 1.0f,
                Color = new(0.8f, 0.2f, 0.2f)
            },
            new()
            {
                Center = new(2.0f, 1.2f, -1.0f),
                Radius = 1.2f,
                Color = new(0.2f, 0.4f, 0.8f)
            },
            new()
            {
                Center = new(0.0f, 0.6f, -3.0f),
                Radius = 0.6f,
                Color = new(0.9f, 0.7f, 0.2f)
            }
        ];

        Aabb[] aabbs = new Aabb[spheres.Length];
        for (int index = 0; index < spheres.Length; index++)
        {
            aabbs[index] = new(spheres[index].Center - new Vector3(spheres[index].Radius), spheres[index].Center + new Vector3(spheres[index].Radius));
        }

        floorVertexBuffer = App.LoadBuffer(floorVertices, BufferUsages.StorageReadOnly);
        floorIndexBuffer = App.LoadBuffer(floorIndices, BufferUsages.StorageReadOnly);
        aabbBuffer = App.LoadBuffer(aabbs, BufferUsages.StorageReadOnly);
        sphereBuffer = App.LoadBuffer(spheres, BufferUsages.StorageReadOnly);
        constantBuffer = App.LoadBuffer([new Constants()], BufferUsages.Constant);

        using Shader computeShader = App.LoadShader("RayTracing.slang", "CSMain");
        using Shader vertexShader = App.LoadShader("RayTracing.slang", "VSMain");
        using Shader fragmentShader = App.LoadShader("RayTracing.slang", "FSMain");

        rayTracingPipeline = App.Context.CreateComputePipeline(new() { ComputeShader = computeShader });
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

        CommandBuffer commandBuffer = App.Context.ComputeQueue.CommandBuffer();

        floorBlas = commandBuffer.BuildAccelerationStructure(new BottomLevelAccelerationStructureDesc()
        {
            Geometries =
            [
                RayTracingGeometry.Triangles(new()
                {
                    VertexBuffer = floorVertexBuffer,
                    VertexFormat = PixelFormat.R32G32B32Float,
                    VertexCount = (uint)floorVertices.Length,
                    VertexStrideInBytes = (uint)sizeof(Vector3),
                    IndexBuffer = floorIndexBuffer,
                    IndexFormat = IndexFormat.UInt32,
                    IndexCount = (uint)floorIndices.Length,
                    Transform = Matrix4x4.Identity
                }, true)
            ],
            BuildFlags = AccelerationStructureBuildFlags.PreferFastTrace
        });

        sphereBlas = commandBuffer.BuildAccelerationStructure(new BottomLevelAccelerationStructureDesc()
        {
            Geometries =
            [
                RayTracingGeometry.Aabbs(new()
                {
                    Buffer = aabbBuffer,
                    Count = (uint)spheres.Length,
                    StrideInBytes = aabbBuffer.Desc.StrideInBytes
                }, true)
            ],
            BuildFlags = AccelerationStructureBuildFlags.PreferFastTrace
        });

        tlas = commandBuffer.BuildAccelerationStructure(new TopLevelAccelerationStructureDesc()
        {
            Instances =
            [
                new()
                {
                    AccelerationStructure = floorBlas,
                    InstanceId = 0,
                    VisibilityMask = 0xFF,
                    Transform = Matrix4x4.Identity
                },
                new()
                {
                    AccelerationStructure = sphereBlas,
                    InstanceId = 1,
                    VisibilityMask = 0xFF,
                    Transform = Matrix4x4.Identity
                }
            ],
            BuildFlags = AccelerationStructureBuildFlags.PreferFastTrace
        });

        commandBuffer.Submit().Wait();
    }

    public TextureLayout RequiredLayout => TextureLayout.ColorAttachment;

    public void Update(double deltaTime)
    {
        totalTime += (float)deltaTime;
    }

    public void Render(CommandBuffer commandBuffer, Texture drawable)
    {
        if (outputTexture is null)
        {
            outputTexture = App.Context.CreateTexture(new()
            {
                Type = TextureType.Texture2D,
                Format = PixelFormat.R32G32B32A32Float,
                Width = App.Width,
                Height = App.Height,
                Depth = 1,
                MipLevels = 1,
                ArrayLayers = 1,
                SampleCount = SampleCount.Count1,
                Usages = TextureUsages.Sampled | TextureUsages.Storage
            });

            commandBuffer.Transition(outputTexture, default, TextureLayout.Undefined, TextureLayout.Sampled);
        }

        Constants constants = new()
        {
            Position = new(12.0f * MathF.Sin(totalTime * 0.3f), 4.0f + MathF.Sin(totalTime * 0.2f), -12.0f * MathF.Cos(totalTime * 0.3f)),
            Scene = tlas.Handle,
            Spheres = sphereBuffer.StorageReadOnlyHandle,
            OutputTexture = outputTexture.StorageHandle,
            Image = outputTexture.SampledHandle,
            Sampler = App.LinearClampSampler.Handle
        };

        constantBuffer.Upload(0, new()
        {
            Pointer = (nint)(&constants),
            SizeInBytes = (uint)sizeof(Constants)
        });

        commandBuffer.Transition(outputTexture, default, TextureLayout.Sampled, TextureLayout.Storage);

        commandBuffer.SetPipeline(rayTracingPipeline);
        commandBuffer.SetConstantBuffer(constantBuffer, 0);
        commandBuffer.Dispatch((App.Width + ThreadGroupSize - 1) / ThreadGroupSize, (App.Height + ThreadGroupSize - 1) / ThreadGroupSize, 1);

        commandBuffer.Transition(outputTexture, default, TextureLayout.Storage, TextureLayout.Sampled);

        commandBuffer.BeginRenderPass([ColorAttachment.DontCare(drawable)], null);

        commandBuffer.SetPipeline(displayPipeline);
        commandBuffer.SetConstantBuffer(constantBuffer, 0);
        commandBuffer.Draw(3, 1, 0, 0);

        commandBuffer.EndRenderPass();
    }

    public void Resize(uint width, uint height)
    {
        outputTexture?.Dispose();
        outputTexture = null;
    }

    public void Dispose()
    {
        outputTexture?.Dispose();

        displayPipeline.Dispose();
        rayTracingPipeline.Dispose();
        constantBuffer.Dispose();
        tlas.Dispose();
        sphereBlas.Dispose();
        floorBlas.Dispose();
        sphereBuffer.Dispose();
        aabbBuffer.Dispose();
        floorIndexBuffer.Dispose();
        floorVertexBuffer.Dispose();
    }
}

[StructLayout(LayoutKind.Sequential)]
file struct Aabb(Vector3 min, Vector3 max)
{
    public Vector3 Min = min;

    public Vector3 Max = max;
}

[StructLayout(LayoutKind.Explicit, Size = 256)]
file struct Constants
{
    [FieldOffset(0)]
    public Vector3 Position;

    [FieldOffset(16)]
    public ResourceHandle Scene;

    [FieldOffset(24)]
    public ResourceHandle Spheres;

    [FieldOffset(32)]
    public ResourceHandle OutputTexture;

    [FieldOffset(40)]
    public ResourceHandle Image;

    [FieldOffset(48)]
    public ResourceHandle Sampler;
}

[StructLayout(LayoutKind.Explicit, Size = 32)]
file struct Sphere
{
    [FieldOffset(0)]
    public Vector3 Center;

    [FieldOffset(12)]
    public float Radius;

    [FieldOffset(16)]
    public Vector3 Color;
}
