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

        string shaderPath = App.ShaderPath("RayTracing.slang");

        Vector3[] floorVertices =
        [
            new(-50.0f, 0.0f, -50.0f),
            new(50.0f, 0.0f, -50.0f),
            new(50.0f, 0.0f, 50.0f),
            new(-50.0f, 0.0f, 50.0f)
        ];
        uint[] floorIndices = [0, 1, 2, 0, 2, 3];

        floorVertexBuffer = CreateStorageBuffer(floorVertices);
        floorIndexBuffer = CreateStorageBuffer(floorIndices);

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

        Vector3[] aabbs = new Vector3[spheres.Length * 2];
        for (int index = 0; index < spheres.Length; index++)
        {
            aabbs[index * 2] = spheres[index].Center - new Vector3(spheres[index].Radius);
            aabbs[(index * 2) + 1] = spheres[index].Center + new Vector3(spheres[index].Radius);
        }

        aabbBuffer = CreateStorageBuffer(aabbs, (uint)(sizeof(Vector3) * 2));
        sphereBuffer = CreateStorageBuffer(spheres);

        constantBuffer = App.Context.CreateBuffer(BufferDesc.Constant((uint)sizeof(Constants)));

        using Shader computeShader = App.Context.CreateShader(ZenithCompiler.CompileFromFile(App.Context.GraphicsApi, shaderPath, "CSMain"));
        using Shader vertexShader = App.Context.CreateShader(ZenithCompiler.CompileFromFile(App.Context.GraphicsApi, shaderPath, "VSMain"));
        using Shader fragmentShader = App.Context.CreateShader(ZenithCompiler.CompileFromFile(App.Context.GraphicsApi, shaderPath, "FSMain"));

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

        CommandBuffer buildCommands = App.Context.ComputeQueue.CommandBuffer();
        BottomLevelAccelerationStructureDesc floorDesc = new()
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
        };
        floorBlas = buildCommands.BuildAccelerationStructure(floorDesc);

        BottomLevelAccelerationStructureDesc sphereDesc = new()
        {
            Geometries =
            [
                RayTracingGeometry.Aabbs(new()
                {
                    Buffer = aabbBuffer,
                    Count = (uint)spheres.Length,
                    StrideInBytes = (uint)(sizeof(Vector3) * 2)
                }, true)
            ],
            BuildFlags = AccelerationStructureBuildFlags.PreferFastTrace
        };
        sphereBlas = buildCommands.BuildAccelerationStructure(sphereDesc);

        TopLevelAccelerationStructureDesc desc = new()
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
        };
        tlas = buildCommands.BuildAccelerationStructure(desc);

        buildCommands.Submit().Wait();
    }

    public TextureLayout RequiredLayout => TextureLayout.ColorAttachment;

    public void Update(double deltaTime)
    {
        totalTime += (float)deltaTime;
    }

    public void Render(CommandBuffer commandBuffer, Texture drawable)
    {
        TextureLayout outputLayout = TextureLayout.Sampled;
        if (outputTexture is null)
        {
            outputTexture = CreateOutputTexture(App.Width, App.Height);
            outputLayout = TextureLayout.Undefined;
        }

        float angle = totalTime * 0.3f;
        Constants constants = new()
        {
            Position = new(12.0f * MathF.Sin(angle),
                           4.0f + MathF.Sin(totalTime * 0.2f),
                           -12.0f * MathF.Cos(angle)),
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

        commandBuffer.Transition(outputTexture, default, outputLayout, TextureLayout.Storage);

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

    private static Buffer CreateStorageBuffer<T>(T[] data, uint strideInBytes = 0) where T : unmanaged
    {
        strideInBytes = strideInBytes is 0 ? (uint)sizeof(T) : strideInBytes;
        Buffer buffer = App.Context.CreateBuffer(BufferDesc.StorageReadOnly((uint)(sizeof(T) * data.Length), strideInBytes));

        fixed (T* pointer = data)
        {
            buffer.Upload(0, new()
            {
                Pointer = (nint)pointer,
                SizeInBytes = (uint)(sizeof(T) * data.Length)
            });
        }

        return buffer;
    }

    private static Texture CreateOutputTexture(uint width, uint height)
    {
        return App.Context.CreateTexture(new()
        {
            Type = TextureType.Texture2D,
            Format = PixelFormat.R32G32B32A32Float,
            Width = width,
            Height = height,
            Depth = 1,
            MipLevels = 1,
            ArrayLayers = 1,
            SampleCount = SampleCount.Count1,
            Usages = TextureUsages.Sampled | TextureUsages.Storage
        });
    }
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
