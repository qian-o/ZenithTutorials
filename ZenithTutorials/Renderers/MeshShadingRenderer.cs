namespace ZenithTutorials.Renderers;

internal unsafe sealed class MeshShadingRenderer : IRenderer
{
    private const uint TaskGroupSize = 32;
    private const uint MeshGroupSize = 120;
    private const uint GridSize = 10;
    private const uint TotalInstances = GridSize * GridSize * GridSize;
    private const uint DispatchGroupCount = (TotalInstances + TaskGroupSize - 1) / TaskGroupSize;
    private const PixelFormat DepthFormat = PixelFormat.D32FloatS8UInt;

    private readonly Buffer vertexBuffer;
    private readonly Buffer triangleBuffer;
    private readonly Buffer constantBuffer;
    private readonly MeshShadingPipeline pipeline;

    private Texture? depthTexture;
    private float totalTime;

    public MeshShadingRenderer()
    {
        if (!App.Context.Capabilities.MeshShadingSupported)
        {
            throw new PlatformNotSupportedException("Mesh Shading is not supported by the selected device.");
        }

        string shaderPath = App.ShaderPath("MeshShading.slang");

        const int longitudeSegments = 12;
        const int latitudeSegments = 6;
        const float radius = 0.5f;

        List<Vertex> sphereVertices = [];
        List<Triangle> sphereTriangles = [];

        sphereVertices.Add(new()
        {
            Position = new(0.0f, radius, 0.0f),
            Normal = Vector3.UnitY
        });

        for (int latitude = 1; latitude < latitudeSegments; latitude++)
        {
            float phi = MathF.PI * latitude / latitudeSegments;
            float sinPhi = MathF.Sin(phi);
            float cosPhi = MathF.Cos(phi);

            for (int longitude = 0; longitude < longitudeSegments; longitude++)
            {
                float theta = 2.0f * MathF.PI * longitude / longitudeSegments;
                Vector3 normal = new(sinPhi * MathF.Cos(theta),
                                     cosPhi,
                                     sinPhi * MathF.Sin(theta));
                sphereVertices.Add(new()
                {
                    Position = normal * radius,
                    Normal = normal
                });
            }
        }

        sphereVertices.Add(new()
        {
            Position = new(0.0f, -radius, 0.0f),
            Normal = -Vector3.UnitY
        });

        for (int longitude = 0; longitude < longitudeSegments; longitude++)
        {
            uint next = (uint)((longitude + 1) % longitudeSegments);
            sphereTriangles.Add(new()
            {
                Index0 = 0,
                Index1 = (uint)(1 + longitude),
                Index2 = 1 + next
            });
        }

        for (int latitude = 0; latitude < latitudeSegments - 2; latitude++)
        {
            for (int longitude = 0; longitude < longitudeSegments; longitude++)
            {
                uint next = (uint)((longitude + 1) % longitudeSegments);
                uint topLeft = (uint)(1 + (latitude * longitudeSegments) + longitude);
                uint topRight = (uint)(1 + (latitude * longitudeSegments)) + next;
                uint bottomLeft = (uint)(1 + ((latitude + 1) * longitudeSegments) + longitude);
                uint bottomRight = (uint)(1 + ((latitude + 1) * longitudeSegments)) + next;
                sphereTriangles.Add(new()
                {
                    Index0 = topLeft,
                    Index1 = bottomLeft,
                    Index2 = topRight
                });
                sphereTriangles.Add(new()
                {
                    Index0 = topRight,
                    Index1 = bottomLeft,
                    Index2 = bottomRight
                });
            }
        }

        uint bottomPole = (uint)(sphereVertices.Count - 1);
        uint lastRing = 1 + ((latitudeSegments - 2) * longitudeSegments);
        for (int longitude = 0; longitude < longitudeSegments; longitude++)
        {
            uint next = (uint)((longitude + 1) % longitudeSegments);
            sphereTriangles.Add(new()
            {
                Index0 = bottomPole,
                Index1 = lastRing + next,
                Index2 = lastRing + (uint)longitude
            });
        }

        vertexBuffer = CreateStorageBuffer<Vertex>([.. sphereVertices]);
        triangleBuffer = CreateStorageBuffer<Triangle>([.. sphereTriangles]);

        constantBuffer = App.Context.CreateBuffer(BufferDesc.Constant((uint)sizeof(Constants)));

        ShaderDesc taskDesc = ZenithCompiler.CompileFromFile(App.Context.GraphicsApi, shaderPath, "ASMain");
        taskDesc.ThreadGroupSize = new()
        {
            X = TaskGroupSize,
            Y = 1,
            Z = 1
        };

        ShaderDesc meshDesc = ZenithCompiler.CompileFromFile(App.Context.GraphicsApi, shaderPath, "MSMain");
        meshDesc.ThreadGroupSize = new()
        {
            X = MeshGroupSize,
            Y = 1,
            Z = 1
        };

        using Shader taskShader = App.Context.CreateShader(taskDesc);
        using Shader meshShader = App.Context.CreateShader(meshDesc);
        using Shader fragmentShader = App.Context.CreateShader(ZenithCompiler.CompileFromFile(App.Context.GraphicsApi, shaderPath, "FSMain"));

        pipeline = App.Context.CreateMeshShadingPipeline(new()
        {
            TaskShader = taskShader,
            MeshShader = meshShader,
            FragmentShader = fragmentShader,
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

        Update(0.0);
    }

    public TextureLayout RequiredLayout => TextureLayout.ColorAttachment;

    public void Update(double deltaTime)
    {
        totalTime += (float)deltaTime;
        float angle = totalTime * 0.3f;
        Vector3 cameraPosition = new(35.0f * MathF.Sin(angle),
                                     20.0f * MathF.Sin(totalTime * 0.2f),
                                     35.0f * MathF.Cos(angle));
        Matrix4x4 view = Matrix4x4.CreateLookAt(cameraPosition, Vector3.Zero, Vector3.UnitY);
        Matrix4x4 projection = Matrix4x4.CreatePerspectiveFieldOfView(float.DegreesToRadians(45.0f), (float)App.Width / App.Height, 0.1f, 200.0f);
        Matrix4x4 viewProjection = view * projection;

        Constants constants = new()
        {
            ViewProjection = viewProjection,
            FrustumPlane0 = NormalizePlane(new(viewProjection.M11 + viewProjection.M14,
                                               viewProjection.M21 + viewProjection.M24,
                                               viewProjection.M31 + viewProjection.M34,
                                               viewProjection.M41 + viewProjection.M44)),
            FrustumPlane1 = NormalizePlane(new(viewProjection.M14 - viewProjection.M11,
                                               viewProjection.M24 - viewProjection.M21,
                                               viewProjection.M34 - viewProjection.M31,
                                               viewProjection.M44 - viewProjection.M41)),
            FrustumPlane2 = NormalizePlane(new(viewProjection.M12 + viewProjection.M14,
                                               viewProjection.M22 + viewProjection.M24,
                                               viewProjection.M32 + viewProjection.M34,
                                               viewProjection.M42 + viewProjection.M44)),
            FrustumPlane3 = NormalizePlane(new(viewProjection.M14 - viewProjection.M12,
                                               viewProjection.M24 - viewProjection.M22,
                                               viewProjection.M34 - viewProjection.M32,
                                               viewProjection.M44 - viewProjection.M42)),
            FrustumPlane4 = NormalizePlane(new(viewProjection.M13,
                                               viewProjection.M23,
                                               viewProjection.M33,
                                               viewProjection.M43)),
            FrustumPlane5 = NormalizePlane(new(viewProjection.M14 - viewProjection.M13,
                                               viewProjection.M24 - viewProjection.M23,
                                               viewProjection.M34 - viewProjection.M33,
                                               viewProjection.M44 - viewProjection.M43)),
            Time = totalTime,
            LightDirection = -Vector3.Normalize(cameraPosition),
            Vertices = vertexBuffer.StorageReadOnlyHandle,
            Triangles = triangleBuffer.StorageReadOnlyHandle
        };

        constantBuffer.Upload(0, new()
        {
            Pointer = (nint)(&constants),
            SizeInBytes = (uint)sizeof(Constants)
        });
    }

    public void Render(CommandBuffer commandBuffer, Texture drawable)
    {
        if (depthTexture is null)
        {
            depthTexture = App.Context.CreateTexture(TextureDesc.DepthStencilAttachment(DepthFormat, App.Width, App.Height, SampleCount.Count1));
            commandBuffer.Transition(depthTexture, default, TextureLayout.Undefined, TextureLayout.DepthStencilAttachment);
        }

        commandBuffer.BeginRenderPass([ColorAttachment.Clear(drawable, new(0.05f, 0.05f, 0.08f, 1.0f))], DepthStencilAttachment.Clear(depthTexture, 1.0f, 0));

        commandBuffer.SetPipeline(pipeline);
        commandBuffer.SetConstantBuffer(constantBuffer, 0);

        commandBuffer.DispatchMesh(DispatchGroupCount, 1, 1);

        commandBuffer.EndRenderPass();
    }

    public void Resize(uint width, uint height)
    {
        depthTexture?.Dispose();
        depthTexture = null;
    }

    public void Dispose()
    {
        depthTexture?.Dispose();

        pipeline.Dispose();
        constantBuffer.Dispose();
        triangleBuffer.Dispose();
        vertexBuffer.Dispose();
    }

    private static Buffer CreateStorageBuffer<T>(T[] data) where T : unmanaged
    {
        Buffer buffer = App.Context.CreateBuffer(BufferDesc.StorageReadOnly((uint)(sizeof(T) * data.Length), (uint)sizeof(T)));

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

    private static Vector4 NormalizePlane(Vector4 plane)
    {
        return plane / new Vector3(plane.X, plane.Y, plane.Z).Length();
    }
}

[StructLayout(LayoutKind.Explicit, Size = 32)]
file struct Vertex
{
    [FieldOffset(0)]
    public Vector3 Position;

    [FieldOffset(16)]
    public Vector3 Normal;
}

[StructLayout(LayoutKind.Explicit, Size = 16)]
file struct Triangle
{
    [FieldOffset(0)]
    public uint Index0;

    [FieldOffset(4)]
    public uint Index1;

    [FieldOffset(8)]
    public uint Index2;
}

[StructLayout(LayoutKind.Explicit, Size = 256)]
file struct Constants
{
    [FieldOffset(0)]
    public Matrix4x4 ViewProjection;

    [FieldOffset(64)]
    public Vector4 FrustumPlane0;

    [FieldOffset(80)]
    public Vector4 FrustumPlane1;

    [FieldOffset(96)]
    public Vector4 FrustumPlane2;

    [FieldOffset(112)]
    public Vector4 FrustumPlane3;

    [FieldOffset(128)]
    public Vector4 FrustumPlane4;

    [FieldOffset(144)]
    public Vector4 FrustumPlane5;

    [FieldOffset(160)]
    public float Time;

    [FieldOffset(164)]
    public Vector3 LightDirection;

    [FieldOffset(176)]
    public ResourceHandle Vertices;

    [FieldOffset(184)]
    public ResourceHandle Triangles;
}
