namespace ZenithTutorials.Renderers;

internal unsafe class MeshShadingRenderer : IRenderer
{
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
                Vector3 normal = new(sinPhi * MathF.Cos(theta), cosPhi, sinPhi * MathF.Sin(theta));
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

        vertexBuffer = App.LoadBuffer([.. sphereVertices], BufferUsages.StorageReadOnly);
        triangleBuffer = App.LoadBuffer([.. sphereTriangles], BufferUsages.StorageReadOnly);
        constantBuffer = App.LoadBuffer([new Constants()], BufferUsages.Constant);

        using Shader taskShader = App.LoadShader("MeshShading.slang", "ASMain");
        using Shader meshShader = App.LoadShader("MeshShading.slang", "MSMain");
        using Shader fragmentShader = App.LoadShader("MeshShading.slang", "FSMain");

        pipeline = App.Context.CreateMeshShadingPipeline(new()
        {
            TaskShader = taskShader,
            MeshShader = meshShader,
            FragmentShader = fragmentShader,
            PrimitiveTopology = PrimitiveTopology.TriangleList,
            AttachmentFormats = new()
            {
                ColorFormats = [App.ColorFormat],
                DepthStencilFormat = PixelFormat.D32FloatS8UInt,
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

    public void Update(double deltaTime)
    {
        totalTime += (float)deltaTime;

        float angle = totalTime * 0.3f;
        Vector3 cameraPosition = new(35.0f * MathF.Sin(angle), 20.0f * MathF.Sin(totalTime * 0.2f), 35.0f * MathF.Cos(angle));

        Matrix4x4 view = Matrix4x4.CreateLookAt(cameraPosition, Vector3.Zero, Vector3.UnitY);
        Matrix4x4 projection = Matrix4x4.CreatePerspectiveFieldOfView(float.DegreesToRadians(45.0f), (float)App.Width / App.Height, 0.1f, 200.0f);
        Matrix4x4 viewProjection = view * projection;

        Constants constants = new()
        {
            ViewProjection = viewProjection,
            FrustumPlanes = new(viewProjection),
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
            depthTexture = App.Context.CreateTexture(TextureDesc.DepthStencilAttachment(PixelFormat.D32FloatS8UInt, App.Width, App.Height, SampleCount.Count1));

            commandBuffer.Transition(depthTexture, default, TextureLayout.Undefined, TextureLayout.DepthStencilAttachment);
        }

        commandBuffer.BeginRenderPass([ColorAttachment.Clear(drawable, new(0.05f, 0.05f, 0.08f, 1.0f))], DepthStencilAttachment.Clear(depthTexture, 1.0f, 0));

        commandBuffer.SetPipeline(pipeline);
        commandBuffer.SetConstantBuffer(constantBuffer, 0);

        commandBuffer.DispatchMesh(32, 1, 1);

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

[InlineArray(6)]
file struct FrustumPlanes
{
    private Vector4 element0;

    public FrustumPlanes(Matrix4x4 matrix)
    {
        Span<Vector4> planes = this;
        planes[0] = NormalizePlane(new(matrix.M11 + matrix.M14, matrix.M21 + matrix.M24, matrix.M31 + matrix.M34, matrix.M41 + matrix.M44));
        planes[1] = NormalizePlane(new(matrix.M14 - matrix.M11, matrix.M24 - matrix.M21, matrix.M34 - matrix.M31, matrix.M44 - matrix.M41));
        planes[2] = NormalizePlane(new(matrix.M12 + matrix.M14, matrix.M22 + matrix.M24, matrix.M32 + matrix.M34, matrix.M42 + matrix.M44));
        planes[3] = NormalizePlane(new(matrix.M14 - matrix.M12, matrix.M24 - matrix.M22, matrix.M34 - matrix.M32, matrix.M44 - matrix.M42));
        planes[4] = NormalizePlane(new(matrix.M13, matrix.M23, matrix.M33, matrix.M43));
        planes[5] = NormalizePlane(new(matrix.M14 - matrix.M13, matrix.M24 - matrix.M23, matrix.M34 - matrix.M33, matrix.M44 - matrix.M43));
    }

    private static Vector4 NormalizePlane(Vector4 plane)
    {
        return plane / new Vector3(plane.X, plane.Y, plane.Z).Length();
    }
}

[StructLayout(LayoutKind.Explicit, Size = 256)]
file struct Constants
{
    [FieldOffset(0)]
    public Matrix4x4 ViewProjection;

    [FieldOffset(64)]
    public FrustumPlanes FrustumPlanes;

    [FieldOffset(160)]
    public Vector3 LightDirection;

    [FieldOffset(176)]
    public ResourceHandle Vertices;

    [FieldOffset(184)]
    public ResourceHandle Triangles;
}
