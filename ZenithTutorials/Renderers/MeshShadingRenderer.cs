namespace ZenithTutorials.Renderers;

internal unsafe class MeshShadingRenderer : IRenderer
{
    private const uint MaxPrimitives = 128;

    private const string ShaderSource = """
        static const uint MaxVertices = 64;
        static const uint MaxPrimitives = 128;

        struct Vertex
        {
            private float4 PositionAndPadding;

            private float4 NormalAndPadding;

            float2 TexCoord;

            private float padding0;

            private float padding1;

            property float3 Position { get { return PositionAndPadding.xyz; } }

            property float3 Normal { get { return NormalAndPadding.xyz; } }
        };

        struct Meshlet
        {
            uint VertexOffset;

            uint VertexCount;

            uint PrimitiveOffset;

            uint PrimitiveCount;
        };

        struct Triangle
        {
            private uint4 IndicesAndPadding;

            property uint3 Indices { get { return IndicesAndPadding.xyz; } }
        };

        struct TransformConstants
        {
            float4x4 MVP;
        };

        struct VertexOutput
        {
            float4 Position : SV_Position;

            float3 Normal : NORMAL;

            float2 TexCoord : TEXCOORD0;
        };

        ConstantBuffer<TransformConstants> transform;
        StructuredBuffer<Vertex> vertices;
        StructuredBuffer<Triangle> indices;
        StructuredBuffer<Meshlet> meshlets;

        [shader("mesh")]
        [numthreads(MaxPrimitives, 1, 1)]
        [outputtopology("triangle")]
        void MSMain(in uint groupID : SV_GroupID,
                    in uint groupThreadID : SV_GroupThreadID,
                    OutputVertices<VertexOutput, MaxVertices> outVertices,
                    OutputIndices<uint3, MaxPrimitives> outIndices)
        {
            Meshlet meshlet = meshlets[groupID];

            SetMeshOutputCounts(meshlet.VertexCount, meshlet.PrimitiveCount);

            if (groupThreadID < meshlet.VertexCount)
            {
                Vertex vertex = vertices[meshlet.VertexOffset + groupThreadID];

                VertexOutput output;
                output.Position = mul(float4(vertex.Position, 1.0), transform.MVP);
                output.Normal = vertex.Normal;
                output.TexCoord = vertex.TexCoord;

                outVertices[groupThreadID] = output;
            }

            if (groupThreadID < meshlet.PrimitiveCount)
            {
                outIndices[groupThreadID] = indices[meshlet.PrimitiveOffset + groupThreadID].Indices;
            }
        }

        [shader("pixel")]
        float4 PSMain(VertexOutput input) : SV_Target
        {
            float3 lightDir = normalize(float3(1.0, 1.0, -1.0));
            float3 normal = normalize(input.Normal);
            float ndotl = max(dot(normal, lightDir), 0.0);

            float3 viewDir = normalize(-input.Position.xyz);
            float3 halfDir = normalize(lightDir + viewDir);
            float spec = pow(max(dot(normal, halfDir), 0.0), 32.0);

            float3 baseColor = float3(input.TexCoord, 0.5);

            float3 ambient = baseColor * 0.2;
            float3 diffuse = baseColor * ndotl * 0.8;
            float3 specular = float3(1.0, 1.0, 1.0) * spec * 0.5;

            return float4(ambient + diffuse + specular, 1.0);
        }
        """;

    private readonly uint meshletCount;
    private readonly Buffer vertexBuffer;
    private readonly Buffer indexBuffer;
    private readonly Buffer meshletBuffer;
    private readonly Buffer constantBuffer;
    private readonly ResourceLayout resourceLayout;
    private readonly ResourceTable resourceTable;
    private readonly MeshShadingPipeline pipeline;

    private float rotationAngle;

    public MeshShadingRenderer()
    {
        if (!App.Context.Capabilities.MeshShadingSupported)
        {
            throw new NotSupportedException("Mesh shading is not supported on this device.");
        }

        Vertex[] cubeVertices =
        [
            new() { Position = new(-0.5f, -0.5f,  0.5f), Normal = new( 0,  0,  1), TexCoord = new(0, 1) },
            new() { Position = new( 0.5f, -0.5f,  0.5f), Normal = new( 0,  0,  1), TexCoord = new(1, 1) },
            new() { Position = new( 0.5f,  0.5f,  0.5f), Normal = new( 0,  0,  1), TexCoord = new(1, 0) },
            new() { Position = new(-0.5f,  0.5f,  0.5f), Normal = new( 0,  0,  1), TexCoord = new(0, 0) },

            new() { Position = new( 0.5f, -0.5f, -0.5f), Normal = new( 0,  0, -1), TexCoord = new(0, 1) },
            new() { Position = new(-0.5f, -0.5f, -0.5f), Normal = new( 0,  0, -1), TexCoord = new(1, 1) },
            new() { Position = new(-0.5f,  0.5f, -0.5f), Normal = new( 0,  0, -1), TexCoord = new(1, 0) },
            new() { Position = new( 0.5f,  0.5f, -0.5f), Normal = new( 0,  0, -1), TexCoord = new(0, 0) },

            new() { Position = new(-0.5f, -0.5f, -0.5f), Normal = new(-1,  0,  0), TexCoord = new(0, 1) },
            new() { Position = new(-0.5f, -0.5f,  0.5f), Normal = new(-1,  0,  0), TexCoord = new(1, 1) },
            new() { Position = new(-0.5f,  0.5f,  0.5f), Normal = new(-1,  0,  0), TexCoord = new(1, 0) },
            new() { Position = new(-0.5f,  0.5f, -0.5f), Normal = new(-1,  0,  0), TexCoord = new(0, 0) },

            new() { Position = new( 0.5f, -0.5f,  0.5f), Normal = new( 1,  0,  0), TexCoord = new(0, 1) },
            new() { Position = new( 0.5f, -0.5f, -0.5f), Normal = new( 1,  0,  0), TexCoord = new(1, 1) },
            new() { Position = new( 0.5f,  0.5f, -0.5f), Normal = new( 1,  0,  0), TexCoord = new(1, 0) },
            new() { Position = new( 0.5f,  0.5f,  0.5f), Normal = new( 1,  0,  0), TexCoord = new(0, 0) },

            new() { Position = new(-0.5f,  0.5f,  0.5f), Normal = new( 0,  1,  0), TexCoord = new(0, 1) },
            new() { Position = new( 0.5f,  0.5f,  0.5f), Normal = new( 0,  1,  0), TexCoord = new(1, 1) },
            new() { Position = new( 0.5f,  0.5f, -0.5f), Normal = new( 0,  1,  0), TexCoord = new(1, 0) },
            new() { Position = new(-0.5f,  0.5f, -0.5f), Normal = new( 0,  1,  0), TexCoord = new(0, 0) },

            new() { Position = new(-0.5f, -0.5f, -0.5f), Normal = new( 0, -1,  0), TexCoord = new(0, 1) },
            new() { Position = new( 0.5f, -0.5f, -0.5f), Normal = new( 0, -1,  0), TexCoord = new(1, 1) },
            new() { Position = new( 0.5f, -0.5f,  0.5f), Normal = new( 0, -1,  0), TexCoord = new(1, 0) },
            new() { Position = new(-0.5f, -0.5f,  0.5f), Normal = new( 0, -1,  0), TexCoord = new(0, 0) }
        ];

        Triangle[] cubeTriangles =
        [
            new() { I0 = 0, I1 = 1, I2 = 2 },
            new() { I0 = 0, I1 = 2, I2 = 3 },
            new() { I0 = 4, I1 = 5, I2 = 6 },
            new() { I0 = 4, I1 = 6, I2 = 7 },
            new() { I0 = 8, I1 = 9, I2 = 10 },
            new() { I0 = 8, I1 = 10, I2 = 11 },
            new() { I0 = 12, I1 = 13, I2 = 14 },
            new() { I0 = 12, I1 = 14, I2 = 15 },
            new() { I0 = 16, I1 = 17, I2 = 18 },
            new() { I0 = 16, I1 = 18, I2 = 19 },
            new() { I0 = 20, I1 = 21, I2 = 22 },
            new() { I0 = 20, I1 = 22, I2 = 23 }
        ];

        Meshlet[] meshlets =
        [
            new()
            {
                VertexOffset = 0,
                VertexCount = (uint)cubeVertices.Length,
                PrimitiveOffset = 0,
                PrimitiveCount = (uint)cubeTriangles.Length
            }
        ];
        meshletCount = (uint)meshlets.Length;

        vertexBuffer = App.Context.CreateBuffer(new()
        {
            SizeInBytes = (uint)(sizeof(Vertex) * cubeVertices.Length),
            StrideInBytes = (uint)sizeof(Vertex),
            Flags = BufferUsageFlags.ShaderResource
        });
        vertexBuffer.Upload(cubeVertices, 0);

        indexBuffer = App.Context.CreateBuffer(new()
        {
            SizeInBytes = (uint)(sizeof(Triangle) * cubeTriangles.Length),
            StrideInBytes = (uint)sizeof(Triangle),
            Flags = BufferUsageFlags.ShaderResource
        });
        indexBuffer.Upload(cubeTriangles, 0);

        meshletBuffer = App.Context.CreateBuffer(new()
        {
            SizeInBytes = (uint)(sizeof(Meshlet) * meshlets.Length),
            StrideInBytes = (uint)sizeof(Meshlet),
            Flags = BufferUsageFlags.ShaderResource
        });
        meshletBuffer.Upload(meshlets, 0);

        constantBuffer = App.Context.CreateBuffer(new()
        {
            SizeInBytes = (uint)sizeof(TransformConstants),
            StrideInBytes = (uint)sizeof(TransformConstants),
            Flags = BufferUsageFlags.Constant | BufferUsageFlags.MapWrite
        });

        resourceLayout = App.Context.CreateResourceLayout(new()
        {
            Bindings = BindingHelper.Bindings
            (
                new() { Type = ResourceType.ConstantBuffer, Count = 1, StageFlags = ShaderStageFlags.Mesh },
                new() { Type = ResourceType.StructuredBuffer, Count = 1, StageFlags = ShaderStageFlags.Mesh },
                new() { Type = ResourceType.StructuredBuffer, Count = 1, StageFlags = ShaderStageFlags.Mesh },
                new() { Type = ResourceType.StructuredBuffer, Count = 1, StageFlags = ShaderStageFlags.Mesh }
            )
        });

        resourceTable = App.Context.CreateResourceTable(new()
        {
            Layout = resourceLayout,
            Resources = [constantBuffer, vertexBuffer, indexBuffer, meshletBuffer]
        });

        using Shader meshShader = App.Context.LoadShaderFromSource(ShaderSource, "MSMain", ShaderStageFlags.Mesh);
        using Shader pixelShader = App.Context.LoadShaderFromSource(ShaderSource, "PSMain", ShaderStageFlags.Pixel);

        pipeline = App.Context.CreateMeshShadingPipeline(new()
        {
            RenderStates = new()
            {
                RasterizerState = RasterizerStates.CullBack,
                DepthStencilState = DepthStencilStates.Default,
                BlendState = BlendStates.Opaque
            },
            Amplification = null,
            Mesh = meshShader,
            Pixel = pixelShader,
            ResourceLayout = resourceLayout,
            PrimitiveTopology = PrimitiveTopology.TriangleList,
            Output = App.FrameBuffer.Output,
            MeshThreadGroupSizeX = MaxPrimitives,
            MeshThreadGroupSizeY = 1,
            MeshThreadGroupSizeZ = 1
        });
    }

    public void Update(double deltaTime)
    {
        rotationAngle += (float)deltaTime;

        Matrix4x4 model = Matrix4x4.CreateRotationY(rotationAngle) * Matrix4x4.CreateRotationX(rotationAngle * 0.5f);
        Matrix4x4 view = Matrix4x4.CreateLookAt(new(0, 0, 3), Vector3.Zero, Vector3.UnitY);
        Matrix4x4 projection = Matrix4x4.CreatePerspectiveFieldOfView(float.DegreesToRadians(45.0f), (float)App.Width / App.Height, 0.1f, 100.0f);

        constantBuffer.Upload([new TransformConstants() { MVP = model * view * projection }], 0);
    }

    public void Render()
    {
        CommandBuffer commandBuffer = App.Context.Graphics.CommandBuffer();

        commandBuffer.BeginRenderPass(App.FrameBuffer, new()
        {
            ColorValues = [new(0.1f, 0.1f, 0.1f, 1.0f)],
            Depth = 1.0f,
            Stencil = 0,
            Flags = ClearFlags.All
        }, resourceTable);

        commandBuffer.SetPipeline(pipeline);
        commandBuffer.SetResourceTable(resourceTable);
        commandBuffer.DispatchMesh(meshletCount, 1, 1);

        commandBuffer.EndRenderPass();

        commandBuffer.Submit(waitForCompletion: true);
    }

    public void Resize(uint width, uint height)
    {
    }

    public void Dispose()
    {
        pipeline.Dispose();
        resourceTable.Dispose();
        resourceLayout.Dispose();
        constantBuffer.Dispose();
        meshletBuffer.Dispose();
        indexBuffer.Dispose();
        vertexBuffer.Dispose();
    }
}

[StructLayout(LayoutKind.Explicit, Size = 48)]
file struct Vertex
{
    [FieldOffset(0)]
    public Vector3 Position;

    [FieldOffset(16)]
    public Vector3 Normal;

    [FieldOffset(32)]
    public Vector2 TexCoord;
}

[StructLayout(LayoutKind.Explicit, Size = 16)]
file struct Triangle
{
    [FieldOffset(0)]
    public uint I0;

    [FieldOffset(4)]
    public uint I1;

    [FieldOffset(8)]
    public uint I2;
}

[StructLayout(LayoutKind.Explicit, Size = 16)]
file struct Meshlet
{
    [FieldOffset(0)]
    public uint VertexOffset;

    [FieldOffset(4)]
    public uint VertexCount;

    [FieldOffset(8)]
    public uint PrimitiveOffset;

    [FieldOffset(12)]
    public uint PrimitiveCount;
}

[StructLayout(LayoutKind.Explicit, Size = 64)]
file struct TransformConstants
{
    [FieldOffset(0)]
    public Matrix4x4 MVP;
}
