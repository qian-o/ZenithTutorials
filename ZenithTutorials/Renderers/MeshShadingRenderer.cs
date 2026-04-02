namespace ZenithTutorials.Renderers;

internal unsafe class MeshShadingRenderer : IRenderer
{
    private const uint ASGroupSize = 32;
    private const uint MeshGroupSize = 120;
    private const uint GridSize = 10;
    private const uint TotalInstances = GridSize * GridSize * GridSize;
    private const uint DispatchGroupCount = (TotalInstances + ASGroupSize - 1) / ASGroupSize;

    private const string ShaderSource = """
        static const uint GridSize = 10;
        static const uint TotalInstances = GridSize * GridSize * GridSize;
        static const float InstanceSpacing = 2.5;
        static const uint ASGroupSize = 32;
        static const float BoundingSphereRadius = 0.5;

        static const uint SphereVertexCount = 62;
        static const uint SphereTriangleCount = 120;
        static const float GridOffset = float(GridSize - 1) * 0.5 * InstanceSpacing;

        struct Vertex
        {
            private float4 PositionAndPadding;

            private float4 NormalAndPadding;

            property float3 Position { get { return PositionAndPadding.xyz; } }

            property float3 Normal { get { return NormalAndPadding.xyz; } }
        };

        struct Triangle
        {
            private uint4 IndicesAndPadding;

            property uint3 Indices { get { return IndicesAndPadding.xyz; } }
        };

        struct SceneConstants
        {
            float4x4 ViewProjection;

            float4 FrustumPlanes[6];

            float Time;

            float3 LightDirection;
        };

        struct VertexOutput
        {
            float4 Position : SV_Position;

            float3 WorldNormal : NORMAL;

            float3 Color : COLOR;
        };

        struct Payload
        {
            uint InstanceIndices[ASGroupSize];
        };

        ConstantBuffer<SceneConstants> scene;
        StructuredBuffer<Vertex> vertices;
        StructuredBuffer<Triangle> indices;

        void DecomposeInstanceID(uint id, out uint x, out uint y, out uint z)
        {
            x = id % GridSize;
            y = (id / GridSize) % GridSize;
            z = id / (GridSize * GridSize);
        }

        float3 InstancePosition(uint id)
        {
            uint x, y, z;
            DecomposeInstanceID(id, x, y, z);
            return float3(x, y, z) * InstanceSpacing - GridOffset;
        }

        float3 InstanceColor(uint id)
        {
            uint x, y, z;
            DecomposeInstanceID(id, x, y, z);
            return float3(x, y, z) / float(GridSize - 1);
        }

        bool IsFrustumCulled(float3 center, float radius)
        {
            for (uint i = 0; i < 6; i++)
            {
                float4 plane = scene.FrustumPlanes[i];
                if (dot(plane.xyz, center) + plane.w < -radius)
                    return true;
            }
            return false;
        }

        groupshared Payload s_payload;
        groupshared uint s_visibleCount;

        [shader("amplification")]
        [numthreads(ASGroupSize, 1, 1)]
        void ASMain(uint groupID : SV_GroupID, uint groupThreadID : SV_GroupThreadID)
        {
            uint instanceIndex = groupID * ASGroupSize + groupThreadID;

            bool visible = false;
            if (instanceIndex < TotalInstances)
            {
                float3 worldPos = InstancePosition(instanceIndex);
                visible = !IsFrustumCulled(worldPos, BoundingSphereRadius);
            }

            if (groupThreadID == 0)
                s_visibleCount = 0;
            GroupMemoryBarrierWithGroupSync();

            if (visible)
            {
                uint offset;
                InterlockedAdd(s_visibleCount, 1, offset);
                s_payload.InstanceIndices[offset] = instanceIndex;
            }
            GroupMemoryBarrierWithGroupSync();

            DispatchMesh(s_visibleCount, 1, 1, s_payload);
        }

        [shader("mesh")]
        [numthreads(120, 1, 1)]
        [outputtopology("triangle")]
        void MSMain(uint groupID : SV_GroupID,
                    uint groupThreadID : SV_GroupThreadID,
                    in payload Payload meshPayload,
                    OutputVertices<VertexOutput, 62> outVertices,
                    OutputIndices<uint3, 120> outIndices)
        {
            uint instanceIndex = meshPayload.InstanceIndices[groupID];
            float3 instancePos = InstancePosition(instanceIndex);
            float3 color = InstanceColor(instanceIndex);

            SetMeshOutputCounts(SphereVertexCount, SphereTriangleCount);

            if (groupThreadID < SphereVertexCount)
            {
                Vertex v = vertices[groupThreadID];
                float3 worldPos = v.Position + instancePos;

                VertexOutput output;
                output.Position = mul(float4(worldPos, 1.0), scene.ViewProjection);
                output.WorldNormal = v.Normal;
                output.Color = color;

                outVertices[groupThreadID] = output;
            }

            if (groupThreadID < SphereTriangleCount)
            {
                outIndices[groupThreadID] = indices[groupThreadID].Indices;
            }
        }

        [shader("pixel")]
        float4 PSMain(VertexOutput input) : SV_Target
        {
            float3 lightDir = normalize(scene.LightDirection);
            float3 normal = normalize(input.WorldNormal);
            float ndotl = max(dot(normal, lightDir), 0.0);

            float3 ambient = input.Color * 0.15;
            float3 diffuse = input.Color * ndotl * 0.85;

            return float4(ambient + diffuse, 1.0);
        }
        """;

    private readonly Buffer vertexBuffer;
    private readonly Buffer indexBuffer;
    private readonly Buffer constantBuffer;
    private readonly ResourceLayout resourceLayout;
    private readonly ResourceTable resourceTable;
    private readonly MeshShadingPipeline pipeline;

    private float totalTime;

    public MeshShadingRenderer()
    {
        if (!App.Context.Capabilities.MeshShadingSupported)
        {
            throw new NotSupportedException("Mesh shading is not supported on this device.");
        }

        const int lonSegments = 12;
        const int latSegments = 6;
        const float radius = 0.5f;

        List<Vertex> sphereVertices = [];
        List<Triangle> sphereTriangles = [];

        sphereVertices.Add(new() { Position = new(0, radius, 0), Normal = Vector3.UnitY });

        for (int lat = 1; lat < latSegments; lat++)
        {
            float phi = MathF.PI * lat / latSegments;
            float sinPhi = MathF.Sin(phi);
            float cosPhi = MathF.Cos(phi);

            for (int lon = 0; lon < lonSegments; lon++)
            {
                float theta = 2.0f * MathF.PI * lon / lonSegments;
                Vector3 normal = new(sinPhi * MathF.Cos(theta), cosPhi, sinPhi * MathF.Sin(theta));
                sphereVertices.Add(new() { Position = normal * radius, Normal = normal });
            }
        }

        sphereVertices.Add(new() { Position = new(0, -radius, 0), Normal = -Vector3.UnitY });

        for (int lon = 0; lon < lonSegments; lon++)
        {
            uint next = (uint)((lon + 1) % lonSegments);
            sphereTriangles.Add(new() { I0 = 0, I1 = (uint)(1 + lon), I2 = 1 + next });
        }

        for (int lat = 0; lat < latSegments - 2; lat++)
        {
            for (int lon = 0; lon < lonSegments; lon++)
            {
                uint next = (uint)((lon + 1) % lonSegments);
                uint tl = (uint)(1 + lat * lonSegments + lon);
                uint tr = (uint)(1 + lat * lonSegments) + next;
                uint bl = (uint)(1 + (lat + 1) * lonSegments + lon);
                uint br = (uint)(1 + (lat + 1) * lonSegments) + next;
                sphereTriangles.Add(new() { I0 = tl, I1 = bl, I2 = tr });
                sphereTriangles.Add(new() { I0 = tr, I1 = bl, I2 = br });
            }
        }

        uint bottomPole = (uint)(sphereVertices.Count - 1);
        uint lastRing = (uint)(1 + (latSegments - 2) * lonSegments);
        for (int lon = 0; lon < lonSegments; lon++)
        {
            uint next = (uint)((lon + 1) % lonSegments);
            sphereTriangles.Add(new() { I0 = bottomPole, I1 = lastRing + next, I2 = lastRing + (uint)lon });
        }

        Vertex[] vertexData = [.. sphereVertices];
        Triangle[] triangleData = [.. sphereTriangles];

        vertexBuffer = App.Context.CreateBuffer(new()
        {
            SizeInBytes = (uint)(sizeof(Vertex) * vertexData.Length),
            StrideInBytes = (uint)sizeof(Vertex),
            Flags = BufferUsageFlags.ShaderResource
        });
        vertexBuffer.Upload(vertexData, 0);

        indexBuffer = App.Context.CreateBuffer(new()
        {
            SizeInBytes = (uint)(sizeof(Triangle) * triangleData.Length),
            StrideInBytes = (uint)sizeof(Triangle),
            Flags = BufferUsageFlags.ShaderResource
        });
        indexBuffer.Upload(triangleData, 0);

        constantBuffer = App.Context.CreateBuffer(new()
        {
            SizeInBytes = (uint)sizeof(SceneConstants),
            StrideInBytes = (uint)sizeof(SceneConstants),
            Flags = BufferUsageFlags.Constant | BufferUsageFlags.MapWrite
        });

        resourceLayout = App.Context.CreateResourceLayout(new()
        {
            Bindings = BindingHelper.Bindings
            (
                new() { Type = ResourceType.ConstantBuffer, Count = 1, StageFlags = ShaderStageFlags.Amplification | ShaderStageFlags.Mesh | ShaderStageFlags.Pixel },
                new() { Type = ResourceType.StructuredBuffer, Count = 1, StageFlags = ShaderStageFlags.Mesh },
                new() { Type = ResourceType.StructuredBuffer, Count = 1, StageFlags = ShaderStageFlags.Mesh }
            )
        });

        resourceTable = App.Context.CreateResourceTable(new()
        {
            Layout = resourceLayout,
            Resources = [constantBuffer, vertexBuffer, indexBuffer]
        });

        using Shader ampShader = App.Context.LoadShaderFromSource(ShaderSource, "ASMain", ShaderStageFlags.Amplification);
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
            Amplification = ampShader,
            Mesh = meshShader,
            Pixel = pixelShader,
            ResourceLayout = resourceLayout,
            PrimitiveTopology = PrimitiveTopology.TriangleList,
            Output = App.FrameBuffer.Output,
            AmplificationThreadGroupSizeX = ASGroupSize,
            AmplificationThreadGroupSizeY = 1,
            AmplificationThreadGroupSizeZ = 1,
            MeshThreadGroupSizeX = MeshGroupSize,
            MeshThreadGroupSizeY = 1,
            MeshThreadGroupSizeZ = 1
        });
    }

    public void Update(double deltaTime)
    {
        totalTime += (float)deltaTime;

        float angle = totalTime * 0.3f;
        Vector3 cameraPos = new(
            35.0f * MathF.Sin(angle),
            20.0f * MathF.Sin(totalTime * 0.2f),
            35.0f * MathF.Cos(angle)
        );

        Matrix4x4 view = Matrix4x4.CreateLookAt(cameraPos, Vector3.Zero, Vector3.UnitY);
        Matrix4x4 projection = Matrix4x4.CreatePerspectiveFieldOfView(
            float.DegreesToRadians(45.0f), (float)App.Width / App.Height, 0.1f, 200.0f);
        Matrix4x4 vp = view * projection;

        constantBuffer.Upload([new SceneConstants
        {
            ViewProjection = vp,
            FrustumPlane0 = NormalizePlane(new(vp.M11 + vp.M14, vp.M21 + vp.M24, vp.M31 + vp.M34, vp.M41 + vp.M44)),
            FrustumPlane1 = NormalizePlane(new(vp.M14 - vp.M11, vp.M24 - vp.M21, vp.M34 - vp.M31, vp.M44 - vp.M41)),
            FrustumPlane2 = NormalizePlane(new(vp.M12 + vp.M14, vp.M22 + vp.M24, vp.M32 + vp.M34, vp.M42 + vp.M44)),
            FrustumPlane3 = NormalizePlane(new(vp.M14 - vp.M12, vp.M24 - vp.M22, vp.M34 - vp.M32, vp.M44 - vp.M42)),
            FrustumPlane4 = NormalizePlane(new(vp.M13,           vp.M23,           vp.M33,           vp.M43)),
            FrustumPlane5 = NormalizePlane(new(vp.M14 - vp.M13, vp.M24 - vp.M23, vp.M34 - vp.M33, vp.M44 - vp.M43)),
            Time = totalTime,
            LightDirection = -Vector3.Normalize(cameraPos)
        }], 0);
    }

    public void Render()
    {
        CommandBuffer commandBuffer = App.Context.Graphics.CommandBuffer();

        commandBuffer.BeginRenderPass(App.FrameBuffer, new()
        {
            ColorValues = [new(0.05f, 0.05f, 0.08f, 1.0f)],
            Depth = 1.0f,
            Stencil = 0,
            Flags = ClearFlags.All
        }, resourceTable);

        commandBuffer.SetPipeline(pipeline);
        commandBuffer.SetResourceTable(resourceTable);
        commandBuffer.DispatchMesh(DispatchGroupCount, 1, 1);

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
        indexBuffer.Dispose();
        vertexBuffer.Dispose();
    }

    private static Vector4 NormalizePlane(Vector4 plane)
    {
        float length = new Vector3(plane.X, plane.Y, plane.Z).Length();
        return plane / length;
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
    public uint I0;

    [FieldOffset(4)]
    public uint I1;

    [FieldOffset(8)]
    public uint I2;
}

[StructLayout(LayoutKind.Explicit, Size = 176)]
file struct SceneConstants
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
}
