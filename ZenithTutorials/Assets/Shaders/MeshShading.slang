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

    property float3 Position
    {
        get {
            return PositionAndPadding.xyz;
        }
    }

    property float3 Normal
    {
        get {
            return NormalAndPadding.xyz;
        }
    }
};

struct Triangle
{
    private uint4 IndicesAndPadding;

    property uint3 Indices
    {
        get {
            return IndicesAndPadding.xyz;
        }
    }
};

struct Payload
{
    uint InstanceIndices[ASGroupSize];
};

struct VertexOutput
{
    float4 Position : SV_POSITION;

    float3 WorldNormal : WORLDNORMAL;

    float3 Color : COLOR;
};

struct Constants
{
    float4x4 ViewProjection;

    float4 FrustumPlanes[6];

    private float4 LightDirectionAndPadding;

    DescriptorHandle<StructuredBuffer<Vertex>> Vertices;

    DescriptorHandle<StructuredBuffer<Triangle>> Triangles;

    property float3 LightDirection
    {
        get {
            return LightDirectionAndPadding.xyz;
        }
    }
};

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
        float4 plane = constants.FrustumPlanes[i];
        if (dot(plane.xyz, center) + plane.w < -radius)
        {
            return true;
        }
    }
    return false;
}

ConstantBuffer<Constants> constants;

groupshared Payload s_payload;
groupshared uint s_visibleCount;

[shader("task")]
[numthreads(ASGroupSize, 1, 1)]
void ASMain(uint groupID: SV_GroupID, uint groupThreadID: SV_GroupThreadID)
{
    uint instanceIndex = groupID * ASGroupSize + groupThreadID;

    bool visible = false;
    if (instanceIndex < TotalInstances)
    {
        float3 worldPos = InstancePosition(instanceIndex);
        visible = !IsFrustumCulled(worldPos, BoundingSphereRadius);
    }

    if (groupThreadID == 0)
    {
        s_visibleCount = 0;
    }

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
void MSMain(uint groupID: SV_GroupID, uint groupThreadID: SV_GroupThreadID, in payload Payload meshPayload, OutputVertices<VertexOutput, 62> outVertices, OutputIndices<uint3, 120> outIndices)
{
    uint instanceIndex = meshPayload.InstanceIndices[groupID];
    float3 instancePos = InstancePosition(instanceIndex);
    float3 color = InstanceColor(instanceIndex);

    SetMeshOutputCounts(SphereVertexCount, SphereTriangleCount);

    if (groupThreadID < SphereVertexCount)
    {
        Vertex v = constants.Vertices[groupThreadID];
        float3 worldPos = v.Position + instancePos;

        VertexOutput output;
        output.Position = mul(float4(worldPos, 1.0), constants.ViewProjection);
        output.WorldNormal = v.Normal;
        output.Color = color;

        outVertices[groupThreadID] = output;
    }

    if (groupThreadID < SphereTriangleCount)
    {
        outIndices[groupThreadID] = constants.Triangles[groupThreadID].Indices;
    }
}

[shader("fragment")]
float4 FSMain(VertexOutput input) : SV_TARGET
{
    float3 lightDir = normalize(constants.LightDirection);
    float3 normal = normalize(input.WorldNormal);
    float ndotl = max(dot(normal, lightDir), 0.0);

    float3 ambient = input.Color * 0.15;
    float3 diffuse = input.Color * ndotl * 0.85;

    return float4(ambient + diffuse, 1.0);
}
