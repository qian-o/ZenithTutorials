namespace ZenithTutorials.Renderers;

internal unsafe class RayTracingRenderer : IRenderer
{
    private const uint ThreadGroupSize = 16;

    private const string ShaderSource = """
        struct Sphere
        {
            private float4 CenterAndRadius;

            private float4 ColorAndPadding;

            property float3 Center { get { return CenterAndRadius.xyz; } }

            property float Radius { get { return CenterAndRadius.w; } }

            property float3 Color { get { return ColorAndPadding.xyz; } }
        };

        RaytracingAccelerationStructure scene;
        StructuredBuffer<Sphere> spheres;
        RWTexture2D<float4> outputTexture;

        struct CameraConstants
        {
            private float4 PositionAndPadding;

            property float3 Position { get { return PositionAndPadding.xyz; } }
        };

        ConstantBuffer<CameraConstants> camera;

        static const float RayEpsilon = 0.001;
        static const uint SphereCount = 3;
        static const float ShadowMin = 0.3;
        static const float FloorFadeStart = 8.0;
        static const float FloorFadeRange = 20.0;

        static const float3 LightDir = float3(0.6667, 0.6667, -0.3333);
        static const float3 LightColor = float3(1.0, 0.98, 0.95);
        static const float3 AmbientColor = float3(0.15, 0.15, 0.2);

        float3 SampleSky(float3 direction)
        {
            float t = 0.5 * (direction.y + 1.0);
            float3 horizon = float3(0.9, 0.85, 0.7);
            float3 zenith = float3(0.4, 0.6, 1.0);
            float3 sky = lerp(horizon, zenith, saturate(t));

            float sunDot = dot(direction, LightDir);
            sky += LightColor * smoothstep(0.995, 0.999, sunDot) * 3.0;

            return sky;
        }

        float3 ACESFilm(float3 x)
        {
            x *= 1.2;
            float3 a = x * (x * 2.51 + 0.03);
            float3 b = x * (x * 2.43 + 0.59) + 0.14;
            return saturate(a / b);
        }

        float SchlickFresnel(float cosTheta, float f0)
        {
            return f0 + (1.0 - f0) * pow(1.0 - cosTheta, 5.0);
        }

        float3 ShadeCheckerboard(float3 hitPoint, float3 normal, float3 rayDirection)
        {
            int checkX = int(floor(hitPoint.x));
            int checkZ = int(floor(hitPoint.z));
            bool isWhite = ((checkX + checkZ) & 1) == 0;
            float3 baseColor = isWhite ? float3(0.787, 0.787, 0.787) : float3(0.1, 0.1, 0.1);

            float NdotL = max(dot(normal, LightDir), 0.0);
            float3 shadowOrigin = hitPoint + normal * RayEpsilon;
            float shadow = lerp(ShadowMin, 1.0, TraceSoftShadow(shadowOrigin, LightDir, hitPoint.xz * 100.0));

            float3 litColor = baseColor * AmbientColor + baseColor * LightColor * NdotL * shadow;

            float ao = 1.0;
            for (uint i = 0; i < SphereCount; i++)
            {
                float3 toSphere = spheres[i].Center - hitPoint;
                float horizDist = length(toSphere.xz);
                float r = spheres[i].Radius;
                float occl = saturate(1.0 - horizDist / (r * 2.0));
                float hFactor = saturate(1.0 - toSphere.y / (r * 3.0));
                ao -= occl * hFactor * 0.4;
            }
            litColor *= max(ao, 0.3);

            float dist = length(hitPoint.xz);
            float fade = saturate((dist - FloorFadeStart) / FloorFadeRange);
            return lerp(litColor, SampleSky(rayDirection), fade);
        }

        float3 ShadeSphere(float3 hitPoint, float3 normal, float3 sphereColor, float3 viewDir)
        {
            float NdotL = max(dot(normal, LightDir), 0.0);

            float3 halfDir = normalize(LightDir + viewDir);
            float spec = pow(max(dot(normal, halfDir), 0.0), 64.0);

            float3 shadowOrigin = hitPoint + normal * RayEpsilon;
            float shadow = lerp(ShadowMin, 1.0, TraceSoftShadow(shadowOrigin, LightDir, hitPoint.xz * 100.0));

            float3 diffuse = sphereColor * LightColor * NdotL * shadow;
            float3 specular = LightColor * spec * shadow;
            float3 ambient = sphereColor * AmbientColor;

            return ambient + diffuse + specular;
        }

        float3 TraceReflection(float3 origin, float3 direction)
        {
            RayDesc reflectRay;
            reflectRay.Origin = origin;
            reflectRay.Direction = direction;
            reflectRay.TMin = RayEpsilon;
            reflectRay.TMax = 1000.0;

            float3 sphereNormal = float3(0.0);
            float3 sphereColor = float3(0.0);

            RayQuery<RAY_FLAG_NONE> query;
            query.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, reflectRay);

            while (query.Proceed())
            {
                if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE)
                {
                    uint sphereIndex = query.CandidatePrimitiveIndex();
                    Sphere sphere = spheres[sphereIndex];

                    float3 ro = query.CandidateObjectRayOrigin();
                    float3 rd = query.CandidateObjectRayDirection();

                    float t = IntersectSphere(ro, rd, sphere);

                    if (t >= query.RayTMin() && t <= query.CommittedRayT())
                    {
                        float3 hitPoint = ro + rd * t;
                        sphereNormal = normalize(hitPoint - sphere.Center);
                        sphereColor = sphere.Color;
                        query.CommitProceduralPrimitiveHit(t);
                    }
                }
            }

            if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
            {
                float3 hitPoint = reflectRay.Origin + reflectRay.Direction * query.CommittedRayT();
                return ShadeCheckerboard(hitPoint, float3(0.0, 1.0, 0.0), reflectRay.Direction);
            }
            else if (query.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT)
            {
                float3 hitPoint = reflectRay.Origin + reflectRay.Direction * query.CommittedRayT();
                float3 viewDir = normalize(origin - hitPoint);
                return ShadeSphere(hitPoint, sphereNormal, sphereColor, viewDir);
            }
            else
            {
                return SampleSky(direction);
            }
        }

        float IntersectSphere(float3 origin, float3 direction, Sphere sphere)
        {
            float3 oc = origin - sphere.Center;

            float b = dot(oc, direction);
            float c = dot(oc, oc) - sphere.Radius * sphere.Radius;
            float discriminant = b * b - c;

            if (discriminant > 0.0)
            {
                float sqrtD = sqrt(discriminant);
                float t1 = -b - sqrtD;

                if (t1 > 0.0)
                {
                    return t1;
                }

                float t2 = -b + sqrtD;

                if (t2 > 0.0)
                {
                    return t2;
                }
            }

            return -1.0;
        }

        float Hash(float2 p)
        {
            float3 p3 = fract(float3(p.xyx) * 0.1031);
            p3 += dot(p3, p3.yzx + 33.33);
            return fract((p3.x + p3.y) * p3.z);
        }

        bool TraceShadowRay(float3 origin, float3 direction)
        {
            RayDesc shadowRay;
            shadowRay.Origin = origin;
            shadowRay.Direction = direction;
            shadowRay.TMin = RayEpsilon;
            shadowRay.TMax = 1000.0;

            RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> shadowQuery;
            shadowQuery.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, shadowRay);

            while (shadowQuery.Proceed())
            {
                if (shadowQuery.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE)
                {
                    uint sphereIndex = shadowQuery.CandidatePrimitiveIndex();
                    Sphere sphere = spheres[sphereIndex];

                    float3 ro = shadowQuery.CandidateObjectRayOrigin();
                    float3 rd = shadowQuery.CandidateObjectRayDirection();

                    float t = IntersectSphere(ro, rd, sphere);

                    if (t >= shadowQuery.RayTMin() && t <= shadowQuery.CommittedRayT())
                    {
                        shadowQuery.CommitProceduralPrimitiveHit(t);
                    }
                }
            }

            return shadowQuery.CommittedStatus() != COMMITTED_NOTHING;
        }

        float TraceSoftShadow(float3 origin, float3 direction, float2 pixelSeed)
        {
            static const uint ShadowSamples = 6;
            static const float SunRadius = 0.04;

            float3 tangent = normalize(cross(direction, float3(0.0, 1.0, 0.0)));
            float3 bitangent = cross(direction, tangent);

            float lit = 0.0;
            for (uint i = 0; i < ShadowSamples; i++)
            {
                float h = Hash(pixelSeed + float2(float(i) * 7.13, float(i) * 3.71));
                float angle = (float(i) + h) * (6.2831853 / float(ShadowSamples));
                float radius = sqrt(Hash(pixelSeed + float2(float(i) * 11.07, 0.0))) * SunRadius;
                float3 jitteredDir = normalize(direction + tangent * cos(angle) * radius + bitangent * sin(angle) * radius);

                if (!TraceShadowRay(origin, jitteredDir))
                {
                    lit += 1.0;
                }
            }

            return lit / float(ShadowSamples);
        }

        [numthreads(16, 16, 1)]
        void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID)
        {
            uint2 pixelCoord = dispatchThreadID.xy;

            uint width, height;
            outputTexture.GetDimensions(width, height);

            if (pixelCoord.x >= width || pixelCoord.y >= height)
            {
                return;
            }

            float2 uv = (float2(pixelCoord) + 0.5) / float2(width, height);
            float2 ndc = uv * 2.0 - 1.0;
            ndc.y = -ndc.y;

            float aspectRatio = float(width) / float(height);
            float fov = tan(radians(45.0) * 0.5);

            float3 cameraPos = camera.Position;
            float3 cameraTarget = float3(0.0, 0.5, 0.0);
            float3 cameraUp = float3(0.0, 1.0, 0.0);

            float3 forward = normalize(cameraTarget - cameraPos);
            float3 right = normalize(cross(forward, cameraUp));
            float3 up = cross(right, forward);

            float3 rayDir = normalize(forward + ndc.x * aspectRatio * fov * right + ndc.y * fov * up);

            RayDesc ray;
            ray.Origin = cameraPos;
            ray.Direction = rayDir;
            ray.TMin = RayEpsilon;
            ray.TMax = 1000.0;

            float3 sphereHitNormal = float3(0.0);
            float3 sphereHitColor = float3(0.0);

            RayQuery<RAY_FLAG_NONE> query;
            query.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, ray);

            while (query.Proceed())
            {
                if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE)
                {
                    uint sphereIndex = query.CandidatePrimitiveIndex();
                    Sphere sphere = spheres[sphereIndex];

                    float3 ro = query.CandidateObjectRayOrigin();
                    float3 rd = query.CandidateObjectRayDirection();

                    float t = IntersectSphere(ro, rd, sphere);

                    if (t >= query.RayTMin() && t <= query.CommittedRayT())
                    {
                        float3 hitPoint = ro + rd * t;

                        sphereHitNormal = normalize(hitPoint - sphere.Center);
                        sphereHitColor = sphere.Color;

                        query.CommitProceduralPrimitiveHit(t);
                    }
                }
            }

            float3 color;

            if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
            {
                float3 hitPoint = ray.Origin + ray.Direction * query.CommittedRayT();
                float3 normal = float3(0.0, 1.0, 0.0);

                float3 directColor = ShadeCheckerboard(hitPoint, normal, rayDir);

                float3 viewDir = normalize(cameraPos - hitPoint);
                float3 halfDir = normalize(LightDir + viewDir);
                float floorSpec = pow(max(dot(normal, halfDir), 0.0), 128.0);
                float3 shadowOrigin = hitPoint + normal * RayEpsilon;
                float specShadow = TraceSoftShadow(shadowOrigin, LightDir, hitPoint.xz * 100.0);
                float specDist = length(hitPoint.xz);
                float specFade = 1.0 - saturate((specDist - FloorFadeStart) / FloorFadeRange);
                directColor += LightColor * floorSpec * 0.4 * specFade * specShadow;

                float3 reflectDir = reflect(rayDir, normal);
                float3 reflectColor = TraceReflection(hitPoint + normal * RayEpsilon, reflectDir);
                float fresnel = SchlickFresnel(max(dot(normal, viewDir), 0.0), 0.02);

                color = lerp(directColor, reflectColor, fresnel);
            }
            else if (query.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT)
            {
                float3 hitPoint = ray.Origin + ray.Direction * query.CommittedRayT();
                float3 viewDir = normalize(cameraPos - hitPoint);

                float3 directColor = ShadeSphere(hitPoint, sphereHitNormal, sphereHitColor, viewDir);

                float3 reflectDir = reflect(rayDir, sphereHitNormal);
                float3 reflectColor = TraceReflection(hitPoint + sphereHitNormal * RayEpsilon, reflectDir);
                float fresnel = SchlickFresnel(max(dot(sphereHitNormal, viewDir), 0.0), 0.04);

                color = lerp(directColor, reflectColor, fresnel);
            }
            else
            {
                color = SampleSky(rayDir);
            }

            color = ACESFilm(color);

            outputTexture[pixelCoord] = float4(color, 1.0);
        }
        """;

    private readonly Buffer floorVertexBuffer;
    private readonly Buffer floorIndexBuffer;
    private readonly Buffer sphereBuffer;
    private readonly Buffer aabbBuffer;
    private readonly Buffer cameraBuffer;
    private readonly BottomLevelAccelerationStructure floorBlas;
    private readonly BottomLevelAccelerationStructure sphereBlas;
    private readonly TopLevelAccelerationStructure tlas;
    private readonly ResourceLayout resourceLayout;
    private readonly ComputePipeline pipeline;

    private Texture? outputTexture;
    private ResourceTable? resourceTable;
    private float totalTime;

    public RayTracingRenderer()
    {
        if (!App.Context.Capabilities.RayTracingSupported)
        {
            throw new NotSupportedException("Ray tracing is not supported on this device.");
        }

        Vector3[] floorVertices =
        [
            new(-50.0f, 0.0f, -50.0f),
            new( 50.0f, 0.0f, -50.0f),
            new( 50.0f, 0.0f,  50.0f),
            new(-50.0f, 0.0f,  50.0f)
        ];
        uint[] floorIndices = [0, 1, 2, 0, 2, 3];

        floorVertexBuffer = App.Context.CreateBuffer(new()
        {
            SizeInBytes = (uint)(sizeof(Vector3) * floorVertices.Length),
            StrideInBytes = (uint)sizeof(Vector3),
            Flags = BufferUsageFlags.Vertex | BufferUsageFlags.AccelerationStructure
        });
        floorVertexBuffer.Upload(floorVertices, 0);

        floorIndexBuffer = App.Context.CreateBuffer(new()
        {
            SizeInBytes = (uint)(sizeof(uint) * floorIndices.Length),
            StrideInBytes = sizeof(uint),
            Flags = BufferUsageFlags.Index | BufferUsageFlags.AccelerationStructure
        });
        floorIndexBuffer.Upload(floorIndices, 0);

        Sphere[] sphereData =
        [
            new() { Center = new(-2.0f, 1.0f, 1.0f), Radius = 1.0f, Color = new(0.8f, 0.2f, 0.2f) },
            new() { Center = new( 2.0f, 1.2f, -1.0f), Radius = 1.2f, Color = new(0.2f, 0.4f, 0.8f) },
            new() { Center = new( 0.0f, 0.6f, -3.0f), Radius = 0.6f, Color = new(0.9f, 0.7f, 0.2f) }
        ];

        sphereBuffer = App.Context.CreateBuffer(new()
        {
            SizeInBytes = (uint)(sizeof(Sphere) * sphereData.Length),
            StrideInBytes = (uint)sizeof(Sphere),
            Flags = BufferUsageFlags.ShaderResource
        });
        sphereBuffer.Upload(sphereData, 0);

        Vector3[] aabbData = new Vector3[sphereData.Length * 2];
        for (int i = 0; i < sphereData.Length; i++)
        {
            aabbData[i * 2] = sphereData[i].Center - new Vector3(sphereData[i].Radius);
            aabbData[(i * 2) + 1] = sphereData[i].Center + new Vector3(sphereData[i].Radius);
        }

        aabbBuffer = App.Context.CreateBuffer(new()
        {
            SizeInBytes = (uint)(sizeof(Vector3) * aabbData.Length),
            StrideInBytes = (uint)(sizeof(Vector3) * 2),
            Flags = BufferUsageFlags.ShaderResource | BufferUsageFlags.AccelerationStructure
        });
        aabbBuffer.Upload(aabbData, 0);

        CommandBuffer commandBuffer = App.Context.Graphics.CommandBuffer();

        floorBlas = commandBuffer.BuildAccelerationStructure(new BottomLevelAccelerationStructureDesc
        {
            Geometries =
            [
                new()
                {
                    Type = RayTracingGeometryType.Triangles,
                    Triangles = new()
                    {
                        VertexBuffer = floorVertexBuffer,
                        VertexFormat = PixelFormat.R32G32B32Float,
                        VertexCount = (uint)floorVertices.Length,
                        VertexStrideInBytes = (uint)sizeof(Vector3),
                        IndexBuffer = floorIndexBuffer,
                        IndexFormat = IndexFormat.UInt32,
                        IndexCount = (uint)floorIndices.Length,
                        Transform = Matrix4x4.Identity
                    },
                    Flags = RayTracingGeometryFlags.Opaque
                }
            ],
            Flags = AccelerationStructureBuildFlags.PreferFastTrace
        });

        sphereBlas = commandBuffer.BuildAccelerationStructure(new BottomLevelAccelerationStructureDesc
        {
            Geometries =
            [
                new()
                {
                    Type = RayTracingGeometryType.AABBs,
                    AABBs = new()
                    {
                        Buffer = aabbBuffer,
                        Count = (uint)sphereData.Length,
                        StrideInBytes = (uint)(sizeof(Vector3) * 2)
                    },
                    Flags = RayTracingGeometryFlags.Opaque
                }
            ],
            Flags = AccelerationStructureBuildFlags.PreferFastTrace
        });

        tlas = commandBuffer.BuildAccelerationStructure(new TopLevelAccelerationStructureDesc
        {
            Instances =
            [
                new()
                {
                    AccelerationStructure = floorBlas,
                    ID = 0,
                    Mask = 0xFF,
                    Transform = Matrix4x4.Identity,
                    Flags = RayTracingInstanceFlags.None
                },
                new()
                {
                    AccelerationStructure = sphereBlas,
                    ID = 1,
                    Mask = 0xFF,
                    Transform = Matrix4x4.Identity,
                    Flags = RayTracingInstanceFlags.None
                }
            ],
            Flags = AccelerationStructureBuildFlags.PreferFastTrace
        });

        commandBuffer.Submit(waitForCompletion: true);

        cameraBuffer = App.Context.CreateBuffer(new()
        {
            SizeInBytes = (uint)sizeof(CameraConstants),
            StrideInBytes = (uint)sizeof(CameraConstants),
            Flags = BufferUsageFlags.Constant | BufferUsageFlags.MapWrite
        });

        resourceLayout = App.Context.CreateResourceLayout(new()
        {
            Bindings = BindingHelper.Bindings
            (
                new() { Type = ResourceType.AccelerationStructure, Count = 1, StageFlags = ShaderStageFlags.Compute },
                new() { Type = ResourceType.StructuredBuffer, Count = 1, StageFlags = ShaderStageFlags.Compute },
                new() { Type = ResourceType.TextureReadWrite, Count = 1, StageFlags = ShaderStageFlags.Compute },
                new() { Type = ResourceType.ConstantBuffer, Count = 1, StageFlags = ShaderStageFlags.Compute }
            )
        });

        using Shader computeShader = App.Context.LoadShaderFromSource(ShaderSource, "CSMain", ShaderStageFlags.Compute);

        pipeline = App.Context.CreateComputePipeline(new()
        {
            Compute = computeShader,
            ResourceLayout = resourceLayout,
            ThreadGroupSizeX = ThreadGroupSize,
            ThreadGroupSizeY = ThreadGroupSize,
            ThreadGroupSizeZ = 1
        });
    }

    public void Update(double deltaTime)
    {
        totalTime += (float)deltaTime;

        float angle = totalTime * 0.3f;
        Vector3 cameraPos = new(
            12.0f * MathF.Sin(angle),
            4.0f + MathF.Sin(totalTime * 0.2f),
            -12.0f * MathF.Cos(angle)
        );

        cameraBuffer.Upload([new CameraConstants { Position = cameraPos }], 0);
    }

    public void Render()
    {
        outputTexture ??= App.Context.CreateTexture(new()
        {
            Type = TextureType.Texture2D,
            Format = PixelFormat.B8G8R8A8UNorm,
            Width = App.Width,
            Height = App.Height,
            Depth = 1,
            MipLevels = 1,
            ArrayLayers = 1,
            SampleCount = SampleCount.Count1,
            Flags = TextureUsageFlags.ShaderResource | TextureUsageFlags.UnorderedAccess
        });

        resourceTable ??= App.Context.CreateResourceTable(new()
        {
            Layout = resourceLayout,
            Resources = [tlas, sphereBuffer, outputTexture, cameraBuffer]
        });

        CommandBuffer commandBuffer = App.Context.Graphics.CommandBuffer();

        commandBuffer.SetPipeline(pipeline);
        commandBuffer.SetResourceTable(resourceTable);

        uint dispatchX = (App.Width + ThreadGroupSize - 1) / ThreadGroupSize;
        uint dispatchY = (App.Height + ThreadGroupSize - 1) / ThreadGroupSize;

        commandBuffer.Dispatch(dispatchX, dispatchY, 1);

        commandBuffer.CopyTexture(outputTexture,
                                  default,
                                  default,
                                  App.FrameBuffer.Desc.ColorAttachments[0].Target,
                                  default,
                                  default,
                                  new() { Width = App.Width, Height = App.Height, Depth = 1 });

        commandBuffer.Submit(waitForCompletion: true);
    }

    public void Resize(uint width, uint height)
    {
        resourceTable?.Dispose();
        resourceTable = null;

        outputTexture?.Dispose();
        outputTexture = null;
    }

    public void Dispose()
    {
        resourceTable?.Dispose();
        outputTexture?.Dispose();

        pipeline.Dispose();
        resourceLayout.Dispose();
        tlas.Dispose();
        sphereBlas.Dispose();
        floorBlas.Dispose();
        cameraBuffer.Dispose();
        aabbBuffer.Dispose();
        sphereBuffer.Dispose();
        floorIndexBuffer.Dispose();
        floorVertexBuffer.Dispose();
    }
}

[StructLayout(LayoutKind.Explicit, Size = 16)]
file struct CameraConstants
{
    [FieldOffset(0)]
    public Vector3 Position;
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
