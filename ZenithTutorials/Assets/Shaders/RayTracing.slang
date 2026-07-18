static const float RayEpsilon = 0.001;
static const float TwoPi = 6.2831853;
static const uint SphereCount = 3;
static const uint ShadowSamples = 6;
static const uint ReflectionSamples = 4;
static const float ShadowMin = 0.3;
static const float SunRadius = 0.04;
static const float SphereRoughness = 0.05;
static const float SphereF0 = 0.15;
static const float FloorFadeStart = 8.0;
static const float FloorFadeRange = 20.0;

static const float3 FloorNormal = float3(0.0, 1.0, 0.0);
static const float3 LightDir = float3(0.6667, 0.6667, -0.3333);
static const float3 LightColor = float3(1.0, 0.98, 0.95);
static const float3 AmbientColor = float3(0.15, 0.15, 0.2);

struct Constants
{
    private float4 PositionAndPadding;

    DescriptorHandle<RaytracingAccelerationStructure> Scene;

    DescriptorHandle<StructuredBuffer<Sphere>> Spheres;

    DescriptorHandle<RWTexture2D<float4>> OutputTexture;

    DescriptorHandle<Texture2D> Image;

    DescriptorHandle<SamplerState> Sampler;

    property float3 Position
    {
        get {
            return PositionAndPadding.xyz;
        }
    }
};

struct Sphere
{
    private float4 CenterAndRadius;

    private float4 ColorAndPadding;

    property float3 Center
    {
        get {
            return CenterAndRadius.xyz;
        }
    }

    property float Radius
    {
        get {
            return CenterAndRadius.w;
        }
    }

    property float3 Color
    {
        get {
            return ColorAndPadding.xyz;
        }
    }
};

uniform Constants constants;

float3 SampleSky(float3 direction)
{
    float t = 0.5 * (direction.y + 1.0);
    float3 horizon = float3(0.7, 0.85, 1.0);
    float3 zenith = float3(0.3, 0.5, 1.0);
    float3 sky = lerp(horizon, zenith, saturate(t));

    float sunDot = dot(direction, LightDir);
    sky += LightColor * smoothstep(0.995, 0.999, sunDot) * 3.0;

    return sky;
}

float3 ACESFilm(float3 x)
{
    x *= 1.6;
    float3 a = x * (x * 2.51 + 0.03);
    float3 b = x * (x * 2.43 + 0.59) + 0.14;
    float3 result = saturate(a / b);

    float luma = dot(result, float3(0.2126, 0.7152, 0.0722));
    result = saturate(lerp(float3(luma, luma, luma), result, 1.5));

    return result;
}

float SchlickFresnel(float cosTheta, float f0)
{
    return f0 + (1.0 - f0) * pow(1.0 - cosTheta, 5.0);
}

float3 ShadeCheckerboard(float3 hitPoint, float3 normal, float3 rayDirection, bool softShadow, out float shadow)
{
    float2 fw = max(abs(fwidth_approx(hitPoint.xz)), 0.001);
    float2 fractPos = fract(hitPoint.xz) - 0.5;
    float2 filtered = clamp(fractPos / fw, -0.5, 0.5);
    float checker = 0.5 - 0.5 * filtered.x * filtered.y;
    float3 baseColor = lerp(float3(0.787, 0.787, 0.787), float3(0.1, 0.1, 0.1), checker);

    float NdotL = max(dot(normal, LightDir), 0.0);
    float3 shadowOrigin = hitPoint + normal * RayEpsilon;
    if (softShadow)
    {
        shadow = lerp(ShadowMin, 1.0, TraceSoftShadow(shadowOrigin, LightDir, hitPoint.xz * 100.0));
    }
    else if (TraceShadowRay(shadowOrigin, LightDir))
    {
        shadow = ShadowMin;
    }
    else
    {
        shadow = 1.0;
    }

    float3 litColor = baseColor * AmbientColor + baseColor * LightColor * NdotL * shadow;

    float ao = 1.0;
    for (uint i = 0; i < SphereCount; i++)
    {
        float3 toSphere = constants.Spheres[i].Center - hitPoint;
        float horizDist = length(toSphere.xz);
        float r = constants.Spheres[i].Radius;
        float occl = saturate(1.0 - horizDist / (r * 2.0));
        float hFactor = saturate(1.0 - toSphere.y / (r * 3.0));
        ao -= occl * hFactor * 0.4;
    }

    litColor *= max(ao, 0.3);

    float dist = length(hitPoint.xz);
    float fade = saturate((dist - FloorFadeStart) / FloorFadeRange);
    return lerp(litColor, SampleSky(rayDirection), fade);
}

float3 ShadeSphere(float3 hitPoint, float3 normal, float3 sphereColor, float3 viewDir, bool softShadow)
{
    float NdotL = max(dot(normal, LightDir), 0.0);

    float3 halfDir = normalize(LightDir + viewDir);
    float spec = pow(max(dot(normal, halfDir), 0.0), 64.0);

    float3 shadowOrigin = hitPoint + normal * RayEpsilon;
    float shadow;
    if (softShadow)
    {
        shadow = lerp(ShadowMin, 1.0, TraceSoftShadow(shadowOrigin, LightDir, hitPoint.xz * 100.0));
    }
    else if (TraceShadowRay(shadowOrigin, LightDir))
    {
        shadow = ShadowMin;
    }
    else
    {
        shadow = 1.0;
    }

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
    query.TraceRayInline(constants.Scene, RAY_FLAG_NONE, 0xFF, reflectRay);

    while (query.Proceed())
    {
        if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE)
        {
            uint sphereIndex = query.CandidatePrimitiveIndex();
            Sphere sphere = constants.Spheres[sphereIndex];

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
        float unused;
        return ShadeCheckerboard(hitPoint, FloorNormal, reflectRay.Direction, false, unused);
    }
    else if (query.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT)
    {
        float3 hitPoint = reflectRay.Origin + reflectRay.Direction * query.CommittedRayT();
        float3 viewDir = normalize(origin - hitPoint);
        return ShadeSphere(hitPoint, sphereNormal, sphereColor, viewDir, false);
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

float2 fwidth_approx(float2 p)
{
    float2 dx = float2(0.02, 0.0);
    float2 dy = float2(0.0, 0.02);
    return abs(fract(p + dx) - fract(p)) + abs(fract(p + dy) - fract(p));
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
    shadowQuery.TraceRayInline(constants.Scene, RAY_FLAG_NONE, 0xFF, shadowRay);

    while (shadowQuery.Proceed())
    {
        if (shadowQuery.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE)
        {
            uint sphereIndex = shadowQuery.CandidatePrimitiveIndex();
            Sphere sphere = constants.Spheres[sphereIndex];

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
    float3 tangent = normalize(cross(direction, float3(0.0, 1.0, 0.0)));
    float3 bitangent = cross(direction, tangent);

    float lit = 0.0;
    for (uint i = 0; i < ShadowSamples; i++)
    {
        float h = Hash(pixelSeed + float2(float(i) * 7.13, float(i) * 3.71));
        float angle = (float(i) + h) * (TwoPi / float(ShadowSamples));
        float radius = sqrt(Hash(pixelSeed + float2(float(i) * 11.07, 0.0))) * SunRadius;
        float3 jitteredDir = normalize(direction + tangent * cos(angle) * radius + bitangent * sin(angle) * radius);

        if (!TraceShadowRay(origin, jitteredDir))
        {
            lit += 1.0;
        }
    }

    return lit / float(ShadowSamples);
}

float3 TraceRoughReflection(float3 origin, float3 reflectDir, float3 normal, float roughness, float2 pixelSeed)
{
    float3 tangent = normalize(cross(reflectDir, normal));
    float3 bitangent = cross(reflectDir, tangent);

    float3 accum = float3(0.0);
    for (uint i = 0; i < ReflectionSamples; i++)
    {
        float h1 = Hash(pixelSeed + float2(float(i) * 5.17, float(i) * 9.23));
        float h2 = Hash(pixelSeed + float2(float(i) * 13.37, float(i) * 2.91));
        float angle = h1 * TwoPi;
        float radius = sqrt(h2) * roughness;
        float3 jitteredDir = normalize(reflectDir + tangent * cos(angle) * radius + bitangent * sin(angle) * radius);
        accum += TraceReflection(origin, jitteredDir);
    }

    return accum / float(ReflectionSamples);
}

float3 ShadeFloor(float3 hitPoint, float3 rayDir, float3 cameraPos)
{
    float shadow;
    float3 directColor = ShadeCheckerboard(hitPoint, FloorNormal, rayDir, true, shadow);

    float3 viewDir = normalize(cameraPos - hitPoint);
    float3 halfDir = normalize(LightDir + viewDir);
    float floorSpec = pow(max(dot(FloorNormal, halfDir), 0.0), 128.0);
    float specDist = length(hitPoint.xz);
    float specFade = 1.0 - saturate((specDist - FloorFadeStart) / FloorFadeRange);
    directColor += LightColor * floorSpec * 0.4 * specFade * shadow;

    float3 reflectDir = reflect(rayDir, FloorNormal);
    float3 reflectColor = TraceReflection(hitPoint + FloorNormal * RayEpsilon, reflectDir);
    float fresnel = SchlickFresnel(max(dot(FloorNormal, viewDir), 0.0), 0.02);

    return lerp(directColor, reflectColor, fresnel);
}

float3 ShadePrimarySphere(float3 hitPoint, float3 rayDir, float3 cameraPos, float3 normal, float3 sphereColor)
{
    float3 viewDir = normalize(cameraPos - hitPoint);

    float3 directColor = ShadeSphere(hitPoint, normal, sphereColor, viewDir, true);

    float3 reflectDir = reflect(rayDir, normal);
    float3 reflectColor = TraceRoughReflection(hitPoint + normal * RayEpsilon,
                                               reflectDir,
                                               normal,
                                               SphereRoughness,
                                               hitPoint.xz * 100.0);
    float fresnel = SchlickFresnel(max(dot(normal, viewDir), 0.0), SphereF0);

    return lerp(directColor, reflectColor, fresnel);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixelCoord = dispatchThreadID.xy;

    uint width, height;
    constants.OutputTexture.GetDimensions(width, height);

    if (pixelCoord.x >= width || pixelCoord.y >= height)
    {
        return;
    }

    float2 uv = (float2(pixelCoord) + 0.5) / float2(width, height);
    float2 ndc = uv * 2.0 - 1.0;
    ndc.y = -ndc.y;

    float aspectRatio = float(width) / float(height);
    float fov = tan(radians(45.0) * 0.5);

    float3 cameraPos = constants.Position;
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
    query.TraceRayInline(constants.Scene, RAY_FLAG_NONE, 0xFF, ray);

    while (query.Proceed())
    {
        if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE)
        {
            uint sphereIndex = query.CandidatePrimitiveIndex();
            Sphere sphere = constants.Spheres[sphereIndex];

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
        color = ShadeFloor(hitPoint, rayDir, cameraPos);
    }
    else if (query.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT)
    {
        float3 hitPoint = ray.Origin + ray.Direction * query.CommittedRayT();
        color = ShadePrimarySphere(hitPoint, rayDir, cameraPos, sphereHitNormal, sphereHitColor);
    }
    else
    {
        color = SampleSky(rayDir);
    }

    color = ACESFilm(color);

    constants.OutputTexture[pixelCoord] = float4(color, 1.0);
}
struct DisplayOutput
{
    float4 Position : SV_POSITION;

    float2 TexCoord : TEXCOORD0;
};

[shader("vertex")]
DisplayOutput VSMain(uint vertexID : SV_VertexID)
{
    float2 texCoord = float2((vertexID << 1) & 2, vertexID & 2);

    DisplayOutput output;
    output.Position = float4(texCoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    output.TexCoord = texCoord;

    return output;
}

[shader("fragment")]
float4 FSMain(DisplayOutput input) : SV_TARGET
{
    return constants.Image.Sample(constants.Sampler, input.TexCoord);
}
