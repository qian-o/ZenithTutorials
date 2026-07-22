// tutorial:begin image-processing-shader
struct Constants
{
    uint Width;

    uint Height;

    DescriptorHandle<Texture2D> Input;

    DescriptorHandle<RWTexture2D<float4>> Output;

    DescriptorHandle<Texture2D> Image;

    DescriptorHandle<SamplerState> Sampler;
};

ConstantBuffer<Constants> constants;

[shader("compute")]
[numthreads(16, 16, 1)]
void CSMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadID.xy;

    if (pixel.x >= constants.Width || pixel.y >= constants.Height)
    {
        return;
    }

    float4 color = constants.Input.Load(int3(pixel, 0));
    float3 linear = pow(color.rgb, 2.2);
    float grayscale = dot(linear, float3(0.2126, 0.7152, 0.0722));
    grayscale = pow(grayscale, 1.0 / 2.2);

    constants.Output[pixel] = float4(grayscale, grayscale, grayscale, color.a);
}

struct VSOutput
{
    float4 Position : SV_POSITION;

    float2 TexCoord : TEXCOORD0;
};

[shader("vertex")]
VSOutput VSMain(uint vertexID: SV_VertexID)
{
    float2 texCoord = float2((vertexID << 1) & 2, vertexID & 2);

    VSOutput output;
    output.Position = float4(texCoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    output.TexCoord = texCoord;

    return output;
}

[shader("fragment")]
float4 FSMain(VSOutput input) : SV_TARGET
{
    return constants.Image.Sample(constants.Sampler, input.TexCoord);
}
// tutorial:end image-processing-shader
