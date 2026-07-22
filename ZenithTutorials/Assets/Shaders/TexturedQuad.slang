// tutorial:begin textured-quad-shader
struct VSInput
{
    float3 Position : POSITION0;

    float2 TexCoord : TEXCOORD0;
};

struct FSInput
{
    float4 Position : SV_POSITION;

    float2 TexCoord : TEXCOORD0;
};

struct Constants
{
    DescriptorHandle<Texture2D> Texture;

    DescriptorHandle<SamplerState> Sampler;
};

ConstantBuffer<Constants> constants;

[shader("vertex")]
FSInput VSMain(VSInput input)
{
    FSInput output;
    output.Position = float4(input.Position, 1.0);
    output.TexCoord = input.TexCoord;

    return output;
}

[shader("fragment")]
float4 FSMain(FSInput input) : SV_TARGET
{
    return constants.Texture.Sample(constants.Sampler, input.TexCoord);
}
// tutorial:end textured-quad-shader
