struct Constants
{
    DescriptorHandle<Texture2D> Image;

    DescriptorHandle<SamplerState> Sampler;
};

ConstantBuffer<Constants> constants;

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
