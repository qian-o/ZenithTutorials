// tutorial:begin cube-shader
struct VSInput
{
    float3 Position : POSITION0;

    float4 Color : COLOR0;
};

struct FSInput
{
    float4 Position : SV_POSITION;

    float4 Color : COLOR0;
};

struct Constants
{
    float4x4 Model;

    float4x4 View;

    float4x4 Projection;
};

ConstantBuffer<Constants> constants;

[shader("vertex")]
FSInput VSMain(VSInput input)
{
    float4x4 modelView = mul(constants.Model, constants.View);
    float4x4 mvp = mul(modelView, constants.Projection);

    FSInput output;
    output.Position = mul(float4(input.Position, 1.0), mvp);
    output.Color = input.Color;

    return output;
}

[shader("fragment")]
float4 FSMain(FSInput input) : SV_TARGET
{
    return input.Color;
}
// tutorial:end cube-shader
