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

[shader("vertex")]
FSInput VSMain(VSInput input)
{
    FSInput output;
    output.Position = float4(input.Position, 1.0);
    output.Color = input.Color;

    return output;
}

[shader("fragment")]
float4 FSMain(FSInput input) : SV_TARGET
{
    return input.Color;
}