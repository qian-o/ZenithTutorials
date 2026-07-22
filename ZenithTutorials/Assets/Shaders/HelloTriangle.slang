// tutorial:begin shader-inputs
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
// tutorial:end shader-inputs

// tutorial:begin vertex-stage
[shader("vertex")]
FSInput VSMain(VSInput input)
{
    FSInput output;
    output.Position = float4(input.Position, 1.0);
    output.Color = input.Color;

    return output;
}
// tutorial:end vertex-stage

// tutorial:begin fragment-stage
[shader("fragment")]
float4 FSMain(FSInput input) : SV_TARGET
{
    return input.Color;
}
// tutorial:end fragment-stage
