// tutorial:begin shader-inputs
struct VSInput
{
    float3 Position : POSITION0;

    float4 Color : COLOR0;

    uint InstanceID : SV_InstanceID;
};

struct VSOutput
{
    float4 Position : SV_POSITION;

    float4 Color : COLOR0;
};
// tutorial:end shader-inputs

// tutorial:begin instance-data
struct Instance
{
    float4x4 Model;

    float4 Color;
};

struct Constants
{
    float4x4 View;

    float4x4 Projection;

    DescriptorHandle<StructuredBuffer<Instance>> Instances;
};

ConstantBuffer<Constants> constants;
// tutorial:end instance-data

// tutorial:begin vertex-stage
[shader("vertex")]
VSOutput VSMain(VSInput input)
{
    Instance instance = constants.Instances[input.InstanceID];
    float4 worldPosition = mul(float4(input.Position, 1.0), instance.Model);
    float4 viewPosition = mul(worldPosition, constants.View);

    VSOutput output;
    output.Position = mul(viewPosition, constants.Projection);
    output.Color = input.Color * instance.Color;

    return output;
}
// tutorial:end vertex-stage

// tutorial:begin fragment-stage
[shader("fragment")]
float4 FSMain(VSOutput input) : SV_TARGET
{
    return input.Color;
}
// tutorial:end fragment-stage
