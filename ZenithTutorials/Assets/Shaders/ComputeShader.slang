struct Constants
{
    uint Width;

    uint Height;

    DescriptorHandle<Texture2D> Input;

    DescriptorHandle<RWTexture2D<float4>> Output;
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
