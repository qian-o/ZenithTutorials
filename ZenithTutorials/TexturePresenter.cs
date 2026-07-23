namespace ZenithTutorials;

internal unsafe sealed class TexturePresenter : IDisposable
{
    private readonly Buffer constantBuffer;
    private readonly GraphicsPipeline pipeline;

    public TexturePresenter()
    {
        constantBuffer = App.LoadBuffer([new Constants()], BufferUsages.Constant);

        using Shader vertexShader = App.LoadShader("PresentTexture.slang", "VSMain");
        using Shader fragmentShader = App.LoadShader("PresentTexture.slang", "FSMain");

        pipeline = App.Context.CreateGraphicsPipeline(new()
        {
            VertexShader = vertexShader,
            FragmentShader = fragmentShader,
            InputLayouts = [],
            PrimitiveTopology = PrimitiveTopology.TriangleList,
            AttachmentFormats = new()
            {
                ColorFormats = [App.ColorFormat],
                SampleCount = SampleCount.Count1
            },
            RenderState = new()
            {
                Rasterizer = RasterizerState.CullNone(),
                DepthStencil = DepthStencilState.DepthNone(),
                Blend = BlendState.Opaque()
            }
        });
    }

    public void Present(CommandBuffer commandBuffer, Texture drawable, Texture texture, bool fitToWindow)
    {
        uint width = fitToWindow ? App.Width : Math.Min(texture.Desc.Width, App.Width);
        uint height = fitToWindow ? App.Height : Math.Min(texture.Desc.Height, App.Height);
        int x = (int)((App.Width - width) / 2);
        int y = (int)((App.Height - height) / 2);

        Constants constants = new()
        {
            Image = texture.SampledHandle,
            Sampler = App.LinearClampSampler.Handle
        };

        constantBuffer.Upload(0, new()
        {
            Pointer = (nint)(&constants),
            SizeInBytes = (uint)sizeof(Constants)
        });

        commandBuffer.BeginRenderPass([ColorAttachment.Clear(drawable, new(0.04f, 0.055f, 0.075f, 1.0f))], null);
        commandBuffer.SetPipeline(pipeline);
        commandBuffer.SetViewports([new() { X = x, Y = y, Width = width, Height = height, MaxDepth = 1.0f }]);
        commandBuffer.SetScissors([new() { X = x, Y = y, Width = width, Height = height }]);
        commandBuffer.SetConstantBuffer(constantBuffer, 0);
        commandBuffer.Draw(3, 1, 0, 0);
        commandBuffer.EndRenderPass();
    }

    public void Dispose()
    {
        pipeline.Dispose();
        constantBuffer.Dispose();
    }
}

[StructLayout(LayoutKind.Explicit, Size = 16)]
file struct Constants
{
    [FieldOffset(0)]
    public ResourceHandle Image;

    [FieldOffset(8)]
    public ResourceHandle Sampler;
}