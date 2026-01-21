using System.Numerics;
using System.Runtime.InteropServices;
using Zenith.NET;
using Zenith.NET.Extensions.Slang;
using Buffer = Zenith.NET.Buffer;

namespace ZenithTutorials.Renderers;

internal class HelloTriangleRenderer : IRenderer
{
    /// <summary>
    /// Vertex structure with position and color data.
    /// LayoutKind.Sequential ensures memory layout matches GPU expectations.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct Vertex(Vector3 position, Vector4 color)
    {
        public Vector3 Position = position;

        public Vector4 Color = color;
    }

    // Slang shader source (HLSL-like syntax, compiled at runtime)
    private const string shaderSource = """
        struct VSInput
        {
            float3 Position : POSITION0;

            float4 Color    : COLOR0;
        };

        struct PSInput
        {
            float4 Position : SV_POSITION;

            float4 Color    : COLOR0;
        };

        PSInput VSMain(VSInput input)
        {
            PSInput output;
            output.Position = float4(input.Position, 1.0);
            output.Color = input.Color;

            return output;
        }

        float4 PSMain(PSInput input) : SV_TARGET
        {
            return input.Color;
        }
        """;

    private readonly Buffer vertexBuffer;
    private readonly GraphicsPipeline pipeline;

    public HelloTriangleRenderer()
    {
        // Define triangle vertices (NDC coordinates: -1 to 1)
        Vertex[] vertices =
        [
            new(new( 0.0f,  0.5f, 0.0f), new(1.0f, 0.0f, 0.0f, 1.0f)), // Top    - Red
            new(new( 0.5f, -0.5f, 0.0f), new(0.0f, 1.0f, 0.0f, 1.0f)), // Right  - Green
            new(new(-0.5f, -0.5f, 0.0f), new(0.0f, 0.0f, 1.0f, 1.0f)), // Left   - Blue
        ];

        // Create GPU buffer for vertex data
        vertexBuffer = App.Context.CreateBuffer(new()
        {
            SizeInBytes = (uint)(Marshal.SizeOf<Vertex>() * vertices.Length),
            StrideInBytes = (uint)Marshal.SizeOf<Vertex>(),
            Flags = BufferUsageFlags.Vertex | BufferUsageFlags.MapWrite
        });
        vertexBuffer.Upload(vertices, 0);

        // Define vertex input layout (must match shader VSInput)
        InputLayout inputLayout = new();
        inputLayout.Add(new() { Format = ElementFormat.Float3, Semantic = ElementSemantic.Position });
        inputLayout.Add(new() { Format = ElementFormat.Float4, Semantic = ElementSemantic.Color });

        // Compile shaders using Slang
        using Shader vertexShader = App.Context.LoadShaderFromSource(shaderSource, "VSMain", ShaderStageFlags.Vertex);
        using Shader pixelShader = App.Context.LoadShaderFromSource(shaderSource, "PSMain", ShaderStageFlags.Pixel);

        // Create graphics pipeline (combines all render state)
        pipeline = App.Context.CreateGraphicsPipeline(new()
        {
            RenderStates = new()
            {
                RasterizerState = RasterizerStates.CullNone,     // Disable back-face culling
                DepthStencilState = DepthStencilStates.Default,  // Enable depth testing
                BlendState = BlendStates.Opaque                  // No alpha blending
            },
            Vertex = vertexShader,
            Pixel = pixelShader,
            ResourceLayouts = [],
            InputLayouts = [inputLayout],
            PrimitiveTopology = PrimitiveTopology.TriangleList,
            Output = App.SwapChain.FrameBuffer.Output
        });
    }

    public void Update(double deltaTime)
    {
    }

    public void Render()
    {
        // Get command buffer from graphics queue
        CommandBuffer commandBuffer = App.Context.Graphics.CommandBuffer();

        // Begin render pass (clears color and depth buffers)
        commandBuffer.BeginRenderPass(App.SwapChain.FrameBuffer, new()
        {
            ColorValues = [new(0.1f, 0.1f, 0.1f, 1.0f)],  // Dark gray background
            Depth = 1.0f,
            Stencil = 0,
            Flags = ClearFlags.All
        });

        // Record draw commands
        commandBuffer.SetPipeline(pipeline);
        commandBuffer.SetVertexBuffer(vertexBuffer, 0, 0);
        commandBuffer.Draw(3, 1, 0, 0);  // 3 vertices, 1 instance

        commandBuffer.EndRenderPass();

        // Submit and wait for GPU to finish
        commandBuffer.Submit(waitForCompletion: true);
    }

    public void Resize(uint width, uint height)
    {
    }

    public void Dispose()
    {
        // Release GPU resources in reverse creation order
        pipeline.Dispose();
        vertexBuffer.Dispose();
    }
}
