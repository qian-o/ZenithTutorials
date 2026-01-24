using System;
using System.IO;
using System.Numerics;
using System.Runtime.InteropServices;
using Zenith.NET;
using Zenith.NET.Extensions.ImageSharp;
using Zenith.NET.Extensions.Slang;
using Buffer = Zenith.NET.Buffer;

namespace ZenithTutorials.Renderers;

internal class TexturedQuadRenderer : IRenderer
{
    private const string ShaderSource = """
        struct VSInput
        {
            float3 Position : POSITION0;

            float2 TexCoord : TEXCOORD0;
        };

        struct PSInput
        {
            float4 Position : SV_POSITION;

            float2 TexCoord : TEXCOORD0;
        };

        Texture2D shaderTexture;
        SamplerState samplerState;

        PSInput VSMain(VSInput input)
        {
            PSInput output;
            output.Position = float4(input.Position, 1.0);
            output.TexCoord = input.TexCoord;

            return output;
        }

        float4 PSMain(PSInput input) : SV_TARGET
        {
            return shaderTexture.Sample(samplerState, input.TexCoord);
        }
        """;

    private readonly Buffer vertexBuffer;
    private readonly Buffer indexBuffer;
    private readonly Texture texture;
    private readonly Sampler sampler;
    private readonly ResourceLayout resourceLayout;
    private readonly ResourceSet resourceSet;
    private readonly GraphicsPipeline pipeline;

    public TexturedQuadRenderer()
    {
        // UV origin (0,0) is top-left, (1,1) is bottom-right
        Vertex[] vertices =
        [
            new(new(-0.5f,  0.5f, 0.0f), new(0.0f, 0.0f)),
            new(new( 0.5f,  0.5f, 0.0f), new(1.0f, 0.0f)),
            new(new( 0.5f, -0.5f, 0.0f), new(1.0f, 1.0f)),
            new(new(-0.5f, -0.5f, 0.0f), new(0.0f, 1.0f)),
        ];

        uint[] indices = [0, 1, 2, 0, 2, 3];

        vertexBuffer = App.Context.CreateBuffer(new()
        {
            SizeInBytes = (uint)(Marshal.SizeOf<Vertex>() * vertices.Length),
            StrideInBytes = (uint)Marshal.SizeOf<Vertex>(),
            Flags = BufferUsageFlags.Vertex | BufferUsageFlags.MapWrite
        });
        vertexBuffer.Upload(vertices, 0);

        indexBuffer = App.Context.CreateBuffer(new()
        {
            SizeInBytes = (uint)(sizeof(uint) * indices.Length),
            StrideInBytes = sizeof(uint),
            Flags = BufferUsageFlags.Index | BufferUsageFlags.MapWrite
        });
        indexBuffer.Upload(indices, 0);

        texture = App.Context.LoadTextureFromFile(Path.Combine(AppContext.BaseDirectory, "Assets", "shoko.png"), generateMipMaps: true);

        sampler = App.Context.CreateSampler(new()
        {
            U = AddressMode.Clamp,
            V = AddressMode.Clamp,
            W = AddressMode.Clamp,
            Filter = Filter.MinLinearMagLinearMipLinear,
            MaxLod = uint.MaxValue
        });

        resourceLayout = App.Context.CreateResourceLayout(new()
        {
            Bindings = BindingHelper.Bindings
            (
                new() { Type = ResourceType.Texture, Count = 1, StageFlags = ShaderStageFlags.Pixel },
                new() { Type = ResourceType.Sampler, Count = 1, StageFlags = ShaderStageFlags.Pixel }
            )
        });

        resourceSet = App.Context.CreateResourceSet(new()
        {
            Layout = resourceLayout,
            Resources = [texture, sampler]
        });

        InputLayout inputLayout = new();
        inputLayout.Add(new() { Format = ElementFormat.Float3, Semantic = ElementSemantic.Position });
        inputLayout.Add(new() { Format = ElementFormat.Float2, Semantic = ElementSemantic.TexCoord });

        using Shader vertexShader = App.Context.LoadShaderFromSource(ShaderSource, "VSMain", ShaderStageFlags.Vertex);
        using Shader pixelShader = App.Context.LoadShaderFromSource(ShaderSource, "PSMain", ShaderStageFlags.Pixel);

        pipeline = App.Context.CreateGraphicsPipeline(new()
        {
            RenderStates = new()
            {
                RasterizerState = RasterizerStates.CullNone,
                DepthStencilState = DepthStencilStates.Default,
                BlendState = BlendStates.Opaque
            },
            Vertex = vertexShader,
            Pixel = pixelShader,
            ResourceLayouts = [resourceLayout],
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
        CommandBuffer commandBuffer = App.Context.Graphics.CommandBuffer();

        commandBuffer.BeginRenderPass(App.SwapChain.FrameBuffer, new()
        {
            ColorValues = [new(0.1f, 0.1f, 0.1f, 1.0f)],
            Depth = 1.0f,
            Stencil = 0,
            Flags = ClearFlags.All
        }, resourceSet);

        commandBuffer.SetPipeline(pipeline);
        commandBuffer.SetResourceSet(resourceSet, 0);
        commandBuffer.SetVertexBuffer(vertexBuffer, 0, 0);
        commandBuffer.SetIndexBuffer(indexBuffer, 0, IndexFormat.UInt32);
        commandBuffer.DrawIndexed(6, 1, 0, 0, 0);

        commandBuffer.EndRenderPass();

        commandBuffer.Submit(waitForCompletion: true);
    }

    public void Resize(uint width, uint height)
    {
    }

    public void Dispose()
    {
        pipeline.Dispose();
        resourceSet.Dispose();
        resourceLayout.Dispose();
        sampler.Dispose();
        texture.Dispose();
        indexBuffer.Dispose();
        vertexBuffer.Dispose();
    }
}

/// <summary>
/// Vertex structure with position and texture coordinates.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
file struct Vertex(Vector3 position, Vector2 texCoord)
{
    public Vector3 Position = position;

    public Vector2 TexCoord = texCoord;
}
