using System.Numerics;
using System.Runtime.InteropServices;
using Zenith.NET;
using Zenith.NET.Extensions.Slang;
using Buffer = Zenith.NET.Buffer;

namespace ZenithTutorials.Renderers;

internal class SpinningCubeRenderer : IRenderer
{
    [StructLayout(LayoutKind.Sequential)]
    private struct Vertex(Vector3 position, Vector4 color)
    {
        public Vector3 Position = position;

        public Vector4 Color = color;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct MVPConstants
    {
        public Matrix4x4 Model;

        public Matrix4x4 View;

        public Matrix4x4 Projection;
    }

    private const string shaderSource = """
        struct MVPConstants
        {
            float4x4 Model;

            float4x4 View;

            float4x4 Projection;
        };

        struct VSInput
        {
            float3 Position : POSITION0;

            float4 Color : COLOR0;
        };

        struct PSInput
        {
            float4 Position : SV_POSITION;

            float4 Color : COLOR0;
        };

        ConstantBuffer<MVPConstants> mvp;

        PSInput VSMain(VSInput input)
        {
            float4 worldPos = mul(float4(input.Position, 1.0), mvp.Model);
            float4 viewPos = mul(worldPos, mvp.View);
        
            PSInput output;
            output.Position = mul(viewPos, mvp.Projection);
            output.Color = input.Color;

            return output;
        }

        float4 PSMain(PSInput input) : SV_TARGET
        {
            return input.Color;
        }
        """;

    private readonly Buffer vertexBuffer;
    private readonly Buffer indexBuffer;
    private readonly Buffer constantBuffer;
    private readonly ResourceLayout resourceLayout;
    private readonly ResourceSet resourceSet;
    private readonly GraphicsPipeline pipeline;

    private float rotationAngle;

    public SpinningCubeRenderer()
    {
        // Define cube vertices (8 corners with different colors)
        Vertex[] vertices =
        [
            // Front face
            new(new(-0.5f, -0.5f,  0.5f), new(1.0f, 0.0f, 0.0f, 1.0f)), // 0: Red
            new(new( 0.5f, -0.5f,  0.5f), new(0.0f, 1.0f, 0.0f, 1.0f)), // 1: Green
            new(new( 0.5f,  0.5f,  0.5f), new(0.0f, 0.0f, 1.0f, 1.0f)), // 2: Blue
            new(new(-0.5f,  0.5f,  0.5f), new(1.0f, 1.0f, 0.0f, 1.0f)), // 3: Yellow
            // Back face
            new(new(-0.5f, -0.5f, -0.5f), new(1.0f, 0.0f, 1.0f, 1.0f)), // 4: Magenta
            new(new( 0.5f, -0.5f, -0.5f), new(0.0f, 1.0f, 1.0f, 1.0f)), // 5: Cyan
            new(new( 0.5f,  0.5f, -0.5f), new(1.0f, 1.0f, 1.0f, 1.0f)), // 6: White
            new(new(-0.5f,  0.5f, -0.5f), new(0.5f, 0.5f, 0.5f, 1.0f)), // 7: Gray
        ];

        // Define cube indices (6 faces × 2 triangles × 3 vertices = 36 indices)
        uint[] indices =
        [
            // Front
            0, 1, 2, 0, 2, 3,
            // Back
            5, 4, 7, 5, 7, 6,
            // Left
            4, 0, 3, 4, 3, 7,
            // Right
            1, 5, 6, 1, 6, 2,
            // Top
            3, 2, 6, 3, 6, 7,
            // Bottom
            4, 5, 1, 4, 1, 0
        ];

        // Create vertex buffer
        vertexBuffer = App.Context.CreateBuffer(new()
        {
            SizeInBytes = (uint)(Marshal.SizeOf<Vertex>() * vertices.Length),
            StrideInBytes = (uint)Marshal.SizeOf<Vertex>(),
            Flags = BufferUsageFlags.Vertex | BufferUsageFlags.MapWrite
        });
        vertexBuffer.Upload(vertices, 0);

        // Create index buffer
        indexBuffer = App.Context.CreateBuffer(new()
        {
            SizeInBytes = (uint)(sizeof(uint) * indices.Length),
            StrideInBytes = sizeof(uint),
            Flags = BufferUsageFlags.Index | BufferUsageFlags.MapWrite
        });
        indexBuffer.Upload(indices, 0);

        // Create constant buffer for MVP matrices
        constantBuffer = App.Context.CreateBuffer(new()
        {
            SizeInBytes = (uint)Marshal.SizeOf<MVPConstants>(),
            StrideInBytes = (uint)Marshal.SizeOf<MVPConstants>(),
            Flags = BufferUsageFlags.Constant | BufferUsageFlags.MapWrite
        });

        // Define resource layout for constant buffer
        resourceLayout = App.Context.CreateResourceLayout(new()
        {
            Bindings = BindingHelper.Bindings
            (
                new ResourceBinding() { Type = ResourceType.ConstantBuffer, Count = 1, StageFlags = ShaderStageFlags.Vertex }
            )
        });

        // Create resource set
        resourceSet = App.Context.CreateResourceSet(new()
        {
            Layout = resourceLayout,
            Resources = [constantBuffer]
        });

        // Define vertex input layout
        InputLayout inputLayout = new();
        inputLayout.Add(new() { Format = ElementFormat.Float3, Semantic = ElementSemantic.Position });
        inputLayout.Add(new() { Format = ElementFormat.Float4, Semantic = ElementSemantic.Color });

        // Compile shaders
        using Shader vertexShader = App.Context.LoadShaderFromSource(shaderSource, "VSMain", ShaderStageFlags.Vertex);
        using Shader pixelShader = App.Context.LoadShaderFromSource(shaderSource, "PSMain", ShaderStageFlags.Pixel);

        // Create graphics pipeline
        pipeline = App.Context.CreateGraphicsPipeline(new()
        {
            RenderStates = new()
            {
                RasterizerState = RasterizerStates.CullBack,      // Enable back-face culling for 3D
                DepthStencilState = DepthStencilStates.Default,   // Enable depth testing
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
        // Rotate cube over time
        rotationAngle += (float)deltaTime;
    }

    public void Render()
    {
        // Update MVP matrices
        Matrix4x4 model = Matrix4x4.CreateRotationY(rotationAngle) * Matrix4x4.CreateRotationX(rotationAngle * 0.5f);
        Matrix4x4 view = Matrix4x4.CreateLookAt(new(0, 0, 3), Vector3.Zero, Vector3.UnitY);
        Matrix4x4 projection = Matrix4x4.CreatePerspectiveFieldOfView(float.DegreesToRadians(45.0f), (float)App.Width / App.Height, 0.1f, 100.0f);

        constantBuffer.Upload([new MVPConstants { Model = model, View = view, Projection = projection }], 0);

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
        commandBuffer.DrawIndexed(36, 1, 0, 0, 0);  // 36 indices

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
        constantBuffer.Dispose();
        indexBuffer.Dispose();
        vertexBuffer.Dispose();
    }
}
