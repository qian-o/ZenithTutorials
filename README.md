# ZenithTutorials

Runnable source for the [Zenith.NET tutorials](https://qian-o.github.io/Zenith.NET/tutorials/).

This repository contains the runnable tutorial code. The documentation explains each workload and links here for complete, buildable C# and Slang sources.

## Build and Run

Install the .NET 10 SDK, then build and run the project:

```shell
dotnet build ZenithTutorials.slnx
dotnet run --project ZenithTutorials/ZenithTutorials.csproj
```

Select a tutorial from the console menu. The shared application host creates the window, graphics context, swap chain, and command buffers; each renderer records one workload.

The host uses DirectX 12 on Windows, Metal on macOS, and Vulkan on Linux. The Linux window path requires X11 or XWayland.

## Tutorials

| Tutorial | Renderer | Shader |
| --- | --- | --- |
| Hello Triangle | [HelloTriangleRenderer.cs](ZenithTutorials/Renderers/HelloTriangleRenderer.cs) | [HelloTriangle.slang](ZenithTutorials/Assets/Shaders/HelloTriangle.slang) |
| Spinning Cube | [SpinningCubeRenderer.cs](ZenithTutorials/Renderers/SpinningCubeRenderer.cs) | [SpinningCube.slang](ZenithTutorials/Assets/Shaders/SpinningCube.slang) |
| Compute Shader | [ComputeShaderRenderer.cs](ZenithTutorials/Renderers/ComputeShaderRenderer.cs) | [ComputeShader.slang](ZenithTutorials/Assets/Shaders/ComputeShader.slang) |
| Indirect Drawing | [IndirectDrawingRenderer.cs](ZenithTutorials/Renderers/IndirectDrawingRenderer.cs) | [IndirectDrawing.slang](ZenithTutorials/Assets/Shaders/IndirectDrawing.slang) |
| Ray Tracing | [RayTracingRenderer.cs](ZenithTutorials/Renderers/RayTracingRenderer.cs) | [RayTracing.slang](ZenithTutorials/Assets/Shaders/RayTracing.slang) |
| Mesh Shading | [MeshShadingRenderer.cs](ZenithTutorials/Renderers/MeshShadingRenderer.cs) | [MeshShading.slang](ZenithTutorials/Assets/Shaders/MeshShading.slang) |

Shared host sources:

- [Program.cs](ZenithTutorials/Program.cs)
- [App.cs](ZenithTutorials/App.cs)
- [IRenderer.cs](ZenithTutorials/IRenderer.cs)
- [CocoaHelper.cs](ZenithTutorials/CocoaHelper.cs)
- [Usings.cs](ZenithTutorials/Usings.cs)
