# ZenithTutorials

Runnable source for the [Zenith.NET tutorials](https://qian-o.github.io/Zenith.NET/tutorials/).

This repository is the source of truth for tutorial code. The documentation explains each workload and links here for complete, buildable C# and Slang sources.

## Repository Layout

ZenithTutorials references the current Zenith.NET source tree directly. Clone both repositories into the same parent directory:

```text
qian-o/
|-- Zenith.NET/
`-- ZenithTutorials/
```

Until the redesigned RHI is merged into `master`, use its matching branch:

```powershell
git clone --branch refactor/rhi-redesign https://github.com/qian-o/Zenith.NET.git
git clone https://github.com/qian-o/ZenithTutorials.git
```

## Build and Run

Install the .NET 10 SDK, then run:

```powershell
dotnet build .\ZenithTutorials\ZenithTutorials.slnx
dotnet run --project .\ZenithTutorials\ZenithTutorials\ZenithTutorials.csproj
```

Select a tutorial from the console menu. The shared application host creates the window, graphics context, swap chain, and command buffers; each renderer records one workload.

## Tutorials

| Tutorial | Renderer | Shader |
| --- | --- | --- |
| Project Setup | [ClearRenderer.cs](ZenithTutorials/Renderers/ClearRenderer.cs) | - |
| Hello Triangle | [HelloTriangleRenderer.cs](ZenithTutorials/Renderers/HelloTriangleRenderer.cs) | [HelloTriangle.slang](ZenithTutorials/Assets/Shaders/HelloTriangle.slang) |
| Textured Quad | [TexturedQuadRenderer.cs](ZenithTutorials/Renderers/TexturedQuadRenderer.cs) | [TexturedQuad.slang](ZenithTutorials/Assets/Shaders/TexturedQuad.slang) |
| Spinning Cube | [SpinningCubeRenderer.cs](ZenithTutorials/Renderers/SpinningCubeRenderer.cs) | [SpinningCube.slang](ZenithTutorials/Assets/Shaders/SpinningCube.slang) |
| Image Processing | [ComputeShaderRenderer.cs](ZenithTutorials/Renderers/ComputeShaderRenderer.cs) | [ComputeShader.slang](ZenithTutorials/Assets/Shaders/ComputeShader.slang) |
| Indirect Drawing | [IndirectDrawingRenderer.cs](ZenithTutorials/Renderers/IndirectDrawingRenderer.cs) | [IndirectDrawing.slang](ZenithTutorials/Assets/Shaders/IndirectDrawing.slang) |
| Ray Tracing | [RayTracingRenderer.cs](ZenithTutorials/Renderers/RayTracingRenderer.cs) | [RayTracing.slang](ZenithTutorials/Assets/Shaders/RayTracing.slang) |
| Mesh Shading | [MeshShadingRenderer.cs](ZenithTutorials/Renderers/MeshShadingRenderer.cs) | [MeshShading.slang](ZenithTutorials/Assets/Shaders/MeshShading.slang) |

Shared host sources:

- [Program.cs](ZenithTutorials/Program.cs)
- [App.cs](ZenithTutorials/App.cs)
- [IRenderer.cs](ZenithTutorials/IRenderer.cs)
- [CocoaHelper.cs](ZenithTutorials/CocoaHelper.cs)
- [Usings.cs](ZenithTutorials/Usings.cs)
