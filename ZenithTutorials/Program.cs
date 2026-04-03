using ZenithTutorials;
using ZenithTutorials.Renderers;

(string Name, Action Run)[] tutorials =
[
    ("Hello Triangle",   App.Run<HelloTriangleRenderer>),
    ("Textured Quad",    App.Run<TexturedQuadRenderer>),
    ("Spinning Cube",    App.Run<SpinningCubeRenderer>),
    ("Compute Shader",   App.Run<ComputeShaderRenderer>),
    ("Indirect Drawing", App.Run<IndirectDrawingRenderer>),
    ("Ray Tracing",      App.Run<RayTracingRenderer>),
    ("Mesh Shading",     App.Run<MeshShadingRenderer>),
];

for (int i = 0; i < tutorials.Length; i++)
{
    Console.WriteLine($"{i + 1}. {tutorials[i].Name}");
}

Console.Write("Select a tutorial to run: ");

if (int.TryParse(Console.ReadKey().KeyChar.ToString(), out int choice) && choice >= 1 && choice <= tutorials.Length)
{
    tutorials[choice - 1].Run();
}
