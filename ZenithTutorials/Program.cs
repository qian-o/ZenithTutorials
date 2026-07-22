using ZenithTutorials;
using ZenithTutorials.Renderers;

(string Name, Action Run)[] tutorials =
[
    ("Hello Triangle", App.Run<HelloTriangleRenderer>),
    ("Textured Quad", App.Run<TexturedQuadRenderer>),
    ("Spinning Cube", App.Run<SpinningCubeRenderer>),
    ("Image Processing", App.Run<ComputeShaderRenderer>),
    ("Indirect Drawing", App.Run<IndirectDrawingRenderer>),
    ("Ray Tracing", App.Run<RayTracingRenderer>),
    ("Mesh Shading", App.Run<MeshShadingRenderer>)
];

for (int index = 0; index < tutorials.Length; index++)
{
    Console.WriteLine($"{index + 1}. {tutorials[index].Name}");
}

Console.Write("Select a tutorial to run: ");

if (!int.TryParse(Console.ReadLine(), out int choice) ||
    choice < 1 ||
    choice > tutorials.Length)
{
    return;
}

Console.WriteLine($"Running '{tutorials[choice - 1].Name}' tutorial...");
tutorials[choice - 1].Run();
