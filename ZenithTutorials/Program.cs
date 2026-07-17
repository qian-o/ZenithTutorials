using ZenithTutorials;
using ZenithTutorials.Renderers;

(string Name, string Slug, Action Run, Action<string> Capture)[] tutorials =
[
    ("Project Setup", "project-setup", App.Run<ClearRenderer>,
        path => App.Capture<ClearRenderer>(path, 0.0)),
    ("Hello Triangle", "hello-triangle", App.Run<HelloTriangleRenderer>,
        path => App.Capture<HelloTriangleRenderer>(path, 0.0)),
    ("Textured Quad", "textured-quad", App.Run<TexturedQuadRenderer>,
        path => App.Capture<TexturedQuadRenderer>(path, 0.0)),
    ("Spinning Cube", "spinning-cube", App.Run<SpinningCubeRenderer>,
        path => App.Capture<SpinningCubeRenderer>(path, 1.0)),
    ("Image Processing", "compute-shader", App.Run<ComputeShaderRenderer>,
        path => App.Capture<ComputeShaderRenderer>(path, 0.0)),
    ("Indirect Drawing", "indirect-drawing", App.Run<IndirectDrawingRenderer>,
        path => App.Capture<IndirectDrawingRenderer>(path, 1.0)),
    ("Ray Tracing", "ray-tracing", App.Run<RayTracingRenderer>,
        path => App.Capture<RayTracingRenderer>(path, 4.0)),
    ("Mesh Shading", "mesh-shading", App.Run<MeshShadingRenderer>,
        path => App.Capture<MeshShadingRenderer>(path, 0.0))
];

if (args is ["--capture", string slug, "--output", string output])
{
    try
    {
        if (slug == "all")
        {
            string outputDirectory = Path.GetFullPath(output);

            foreach ((string Name, string Slug, Action Run, Action<string> Capture) tutorial in
                     tutorials.Where(item => item.Slug != "project-setup"))
            {
                tutorial.Capture(Path.Combine(outputDirectory, $"{tutorial.Slug}.png"));
            }
        }
        else
        {
            (string Name, string Slug, Action Run, Action<string> Capture) tutorial =
                tutorials.Single(item => item.Slug == slug);
            tutorial.Capture(Path.GetFullPath(output));
        }
    }
    finally
    {
        App.Shutdown();
    }

    return;
}

for (int index = 0; index < tutorials.Length; index++)
{
    Console.WriteLine($"{index + 1}. {tutorials[index].Name}");
}

Console.Write("Select a tutorial to run: ");

if (int.TryParse(Console.ReadKey().KeyChar.ToString(), out int choice) &&
    choice >= 1 &&
    choice <= tutorials.Length)
{
    Console.WriteLine($"\nRunning '{tutorials[choice - 1].Name}' tutorial...");

    tutorials[choice - 1].Run();
}
