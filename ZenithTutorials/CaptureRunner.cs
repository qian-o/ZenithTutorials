using ZenithTutorials.Renderers;

namespace ZenithTutorials;

internal static class CaptureRunner
{
    public static bool TryRun(string[] args)
    {
        if (args is not ["--capture", string slug, "--output", string output])
        {
            return false;
        }

        (string Slug, Action<string> Capture)[] tutorials =
        [
            ("hello-triangle", path => App.Capture<HelloTriangleRenderer>(path, 0.0)),
            ("textured-quad", path => App.Capture<TexturedQuadRenderer>(path, 0.0)),
            ("spinning-cube", path => App.Capture<SpinningCubeRenderer>(path, 1.0)),
            ("compute-shader", path => App.Capture<ComputeShaderRenderer>(path, 0.0)),
            ("indirect-drawing", path => App.Capture<IndirectDrawingRenderer>(path, 1.0)),
            ("ray-tracing", path => App.Capture<RayTracingRenderer>(path, 4.0)),
            ("mesh-shading", path => App.Capture<MeshShadingRenderer>(path, 0.0))
        ];

        int tutorialIndex = Array.FindIndex(tutorials, item => item.Slug == slug);
        if (slug != "all" && tutorialIndex < 0)
        {
            Console.Error.WriteLine($"Unknown tutorial slug: '{slug}'.");
            Environment.ExitCode = 1;
            return true;
        }

        try
        {
            if (slug == "all")
            {
                string outputDirectory = Path.GetFullPath(output);

                foreach ((string tutorialSlug, Action<string> capture) in tutorials)
                {
                    capture(Path.Combine(outputDirectory, $"{tutorialSlug}.png"));
                }
            }
            else
            {
                tutorials[tutorialIndex].Capture(Path.GetFullPath(output));
            }
        }
        finally
        {
            App.Shutdown();
        }

        return true;
    }
}