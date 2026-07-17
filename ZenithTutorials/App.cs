using Silk.NET.Windowing;
using Zenith.NET.DirectX12;
using Zenith.NET.Metal;
using Zenith.NET.Vulkan;

namespace ZenithTutorials;

internal static class App
{
    private const uint CaptureWidth = 1280;
    private const uint CaptureHeight = 720;

    private static IWindow? window;
    private static SwapChain? swapChain;

    private static uint width = CaptureWidth;
    private static uint height = CaptureHeight;

    static App()
    {
        if (!OperatingSystem.IsWindows() && !OperatingSystem.IsMacOS() && !OperatingSystem.IsLinux())
        {
            throw new PlatformNotSupportedException("The tutorials support Windows, macOS, and Linux.");
        }

        if (OperatingSystem.IsWindows())
        {
            Context = GraphicsContext.CreateDirectX12(useValidationLayer: true);
        }
        else if (OperatingSystem.IsMacOS())
        {
            Context = GraphicsContext.CreateMetal(useValidationLayer: true);
        }
        else
        {
            Context = GraphicsContext.CreateVulkan(useValidationLayer: true);
        }

        Context.ValidationMessage += static (_, args) => Console.WriteLine($"[{args.Severity}] {args.Message}");
    }

    public static GraphicsContext Context { get; }

    public static PixelFormat ColorFormat => PixelFormat.B8G8R8A8UNorm;

    public static uint Width => width;

    public static uint Height => height;

    public static string ShaderPath(string file)
    {
        return Path.Combine(AppContext.BaseDirectory, "Assets", "Shaders", file);
    }

    public static void Run<TRenderer>() where TRenderer : IRenderer, new()
    {
        window = Window.Create(WindowOptions.Default with
        {
            Size = new((int)CaptureWidth, (int)CaptureHeight),
            API = GraphicsAPI.None,
            Title = "Zenith.NET Tutorials"
        });

        window.Initialize();
        window.Center();
        UpdateDrawableSize();

        Surface surface;
        if (OperatingSystem.IsWindows())
        {
            surface = Surface.Win32(window.Native!.Win32!.Value.Hwnd, Width, Height);
        }
        else if (OperatingSystem.IsMacOS())
        {
            surface = Surface.Apple(CocoaHelper.CreateLayer(window.Native!.Cocoa!.Value), Width, Height);
        }
        else
        {
            surface = Surface.Xlib(
                window.Native!.X11!.Value.Display,
                (nint)window.Native.X11.Value.Window,
                Width,
                Height);
        }

        swapChain = Context.CreateSwapChain(new()
        {
            Surface = surface,
            Format = ColorFormat
        });
        try
        {
            using TRenderer renderer = new();

            window.Update += delta =>
            {
                if (Width is 0 || Height is 0)
                {
                    return;
                }

                renderer.Update(delta);
            };

            window.Render += _ =>
            {
                if (Width is 0 || Height is 0)
                {
                    return;
                }

                Texture drawable = swapChain.Drawable;
                CommandBuffer commandBuffer = Context.GraphicsQueue.CommandBuffer();

                renderer.Render(commandBuffer, drawable);
                commandBuffer.Transition(drawable, default, TextureLayout.ColorAttachment, TextureLayout.Present);
                commandBuffer.Submit().Wait();

                swapChain.Present();
            };

            window.Resize += _ =>
            {
                UpdateDrawableSize();

                if (Width is 0 || Height is 0)
                {
                    return;
                }

                renderer.Resize(Width, Height);
                swapChain.Resize(Width, Height);
            };

            window.Run();
        }
        finally
        {
            swapChain.Dispose();
            window.Dispose();
            Context.Dispose();
        }
    }

    public static void Capture<TRenderer>(string filePath, double elapsedTime) where TRenderer : IRenderer, new()
    {
        using Texture drawable = Context.CreateTexture(new()
        {
            Type = TextureType.Texture2D,
            Format = ColorFormat,
            Width = CaptureWidth,
            Height = CaptureHeight,
            Depth = 1,
            MipLevels = 1,
            ArrayLayers = 1,
            SampleCount = SampleCount.Count1,
            Usages = TextureUsages.Sampled | TextureUsages.ColorAttachment | TextureUsages.TransferSrc
        });
        using TRenderer renderer = new();

        renderer.Update(elapsedTime);

        CommandBuffer commandBuffer = Context.GraphicsQueue.CommandBuffer();
        renderer.Render(commandBuffer, drawable);
        ScreenCapture.CaptureToFile(commandBuffer, drawable, filePath);
    }

    public static void Shutdown()
    {
        Context.Dispose();
    }

    private static void UpdateDrawableSize()
    {
        width = (uint)window!.FramebufferSize.X;
        height = (uint)window.FramebufferSize.Y;
    }
}