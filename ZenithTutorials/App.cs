using Silk.NET.Windowing;
using Zenith.NET.DirectX12;
using Zenith.NET.Metal;
using Zenith.NET.Vulkan;

namespace ZenithTutorials;

internal static class App
{
    private static readonly IWindow window;
    private static readonly SwapChain swapChain;

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

        window = Window.Create(WindowOptions.Default with
        {
            API = GraphicsAPI.None,
            Title = "Zenith.NET Tutorials",
            Size = new(1280, 720)
        });

        window.Initialize();
        window.Center();

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
            surface = Surface.Xlib(window.Native!.X11!.Value.Display, (nint)window.Native.X11.Value.Window, Width, Height);
        }

        swapChain = Context.CreateSwapChain(new()
        {
            Surface = surface,
            Format = PixelFormat.B8G8R8A8UNorm
        });
    }

    public static GraphicsContext Context { get; }

    public static PixelFormat ColorFormat => swapChain.Desc.Format;

    public static uint Width => (uint)window.FramebufferSize.X;

    public static uint Height => (uint)window.FramebufferSize.Y;

    public static string ShaderPath(string file)
    {
        return Path.Combine(AppContext.BaseDirectory, "Assets", "Shaders", file);
    }

    public static void Run<TRenderer>() where TRenderer : IRenderer, new()
    {
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
                commandBuffer.Transition(drawable, default, TextureLayout.Present);
                commandBuffer.Submit().Wait();

                swapChain.Present();
            };

            window.Resize += _ =>
            {
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
}