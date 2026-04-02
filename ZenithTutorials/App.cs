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
            throw new PlatformNotSupportedException("This application only supports Windows, macOS, and Linux.");
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

        Context.ValidationMessage += static (sender, args) => Console.WriteLine($"[{args.Source} - {args.Severity}] {args.Message}");

        window = Window.Create(WindowOptions.Default with
        {
            API = GraphicsAPI.None,
            Title = "Tutorial - Zenith.NET",
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

        swapChain = Context.CreateSwapChain(new() { Surface = surface, ColorTargetFormat = PixelFormat.B8G8R8A8UNorm, DepthStencilTargetFormat = PixelFormat.D32FloatS8UInt });
    }

    public static GraphicsContext Context { get; }

    public static uint Width => (uint)window.FramebufferSize.X;

    public static uint Height => (uint)window.FramebufferSize.Y;

    public static FrameBuffer FrameBuffer => swapChain.FrameBuffer;

    public static void Run<TRenderer>() where TRenderer : IRenderer, new()
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

        window.Render += delta =>
        {
            if (Width is 0 || Height is 0)
            {
                return;
            }

            renderer.Render();
            swapChain.Present();
        };

        window.Resize += size =>
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

    public static void Cleanup()
    {
        swapChain.Dispose();
        window.Dispose();

        Context.Dispose();
    }
}
