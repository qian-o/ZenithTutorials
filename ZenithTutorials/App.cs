using Silk.NET.Windowing;
using Zenith.NET.DirectX12;
using Zenith.NET.Metal;
using Zenith.NET.Vulkan;

namespace ZenithTutorials;

internal static class App
{
    private static readonly IWindow window;

    static App()
    {
        // Ensure platform is supported
        if (!OperatingSystem.IsWindows() && !OperatingSystem.IsMacOS() && !OperatingSystem.IsLinux())
        {
            throw new PlatformNotSupportedException("This tutorial only supports Windows, macOS, and Linux.");
        }

        // Create window with no graphics API (we manage rendering ourselves)
        window = Window.Create(WindowOptions.Default with
        {
            API = GraphicsAPI.None,
            Title = "Zenith.NET Tutorial",
            Size = new(1280, 720)
        });

        window.Initialize();

        // Create graphics context and surface based on platform
        Surface surface;
        if (OperatingSystem.IsWindows())
        {
            Context = GraphicsContext.CreateDirectX12(useValidationLayer: true);

            surface = Surface.Win32(window.Native!.Win32!.Value.Hwnd, Width, Height);
        }
        else if (OperatingSystem.IsMacOS())
        {
            Context = GraphicsContext.CreateMetal(useValidationLayer: true);

            surface = Surface.Apple(CocoaHelper.CreateLayer(window.Native!.Cocoa!.Value), Width, Height);
        }
        else
        {
            Context = GraphicsContext.CreateVulkan(useValidationLayer: true);

            surface = Surface.Xlib(window.Native!.X11!.Value.Display, (nint)window.Native.X11.Value.Window, Width, Height);
        }

        // Log validation messages for debugging
        Context.ValidationMessage += (sender, args) =>
        {
            Console.WriteLine($"[{args.Source} - {args.Severity}] {args.Message}");
        };

        // Create swap chain for double-buffered rendering
        SwapChain = Context.CreateSwapChain(new()
        {
            Surface = surface,
            ColorTargetFormat = PixelFormat.B8G8R8A8UNorm,
            DepthStencilTargetFormat = PixelFormat.D32FloatS8UInt
        });
    }

    public static GraphicsContext Context { get; }

    public static SwapChain SwapChain { get; }

    public static uint Width => (uint)window.Size.X;

    public static uint Height => (uint)window.Size.Y;

    public static void Run<TRenderer>() where TRenderer : IRenderer, new()
    {
        using TRenderer renderer = new();

        window.Update += renderer.Update;

        window.Render += delta =>
        {
            // Skip rendering when window is minimized
            if (Width <= 0 || Height <= 0)
            {
                return;
            }

            renderer.Render();
            SwapChain.Present();
        };

        window.Resize += size =>
        {
            if (Width <= 0 || Height <= 0)
            {
                return;
            }

            // Notify renderer first, then resize swap chain
            renderer.Resize(Width, Height);
            SwapChain.Resize(Width, Height);
        };

        window.Run();
    }

    public static void Cleanup()
    {
        SwapChain.Dispose();
        Context.Dispose();
        window.Dispose();
    }
}
