using System;
using Silk.NET.Windowing;
using Zenith.NET;
using Zenith.NET.DirectX12;
using Zenith.NET.Metal;
using Zenith.NET.Vulkan;

namespace ZenithTutorials;

internal static class App
{
    private static readonly IWindow window;

    static App()
    {
        // Create window with no graphics API (we manage rendering ourselves)
        window = Window.Create(WindowOptions.Default with
        {
            API = GraphicsAPI.None,
            Title = "Zenith.NET Tutorial",
            Size = new(1280, 720)
        });

        window.Initialize();

        // Select graphics backend based on platform
        if (OperatingSystem.IsWindows())
        {
            Context = GraphicsContext.CreateDirectX12(useValidationLayer: true);
        }
        else if (OperatingSystem.IsMacOS() || OperatingSystem.IsIOS())
        {
            Context = GraphicsContext.CreateMetal(useValidationLayer: true);
        }
        else
        {
            Context = GraphicsContext.CreateVulkan(useValidationLayer: true);
        }

        // Log validation messages for debugging
        Context.ValidationMessage += (sender, args) =>
        {
            Console.WriteLine($"[{args.Source} - {args.Severity}] {args.Message}");
        };

        // Create platform-specific surface for rendering
        Surface surface;
        if (OperatingSystem.IsWindows())
        {
            surface = Surface.Win32(window.Native!.Win32!.Value.Hwnd, Width, Height);
        }
        else if (OperatingSystem.IsMacOS() || OperatingSystem.IsIOS())
        {
            throw new NotImplementedException("TODO: Get CAMetalLayer from Silk.NET.Windowing");
        }
        else if (OperatingSystem.IsLinux())
        {
            surface = Surface.Xlib(window.Native!.X11!.Value.Display, (nint)window.Native.X11.Value.Window, Width, Height);
        }
        else
        {
            throw new NotImplementedException();
        }

        // Create swap chain for double-buffered rendering
        SwapChain = Context.CreateSwapChain(new()
        {
            Surface = surface,
            ColorTargetFormat = PixelFormat.R8G8B8A8UNorm,
            DepthStencilTargetFormat = PixelFormat.D24UNormS8UInt
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
