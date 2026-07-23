using Silk.NET.Windowing;
using Zenith.NET.DirectX12;
using Zenith.NET.Metal;
using Zenith.NET.Vulkan;

namespace ZenithTutorials;

internal unsafe static class App
{
    private static IWindow? window;
    private static SwapChain? swapChain;

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

        LinearClampSampler = Context.CreateSampler(SamplerDesc.LinearClamp());
    }

    public static GraphicsContext Context { get; }

    public static Sampler LinearClampSampler { get; }

    public static PixelFormat ColorFormat => PixelFormat.B8G8R8A8UNorm;

    public static uint Width => (uint)(window?.FramebufferSize.X ?? 0);

    public static uint Height => (uint)(window?.FramebufferSize.Y ?? 0);

    public static void Run<TRenderer>() where TRenderer : IRenderer, new()
    {
        try
        {
            window = Window.Create(WindowOptions.Default with
            {
                API = GraphicsAPI.None,
                Title = "Zenith.NET Tutorials"
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
                Format = ColorFormat
            });

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

                CommandBuffer commandBuffer = Context.GraphicsQueue.CommandBuffer();

                commandBuffer.Transition(swapChain.Drawable, default, TextureLayout.Undefined, TextureLayout.ColorAttachment);

                renderer.Render(commandBuffer, swapChain.Drawable);

                commandBuffer.Transition(swapChain.Drawable, default, TextureLayout.ColorAttachment, TextureLayout.Present);

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
            swapChain?.Dispose();
            window?.Dispose();

            LinearClampSampler.Dispose();
            Context.Dispose();
        }
    }

    public static Buffer LoadBuffer<T>(T[] data, BufferUsages usages) where T : unmanaged
    {
        Buffer buffer = Context.CreateBuffer(new()
        {
            SizeInBytes = (uint)(sizeof(T) * data.Length),
            StrideInBytes = (uint)sizeof(T),
            Usages = usages,
            Residency = MemoryResidency.CpuWriteOnly
        });

        fixed (T* pointer = data)
        {
            buffer.Upload(0, new()
            {
                Pointer = (nint)pointer,
                SizeInBytes = (uint)(sizeof(T) * data.Length)
            });
        }

        return buffer;
    }

    public static Texture LoadTexture(string file, bool generateMipMaps)
    {
        return Context.LoadTextureFromFile(Path.Combine(AppContext.BaseDirectory, "Assets", "Textures", file), generateMipMaps);
    }

    public static Shader LoadShader(string file, string name)
    {
        return Context.CreateShader(ZenithCompiler.CompileFromFile(Context.GraphicsApi, Path.Combine(AppContext.BaseDirectory, "Assets", "Shaders", file), name));
    }
}