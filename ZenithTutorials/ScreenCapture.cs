using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace ZenithTutorials;

internal static class ScreenCapture
{
    // tutorial:begin capture-screenshot
    public static unsafe void CaptureToFile(CommandBuffer commandBuffer, Texture texture, TextureLayout currentLayout, string filePath)
    {
        uint width = texture.Desc.Width;
        uint height = texture.Desc.Height;
        byte[] pixels = new byte[width * height * 4];

        fixed (byte* pointer = pixels)
        {
            commandBuffer.Transition(texture, default, currentLayout, TextureLayout.CopySrc);
            commandBuffer.Download(texture,
                                   default,
                                   default,
                                   new()
                                   {
                                       Width = width,
                                       Height = height,
                                       Depth = 1
                                   },
                                   new()
                                   {
                                       Pointer = (nint)pointer,
                                       SizeInBytes = (uint)pixels.Length,
                                       RowStrideInBytes = width * 4,
                                       SliceStrideInBytes = (uint)pixels.Length
                                   });

            commandBuffer.Submit().Wait();
        }

        Directory.CreateDirectory(Path.GetDirectoryName(filePath)!);

        using Image<Bgra32> image = Image.LoadPixelData<Bgra32>(pixels, (int)width, (int)height);
        image.SaveAsPng(filePath);

        Console.WriteLine($"Screenshot saved to: {filePath}");
    }
    // tutorial:end capture-screenshot
}