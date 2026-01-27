using System.Runtime.CompilerServices;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Zenith.NET.DirectX12;
using Zenith.NET.Metal;
using Zenith.NET.Vulkan;

namespace ZenithTutorials;

public static class ScreenCapture
{
    public static void CaptureToFile(GraphicsContext context, FrameBuffer frameBuffer, string filePath)
    {
        byte[] pixels = CapturePixels(context, frameBuffer);
        uint width = frameBuffer.Desc.ColorAttachments[0].Target.Desc.Width;
        uint height = frameBuffer.Desc.ColorAttachments[0].Target.Desc.Height;

        SaveAsPng(pixels, (int)width, (int)height, filePath);
    }

    public static void CaptureWithDialog(GraphicsContext context, FrameBuffer frameBuffer)
    {
        if (ShowSaveFileDialog() is string filePath)
        {
            CaptureToFile(context, frameBuffer, filePath);

            Console.WriteLine($"Screenshot saved to: {filePath}");
        }
    }

    public static byte[] CapturePixels(GraphicsContext context, FrameBuffer frameBuffer)
    {
        Texture colorTarget = frameBuffer.Desc.ColorAttachments[0].Target;
        uint width = colorTarget.Desc.Width;
        uint height = colorTarget.Desc.Height;
        uint rowPitch = ZenithHelper.Align(width * 4u, GraphicsContext.TextureRowPitchAlignment);

        using Buffer buffer = context.CreateBuffer(new()
        {
            SizeInBytes = rowPitch * height,
            StrideInBytes = 4,
            Flags = BufferUsageFlags.MapRead
        });

        CommandBuffer cmd = context.Copy.CommandBuffer();

        cmd.CopyTextureToBuffer(colorTarget,
                                default,
                                default,
                                new() { Width = width, Height = height, Depth = 1 },
                                buffer,
                                0);

        cmd.Submit(true);

        byte[] pixels = new byte[width * height * 4];

        MappedMemory mappedMemory = buffer.Map();

        unsafe
        {
            fixed (byte* pixelsPtr = pixels)
            {
                for (int i = 0; i < height; i++)
                {
                    Unsafe.CopyBlock(pixelsPtr + (width * 4 * i), (void*)(mappedMemory.Pointer + (rowPitch * i)), width * 4);
                }
            }
        }

        buffer.Unmap();

        return pixels;
    }

    public static Image<Rgba32> CaptureToImage(GraphicsContext context, FrameBuffer frameBuffer)
    {
        byte[] pixels = CapturePixels(context, frameBuffer);
        uint width = frameBuffer.Desc.ColorAttachments[0].Target.Desc.Width;
        uint height = frameBuffer.Desc.ColorAttachments[0].Target.Desc.Height;

        return Image.LoadPixelData<Rgba32>(pixels, (int)width, (int)height);
    }

    private static string? ShowSaveFileDialog()
    {
        nint result = NativeFileDialog.GetSaveFileName(out string? filePath,
                                                       "PNG Image (*.png)\0*.png\0All Files (*.*)\0*.*\0",
                                                       "png",
                                                       $"Screenshot_{DateTime.Now:yyyyMMdd_HHmmss}");

        return result == 0 ? filePath : null;
    }

    private static void SaveAsPng(byte[] pixels, int width, int height, string filePath)
    {
        using Image<Rgba32> image = Image.LoadPixelData<Rgba32>(pixels, width, height);
        image.SaveAsPng(filePath);
    }

    private static class NativeFileDialog
    {
        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
        private struct OpenFileName
        {
            public int lStructSize;
            public nint hwndOwner;
            public nint hInstance;
            public string lpstrFilter;
            public nint lpstrCustomFilter;
            public int nMaxCustFilter;
            public int nFilterIndex;
            public nint lpstrFile;
            public int nMaxFile;
            public nint lpstrFileTitle;
            public int nMaxFileTitle;
            public string? lpstrInitialDir;
            public string? lpstrTitle;
            public int Flags;
            public short nFileOffset;
            public short nFileExtension;
            public string? lpstrDefExt;
            public nint lCustData;
            public nint lpfnHook;
            public string? lpTemplateName;
            public nint pvReserved;
            public int dwReserved;
            public int FlagsEx;
        }

        private const int OFN_OVERWRITEPROMPT = 0x00000002;
        private const int OFN_PATHMUSTEXIST = 0x00000800;
        private const int OFN_EXPLORER = 0x00080000;
        private const int MAX_PATH = 260;

        [DllImport("comdlg32.dll", EntryPoint = "GetSaveFileNameW", CharSet = CharSet.Unicode, SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
#pragma warning disable SYSLIB1054
        private static extern bool GetSaveFileNameNative(ref OpenFileName lpofn);
#pragma warning restore SYSLIB1054

        public static unsafe nint GetSaveFileName(out string? filePath, string filter, string defaultExt, string defaultFileName)
        {
            filePath = null;

            char[] fileBuffer = new char[MAX_PATH];
            defaultFileName.AsSpan().CopyTo(fileBuffer);

            fixed (char* filePtr = fileBuffer)
            {
                OpenFileName ofn = new()
                {
                    lStructSize = Marshal.SizeOf<OpenFileName>(),
                    hwndOwner = nint.Zero,
                    lpstrFilter = filter,
                    lpstrFile = (nint)filePtr,
                    nMaxFile = MAX_PATH,
                    lpstrDefExt = defaultExt,
                    lpstrTitle = "Save Screenshot",
                    Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_EXPLORER
                };

                if (GetSaveFileNameNative(ref ofn))
                {
                    filePath = new string(fileBuffer).TrimEnd('\0');
                    return 0;
                }
            }

            return 1;
        }
    }
}
