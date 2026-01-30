using System.Runtime.InteropServices;

namespace ZenithTutorials;

internal static unsafe partial class CocoaHelper
{
    private const string LibObjC = "/usr/lib/libobjc.A.dylib";

    [LibraryImport(LibObjC, EntryPoint = "objc_getClass", StringMarshalling = StringMarshalling.Utf8)]
    private static partial nint GetClass(string name);

    [LibraryImport(LibObjC, EntryPoint = "sel_registerName", StringMarshalling = StringMarshalling.Utf8)]
    private static partial nint Selector(string name);

    [LibraryImport(LibObjC, EntryPoint = "objc_msgSend")]
    private static partial nint Send(nint receiver, nint selector);

    [LibraryImport(LibObjC, EntryPoint = "objc_msgSend")]
    private static partial nint Send(nint receiver, nint selector, [MarshalAs(UnmanagedType.I1)] bool arg);

    [LibraryImport(LibObjC, EntryPoint = "objc_msgSend")]
    private static partial nint Send(nint receiver, nint selector, nint arg);

    public static nint CreateLayer(nint cocoaWindow)
    {
        nint nsView = Send(cocoaWindow, Selector("contentView"));

        Send(nsView, Selector("setWantsLayer:"), true);

        nint layer = Send(GetClass("CAMetalLayer"), Selector("layer"));

        Send(layer, Selector("retain"));
        Send(nsView, Selector("setLayer:"), layer);

        return layer;
    }
}
