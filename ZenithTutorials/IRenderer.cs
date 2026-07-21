namespace ZenithTutorials;

internal interface IRenderer : IDisposable
{
    TextureLayout RequiredLayout { get; }

    void Update(double deltaTime);

    void Render(CommandBuffer commandBuffer, Texture drawable);

    void Resize(uint width, uint height);
}