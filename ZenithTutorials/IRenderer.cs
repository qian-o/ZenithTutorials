namespace ZenithTutorials;

// tutorial:begin renderer-contract
internal interface IRenderer : IDisposable
{
    TextureLayout RequiredLayout { get; }

    void Update(double deltaTime);

    void Render(CommandBuffer commandBuffer, Texture drawable);

    void Resize(uint width, uint height);
}
// tutorial:end renderer-contract