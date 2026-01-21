using System;

namespace ZenithTutorials;

internal interface IRenderer : IDisposable
{
    void Update(double deltaTime);

    void Render();

    void Resize(uint width, uint height);
}