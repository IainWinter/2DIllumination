#pragma once

struct Texture {
    int width;
    int height;
};

namespace internal {
    struct TextureInternal {
        Texture* external;
    };
}


// return a pointer to a struct which contains some data
// but also lives inside of a larger struct with more info
// this way the user can have direct access
// while the renderer can also have direct access, but not
// spill out its internals

struct SomeUserDataAboutASprite {
    float x, y;
};

// inside the renderer

struct RendererRepresentationOfASprite {
    SomeUserDataAboutASprite* userData;
    GLuint handle;
};

class Renderer {
public:
    SomeUserDataAboutASprite* createSprite() {
        SomeUserDataAboutASprite* s = new SomeUserDataAboutASprite();
        sprites.push_back({s, 0});
        return s;
    }

private:
    std::vector<RendererRepresentationOfASprite> sprites;
};