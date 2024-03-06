#pragma once

#include "gl/glad.h"

#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include "glm/vec4.hpp"
#include "glm/mat2x2.hpp"
#include "glm/mat3x3.hpp"
using namespace glm;

#include <vector>

// Make a chunked sprite renderer

// Each Texture:
//    - width
//    - height
//    - handle

// Each TextureView
//    - texture
//    - uv pos
//    - uv size
//    - pixel pos
//    - pixel size

// Each Sprite:
//    - texture view
//    - tint
//    - position
//    - rotation
//    - scale

// Each CellSprite:
//    - list of textures
//    - list of tiles
//    - bounding box for broad phase

// Each CellTile:
//    - texture id
//    - uv pos
//    - uv size
//    - offset
//    - triangle list for collision
//    - bounding box for board phase
//    - an array of data per pixel for the game play code to use for the damage model

// Each Particle:
//    - a bunch of options
//    - I found no need for textures if a large count is used
//      especially if the particles are small and in 3d spaces

// Each Line:
//    - start position
//    - end position
//    - width
//    - width ratio to end
//    - color

// The renderer should be a retained mode.
// I dont want to be submitting draws to it, just creating objects inside of it

// Rendering:
//     - Upload texture changes
//     - Calculate radiance
//     - Calculate shadow map
//     - Cull on screen objects
//     - Draw sprites
//     - Draw cell sprites
//     - Draw particles

// How are sprites loaded
//    Provided to the renderer in binary format
//    Only support RGBA8

// How are random asteroids generated:
//    Single time algos, generate a batch at load time

//
//  Rendering composition types
//

struct Transform2D {
    vec2 position;
    float rotation;
    vec2 scale;
};

struct Transform3D {
    vec3 position;
    vec3 rotation;
    vec2 scale;
};

struct BoundingBox {
    vec2 min;
    vec2 max;
};

struct Triangle {
    vec2 a;
    vec2 b;
    vec2 c;
};

//
//  Rendering objects
//

struct Texture {
    int width;
    int height;
    GLuint handle;
};

struct TextureView {
    Texture* texture;
    vec2 uvPos;
    vec2 uvSize;
    ivec2 pixelPos;
    ivec2 pixelSize;
};

struct Shader {
    GLuint handle;
};

struct Sprite {
    TextureView* texture;
    vec4 tint;
    Transform2D transform;
};

struct Trail {
    vec2 start;
    vec2 end;
    float width;
    float widthRatioToEnd;
    vec4 color;
};

struct Particle {
    Transform3D transform;
    vec3 velocity;
    vec3 acceleration;
    vec4 color;
    float life;
    float lifeTotal;
};

struct CellSprite {
    struct Cell {
        float health;
    };

    struct Tile {
        TextureView* texture;
        BoundingBox boundingBox;
        vec2 offset;
        ivec2 pixelOffset;
        std::vector<Triangle> triangleList;
        std::vector<vec4> cells;
    };

    Transform2D transform;
    std::vector<TextureView*> textures;
    std::vector<Tile> tiles;
    BoundingBox boundingBox;
};

// make basically a C style api because these rendering libraries are always in C

namespace render {
    void create();
    void destroy();

    void pollEvents();
    void draw(int tick);

    int instCount();

    Shader* shaderCreate(const char* vertexSource, const char* fragmentSource);
    void shaderDestroy(Shader* shader);

    Texture* textureCreateEmpty(int width, int height);
    Texture* textureCreate(int width, int height, const char* pixels);
    void textureDestroy(Texture* texture);

    TextureView* textureViewCreate(Texture* texture, const vec2& uvPos, const vec2& uvSize, const ivec2& pixelPos, const ivec2& pixelSize);
    void textureViewDestroy(TextureView* textureView);

    Sprite* spriteCreate(TextureView* view, const vec4& tint, const Transform2D& transform);
    void spriteDestroy(Sprite* sprite);
}