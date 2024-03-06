#pragma once

#include "glm/vec2.hpp"
using namespace glm;

// Main loop

struct Texture {
    int handle;
    int width;
    int height;
};

class Renderer {
public:
    Renderer();
    ~Renderer();

    Texture* createTexture(int width, int height, const char* data);
    void destroyTexture(Texture* texture);
};

void renderCreate();
void renderDestroy();
bool renderBeginFrame();
void renderEndFrame();

void gameCreate();
void gameDestroy();
void gamePollEvents();
bool gameBeginTick();

// Rendering objects

struct Texture {
    int handle;
    int width, height;
};

struct Sprite {
    int handle;
    int textureHandle;

    vec2 pos;
    vec2 size;

    vec2 uv;
    vec2 uvSize;
};

struct Polygon {
    int handle;
    std::vector<vec2> points;
};

Texture textureCreate(int width, int height, const char* data);
void textureDestroy(Texture texture);
void textureWrite(Texture texture, const char* data);
void textureWriteRegion(Texture texture, const char* data, int x, int y, int width, int height);

Sprite spriteCreate(Texture texture);
void spriteDestroy(Sprite sprite);

Polygon polygonCreate(const std::vector<vec2>& points);
void polygonDestroy(Polygon polygon);