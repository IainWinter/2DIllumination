#pragma once

#include "glm/vec2.hpp"
#include "glm/vec4.hpp"
using namespace glm;

#include "gl/glad.h"

struct InputState {
    bool running;

    vec2 mouseScreenPosition;
    vec2 lastMousePressedScreenPosition;
    float mouseWheel;
    bool isMousePressed;
    bool isMousePressedLast;

    float keysDown[512];
};

struct RenderCreateInfo {
    int pixelScale;
};

struct Sprite {
    vec2 position;
    float rotation;
    vec2 size;
    vec4 color;

    GLuint diffuse;
    GLuint normal;

    vec2 velocity;
    float life;
};

GLuint createTextureFromFile(const char* filename);

void renderCreate(int width, int height);
void renderDestroy();

void render();

void pollEvents(InputState* pInput);

void addSprite(const Sprite& sprite);