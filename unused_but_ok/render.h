#pragma once

// forward facing renderer
// has nothing to do with buffers or textures at the level of the gpu
// this renderer is super opinionated

// load all static textures from a directory and pack them into a texture atlas, map by name
// create sprite via texture paths
    // set position
// create trail
    // set position
// spawn particle

#include <string>

#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include "glm/vec4.hpp"
using namespace glm;

namespace PixelRender {

struct Sprite {
    vec2 position;
    vec4 tint;
};

struct Trail {
    vec2 position;
    float width;
};

struct ParticleCreateInfo {
    vec2 position;
    vec2 velocity;
    float rotation;
    float rotationVelocity;
    vec2 scaleInital;
    vec2 scaleFinal;
    float scaleExponent;
    vec4 colorInitial;
    vec4 colorFinal;
    float colorExponent;
    float lifetime;
};

void create();
void destroy();

void pollEvents();

void draw();

void setPixelSize(float pixelSize);
void setClearColor(vec4 clearColor);

void loadTexturesFromDirectory(const std::string& directory);
void loadTextureFromFile(const std::string& filename);

Sprite* createSprite(const std::string& textureName);
void destroySprite(Sprite* sprite);

Trail* createTrail();
void destroyTrail(Trail* trail);

void spawnParticle(ParticleCreateInfo info);

}