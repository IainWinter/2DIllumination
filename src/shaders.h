#pragma once

#include "gl/glad.h"
#include <stdio.h>
#include <stdlib.h>

struct Shader {
    GLuint handle;
};

Shader createShader(const char* vertex, const char* fragment);
Shader createComputeShader(const char* source);

struct SpriteShader {
    Shader program;

    GLuint viewLocation;
    GLuint projLocation;
    GLuint modelLocation;
    GLuint diffuseTextureLocation;
    GLuint normalTextureLocation;
    GLuint lightPositionLocation;
    GLuint tintLocation;
};

struct VoronoiGeneratorSeedShader {
    Shader program;

    GLuint viewLocation;
    GLuint projLocation;
    GLuint modelLocation;
    GLuint diffuseTextureLocation;
};

struct VoronoiGeneratorShader {
    Shader program;

    GLuint inVoronoiLocation;
    GLuint offsetLocation;
    GLuint resolutionLocation;
};

struct DistanceFieldFromVoronoiShader {
    Shader program;

    GLuint inVoronoiLocation;
    GLuint resolutionLocation;
    GLuint distanceScaleLocation;
};

struct RayMarchShader {
    Shader program;
    
    GLuint inDistanceFieldLocation;
    GLuint inSceneTextureLocation;

    GLuint resolutionLocation;
    GLuint timeLocation;

    GLuint maxStepsLocation;
    GLuint raysPerPixelLocation;

    GLuint distanceScaleLocation;
    GLuint emissiveScaleLocation;

    GLuint angleDebugLocation;
    GLuint offsetDebugLocation;
};

struct DrawTextureShader {
    Shader program;

    GLuint textureLocation;
};

SpriteShader createSpriteShader();
VoronoiGeneratorSeedShader createVoronoiGeneratorSeedShader();
VoronoiGeneratorShader createVoronoiGeneratorShader();
DistanceFieldFromVoronoiShader createDistanceFieldFromVoronoiShader();
RayMarchShader createRayMarchShader();
DrawTextureShader createDrawTextureShader();