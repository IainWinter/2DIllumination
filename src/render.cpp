#include "render.h"
#include "clock.h"
#include "ring.h"

#include "gl/glad.h"

#include "glm/vec2.hpp"
#include "glm/mat4x4.hpp"
#include "glm/gtc/matrix_transform.hpp"
using namespace glm;

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_sdl2.h"
#include "imgui/imgui_impl_opengl3.h"

#include "SDL2/SDL.h"
#include "SDL2/SDL_opengl.h"

#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

void errorCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam);

// Shader types

struct GBufferShader {
    GLuint handle;
    GLuint viewLocation;
    GLuint projLocation;
    GLuint modelLocation;
    GLuint diffuseTextureLocation;
    GLuint normalTextureLocation;
    GLuint tintLocation;
    GLuint lightPositionLocation;
};

struct VoronoiGeneratorSeedShader {
    GLuint handle;
    GLuint viewLocation;
    GLuint projLocation;
    GLuint modelLocation;
    GLuint diffuseTextureLocation;
    GLuint tintLocation;
};

struct VoronoiGeneratorShader {
    GLuint handle;
    GLuint inVoronoiLocation;
    GLuint offsetLocation;
    GLuint resolutionLocation;
};

struct DistanceFieldFromVoronoiShader {
    GLuint handle;
    GLuint inVoronoiLocation;
    GLuint resolutionLocation;
    GLuint distanceScaleLocation;
};

struct RayMarchShader {
    GLuint handle;
    GLuint inDistanceFieldLocation;
    GLuint inSceneDiffuseLocation;
    GLuint inSceneNormalLocation;
    GLuint inLastEmissiveTextureLocation;
    GLuint inAccumulatedEmissiveTextureLocation;

    GLuint bounceLightEnabledLocation;
    GLuint bounceLightDampeningLocation;

    GLuint resolutionLocation;
    GLuint timeLocation;

    GLuint maxStepsLocation;
    GLuint raysPerPixelLocation;

    GLuint distanceScaleLocation;
    GLuint emissiveScaleLocation;
};

struct GaussianBlurShader {
    GLuint handle;
    GLuint blurScaleLocation;
    GLuint textureResolutionLocation;
    GLuint textureLocation;
};

struct DrawTextureShader {
    GLuint handle;
    GLuint textureLocation;
};

struct QuadVertex {
    vec2 pos;
    vec2 uv;
};

struct ScreenQuadMesh {
    GLuint vertexBuffer;
    GLuint indexBuffer;
    GLuint vertexArray;
};

struct SpriteInstanceMesh {
    GLuint vertexBuffer;
    GLuint indexBuffer;
    GLuint vertexArray;
};

// End shader types

struct RenderTexturePair {
    GLuint framebuffer;
    GLuint texture;
};

// State

SDL_Window* r_window;
SDL_GLContext r_opengl;

int r_screenWidth;
int r_screenHeight;
float r_aspect;

int r_rayMarchWidth;
int r_rayMarchHeight;

float r_cameraHeight;
vec2 r_cameraSize;

mat4 r_view;
mat4 r_projection;

GBufferShader r_gBufferShader;
VoronoiGeneratorShader r_voronoiGeneratorShader;
DistanceFieldFromVoronoiShader r_distanceFieldFromVoronoiShader;
DrawTextureShader r_drawTextureShader;
RayMarchShader r_rayMarchShader;
GaussianBlurShader r_gaussianBlurShader;

GLuint r_gBufferFramebuffer;

GLuint r_voronoiTexture;
GLuint r_voronoiFramebuffer;
GLuint r_voronoiTexture2;
GLuint r_voronoiFramebuffer2;
GLuint r_distanceFieldTexture;
GLuint r_distanceFieldFramebuffer;
GLuint r_gDiffuseTexture;
GLuint r_gNormalTexture;
GLuint r_gDepthTexture;
GLuint r_accumulationEmissiveTexture;
GLuint r_globalEmissiveTexture;
GLuint r_globalEmissiveFramebuffer;
GLuint r_globalEmissiveTexture2;
GLuint r_globalEmissiveFramebuffer2;
GLuint r_emissiveBlurredTexture;
GLuint r_emissiveBlurredFramebuffer;

Ring<RenderTexturePair> r_globalEmissiveRing;

int emissiveFramebufferIndex = 0;

SpriteInstanceMesh r_spriteInstanceMesh;
ScreenQuadMesh r_screenQuadMesh;

Clock r_clock;

// random shit

int limitPasses = 11;
int maxSteps = 12;
int raysPerPixel = 1;
float distanceScale = 1.0f;
float emissiveScale = 1.0f;
int layerIndex = 4;
bool bounceLightEnabled = true;
float bounceLightDampening = 1.f;
vec3 lightPosition = vec3(0, 0, 1);
vec2 blurAmount = vec2(0, 0);

std::vector<Sprite> sprites;

// End State

// Shader code

const char* screenQuadVertex = R"(
    #version 330 core

    layout(location = 0) in vec2 vert_pos;
    layout(location = 1) in vec2 vert_uv;

    out vec2 frag_uv;

    void main() {
        frag_uv = vert_uv;
        gl_Position = vec4(vert_pos, 0.0, 1.0);
    }
)";

const char* spriteVertex = R"(
    #version 330 core
    
    layout(location = 0) in vec2 vert_pos;
    layout(location = 1) in vec2 vert_uv;

    out vec2 frag_uv;
    out vec2 frag_screenPosition;
    out vec3 frag_worldPosition;
    out mat3 frag_normalMatrix;

    uniform mat4 u_view;
    uniform mat4 u_proj;
    uniform mat4 u_model;

    void main() {
        vec4 worldPos = u_model * vec4(vert_pos, 0.0, 1.0);
        vec4 clipPos = u_proj * u_view * worldPos;

        frag_uv = vert_uv;
        frag_screenPosition = (clipPos.xy / clipPos.w + 1.0) / 2.0;
        frag_worldPosition = worldPos.xyz;
        frag_normalMatrix = transpose(inverse(mat3(u_model)));
        gl_Position = clipPos;
    }
)";

const char* spriteFragment = R"(
    #version 330 core

    in vec2 frag_uv;
    in vec2 frag_screenPosition;
    in vec3 frag_worldPosition;
    in mat3 frag_normalMatrix;

    layout (location = 0) out vec4 final_color;
    layout (location = 1) out vec4 final_normal;
    layout (location = 2) out vec4 final_screenPosition;

    uniform sampler2D u_diffuse;
    uniform sampler2D u_normal;
    uniform vec3 u_lightPosition;

    uniform vec4 u_tint;

    void main() {
        vec4 color = u_tint * texture(u_diffuse, frag_uv);
        vec3 normalRaw = texture(u_normal, frag_uv).rgb;

        vec3 normal = normalRaw * 2.0 - 1.0;
        normal += vec3(0, 0, 1); // bias to pointing up
        normal = normalize(frag_normalMatrix * normal);

        // not sure if this is needed cus the texture is storing floats
        normalRaw = (normal + 1) * 0.5;

        // Doesn't work well for flashlight

        //vec3 lightDir = normalize(u_lightPosition - frag_worldPosition);
        //vec3 viewDir = vec3(0, 0, 1);
        //vec3 halfDir = normalize(lightDir + viewDir);

        float intensity = 1;//max(dot(normal, halfDir), 0.0);

        if (color.a == 0) {
            discard;
        }

        final_color = vec4(color.rgb * intensity, color.a);
        final_normal = vec4(normalRaw, 1);
        final_screenPosition = vec4(frag_screenPosition, 0, 1);
    }
)";

const char* voronoiGeneratorFragment = R"(
    #version 330 core

    in vec2 frag_uv;
    out vec4 final_color;

    uniform sampler2D u_in;

    uniform int u_offset;
    uniform ivec2 u_resolution;

    void main() {
        float closestDist = 9999999.0;
        vec2 closestPoint = vec2(0.0);

        for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            vec2 offset = frag_uv + vec2(x, y) / vec2(u_resolution) * float(u_offset);

            // if clamp to border isn't working
            if (   offset.x < 0 || offset.x > 1.0
                || offset.y < 0 || offset.y > 1.0)
            {
                continue;
            }

            vec2 objScreenUv = texture(u_in, offset).xy;
            float dist = distance(frag_uv, objScreenUv);

            if (objScreenUv.x != 0.0 && objScreenUv.y != 0 && dist < closestDist) {
                closestDist = dist;
                closestPoint = objScreenUv;
            }
        }
        }

        final_color = vec4(closestPoint, 0, 1);
    }
)";

const char* distanceFieldFromVoronoiFragment = R"(
    #version 330 core

    in vec2 frag_uv;
    out vec4 final_color;

    uniform sampler2D u_inVoronoi;

    uniform ivec2 u_resolution;
    uniform float u_distanceScale;

    void main() {
        vec2 screenUv = frag_uv;

        vec2 fragScreenUv = texture(u_inVoronoi, frag_uv).xy;
        float dist = distance(screenUv, fragScreenUv);
        float mapped = clamp(dist * u_distanceScale, 0.0, 1.0);

        final_color = vec4(vec3(mapped), 1.0);
    }
)";

const char* rayMarchFragment = R"(
    #version 330 core

    in vec2 frag_uv;
    
    layout (location = 0) out vec4 final_color;
    layout (location = 1) out vec4 acc_color;

    uniform sampler2D u_inDistanceField;
    uniform sampler2D u_inGDiffuseTexture;
    uniform sampler2D u_inGNormalTexture;
    uniform sampler2D u_lastEmissiveTexture; // last texture should rename
    uniform sampler2D r_accumulationEmissiveTexture;

    uniform ivec2 u_resolution;
    uniform float u_time;
    uniform int u_maxSteps;
    uniform int u_raysPerPixel;
    uniform float u_distanceScale;
    uniform float u_emissiveScale;

    uniform float u_bounceLightEnabled;
    uniform float u_bounceLightDampening;

    float epsilon() {
        return 0.5 / max(u_resolution.x, u_resolution.y);
    }

    float minStepDistance() {
        return 1.0 / min(u_resolution.x, u_resolution.y);
    }

    bool rayMarch(vec2 origin, vec2 dir, out vec2 hitPos, out float rayDistance) {
        float currentDistance = 0.0;
        for (int i = 0; i < u_maxSteps; i++) {
            vec2 point = origin + dir * currentDistance;

            if (point.x < 0 || point.x >= 1 || point.y < 0 || point.y >= 1) {
                return false;
            }

            float distance = texture(u_inDistanceField, point).r / u_distanceScale;

            if (distance < epsilon()) {
                hitPos = point;
                rayDistance = currentDistance;
                return true;
            }

            distance = max(distance, minStepDistance());
            currentDistance += distance;
        }

        return false;
    }

    float random (vec2 st) {
        return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
    }

    vec3 lin_to_srgb(vec3 color) {
        vec3 x = color * 12.92;
        vec3 y = 1.055 * pow(clamp(color, 0.0, 1.0), vec3(0.4166667)) - 0.055;
        vec3 clr = color;
        clr.r = (color.r < 0.0031308) ? x.r : y.r;
        clr.g = (color.g < 0.0031308) ? x.g : y.g;
        clr.b = (color.b < 0.0031308) ? x.b : y.b;
        return clr;
    }

    void main() {
        float pixelEmissive = 0.0;
        vec3 pixelColor = vec3(0.0);

        float PI = 3.141596;

        // get the direction of a pixel, aka the normal
        // if 0, move in all direction

        float rand2PI = random(frag_uv * vec2(u_time, -u_time)) * 2.0 * PI;
        float goldenAngle = PI * 0.7639320225; // magic number that gives us a good ray distribution.

        for (int i = 0; i < u_raysPerPixel; i++) {
            float angle = rand2PI + goldenAngle * float(i);
            vec2 dir = vec2(cos(angle), sin(angle));
            vec2 origin = frag_uv;

            vec2 rayHitPos;
            float rayDistance;
            bool rayHit = rayMarch(origin, dir, rayHitPos, rayDistance);
            if (rayHit) {
                vec2 delta = 1.0 / vec2(u_resolution);
                
                vec4 hitPixelColorRaw = texture(u_inGDiffuseTexture, rayHitPos);
                vec4 hitPixelNormalRaw = texture(u_inGNormalTexture, rayHitPos);

                vec3 normal = hitPixelNormalRaw.rgb * 2.0 - 1.0;
                vec3 color = hitPixelColorRaw.rgb;
                float emissive = max(color.r, max(color.g, color.b)) * u_emissiveScale;

                float lastEmission = 0.0;
                vec3 lastColor = vec3(0.0);

                if (u_bounceLightEnabled > 0 && rayDistance > epsilon() && emissive < epsilon()) {
                    vec4 samplePixel = texture(u_lastEmissiveTexture, rayHitPos);

                    if (samplePixel.a > lastEmission) {
                        lastEmission = samplePixel.a;
                        lastColor = samplePixel.rgb;
                    }
                }

                float pointed = 1;
                // if (hitPixelNormalRaw.rgb != vec3(0)) { 
                //     pointed = clamp(dot(normal, vec3(dir, 1)), 0, 1);
                // }

                pixelEmissive += emissive * pointed + lastEmission * u_bounceLightDampening;
                pixelColor    += color + lastColor * pointed;
            }
        }

        pixelColor /= pixelEmissive;
        pixelEmissive /= float(u_raysPerPixel);

        vec4 color = vec4(lin_to_srgb(pixelColor * pixelEmissive), pixelEmissive);

        final_color = color;
        acc_color = (color + texture(r_accumulationEmissiveTexture, frag_uv)); // average also decay, should be based on time
    }
)";

const char* copyTextureFragment = R"(
    #version 330 core

    in vec2 frag_uv;

    out vec4 final_color;

    uniform sampler2D u_texture;

    void main() {
        final_color = texture(u_texture, frag_uv);
    }
)";

const char* gaussianFragment = R"(
    #version 330 core

    in vec2 frag_uv;

    out vec4 final_color;

    uniform vec2 u_blurScale;
    uniform vec2 u_textureResolution;
    uniform sampler2D u_texture;

    uniform float u_scale = 1.94819;

    const int blurRad = 64;
    const float weight[64] = float[] (0.026597, 0.026538, 0.026361, 0.026070, 0.025668, 0.025159, 0.024552, 0.023853,
                                      0.023071, 0.022215, 0.021297, 0.020326, 0.019313, 0.018269, 0.017205, 0.016132,
                                      0.015058, 0.013993, 0.012946, 0.011924, 0.010934, 0.009982, 0.009072, 0.008209,
                                      0.007395, 0.006632, 0.005921, 0.005263, 0.004658, 0.004104, 0.003599, 0.003143,
                                      0.002733, 0.002365, 0.002038, 0.001748, 0.001493, 0.001269, 0.001075, 0.000906,
                                      0.000760, 0.000635, 0.000528, 0.000437, 0.000360, 0.000295, 0.000241, 0.000196,
                                      0.000159, 0.000128, 0.000103, 0.000082, 0.000065, 0.000052, 0.000041, 0.000032,
                                      0.000025, 0.000019, 0.000015, 0.000012, 0.000009, 0.000007, 0.000005, 0.000004);

    void main() {
        vec2 fragSize = 1.0 / u_textureResolution;

        vec4 color = texture(u_texture, frag_uv)  * weight[0] * u_scale;

        for (int i = 1; i < blurRad; i++) {
            color = color
                  + texture(u_texture, frag_uv + fragSize * float(i) * u_blurScale) * weight[i] * u_scale;
                  + texture(u_texture, frag_uv - fragSize * float(i) * u_blurScale) * weight[i] * u_scale;
        }

        final_color = color;
    }
)";

// End Shader code

void printWithLineNumbers(const char* source) {
    // get total number of newlines

    int totalLineCount = 0;
    const char* itr = source;
    while (*itr != '\0') {
        totalLineCount += 1;
        itr += 1;
    }

    int digitCount = (int)log10(totalLineCount) + 1;

    std::istringstream stream(source);

    std::string line;
    int lineNumber = 0;

    while (std::getline(stream, line)) {
        printf("%*d| %s\n", digitCount, lineNumber, line.c_str());
        lineNumber += 1;
    }
}

GLuint createShader(const char* vertex, const char* fragment) {
    GLuint shader = glCreateProgram();
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(vertexShader, 1, &vertex, NULL);
    glCompileShader(vertexShader);

    // check status
    GLint status;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE) {
        GLint length;
        glGetShaderiv(vertexShader, GL_INFO_LOG_LENGTH, &length);
        char* log = (char*)malloc(length);
        glGetShaderInfoLog(vertexShader, length, &length, log);
        printf("Vertex shader error: %s\n", log);
        printWithLineNumbers(vertex);
        free(log);
        throw nullptr;
    }

    glShaderSource(fragmentShader, 1, &fragment, NULL);
    glCompileShader(fragmentShader);

    // check status
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE) {
        GLint length;
        glGetShaderiv(fragmentShader, GL_INFO_LOG_LENGTH, &length);
        char* log = (char*)malloc(length);
        glGetShaderInfoLog(fragmentShader, length, &length, log);
        printf("Fragment shader error: %s\n", log);
        printWithLineNumbers(fragment);
        free(log);
        throw nullptr;
    }

    glAttachShader(shader, vertexShader);
    glAttachShader(shader, fragmentShader);
    glLinkProgram(shader);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shader;
}

GLuint createTextureFromFile(const char* filename) {
    int width, height, channels;
    uint8_t* pixels = stbi_load(filename, &width, &height, &channels, 0);

    GLenum formats[4] = { GL_RED, GL_RG, GL_RGB, GL_RGBA };
    GLenum format = formats[channels - 1];

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, pixels);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    free(pixels);

    return texture;
}

void createShaders() {
    r_gBufferShader.handle = createShader(spriteVertex, spriteFragment);
    r_gBufferShader.viewLocation = glGetUniformLocation(r_gBufferShader.handle, "u_view");
    r_gBufferShader.projLocation = glGetUniformLocation(r_gBufferShader.handle, "u_proj");
    r_gBufferShader.modelLocation = glGetUniformLocation(r_gBufferShader.handle, "u_model");
    r_gBufferShader.diffuseTextureLocation = glGetUniformLocation(r_gBufferShader.handle, "u_diffuse");
    r_gBufferShader.normalTextureLocation = glGetUniformLocation(r_gBufferShader.handle, "u_normal");
    r_gBufferShader.tintLocation = glGetUniformLocation(r_gBufferShader.handle, "u_tint");
    r_gBufferShader.lightPositionLocation = glGetUniformLocation(r_gBufferShader.handle, "u_lightPosition");

    r_voronoiGeneratorShader.handle = createShader(screenQuadVertex, voronoiGeneratorFragment);
    r_voronoiGeneratorShader.inVoronoiLocation = glGetUniformLocation(r_voronoiGeneratorShader.handle, "u_in");
    r_voronoiGeneratorShader.offsetLocation = glGetUniformLocation(r_voronoiGeneratorShader.handle, "u_offset");
    r_voronoiGeneratorShader.resolutionLocation = glGetUniformLocation(r_voronoiGeneratorShader.handle, "u_resolution");

    r_distanceFieldFromVoronoiShader.handle = createShader(screenQuadVertex, distanceFieldFromVoronoiFragment);
    r_distanceFieldFromVoronoiShader.inVoronoiLocation = glGetUniformLocation(r_distanceFieldFromVoronoiShader.handle, "u_in");
    r_distanceFieldFromVoronoiShader.resolutionLocation = glGetUniformLocation(r_distanceFieldFromVoronoiShader.handle, "u_resolution");
    r_distanceFieldFromVoronoiShader.distanceScaleLocation = glGetUniformLocation(r_distanceFieldFromVoronoiShader.handle, "u_distanceScale");

    r_rayMarchShader.handle = createShader(screenQuadVertex, rayMarchFragment);
    r_rayMarchShader.inDistanceFieldLocation = glGetUniformLocation(r_rayMarchShader.handle, "u_inDistanceField");
    r_rayMarchShader.inSceneDiffuseLocation = glGetUniformLocation(r_rayMarchShader.handle, "u_inGDiffuseTexture");
    r_rayMarchShader.inSceneNormalLocation = glGetUniformLocation(r_rayMarchShader.handle, "u_inGNormalTexture");
    r_rayMarchShader.inLastEmissiveTextureLocation = glGetUniformLocation(r_rayMarchShader.handle, "u_lastEmissiveTexture");
    r_rayMarchShader.inAccumulatedEmissiveTextureLocation = glGetUniformLocation(r_rayMarchShader.handle, "r_accumulationEmissiveTexture");
    r_rayMarchShader.bounceLightEnabledLocation = glGetUniformLocation(r_rayMarchShader.handle, "u_bounceLightEnabled");
    r_rayMarchShader.bounceLightDampeningLocation = glGetUniformLocation(r_rayMarchShader.handle, "u_bounceLightDampening");
    r_rayMarchShader.resolutionLocation = glGetUniformLocation(r_rayMarchShader.handle, "u_resolution");
    r_rayMarchShader.timeLocation = glGetUniformLocation(r_rayMarchShader.handle, "u_time");
    r_rayMarchShader.maxStepsLocation = glGetUniformLocation(r_rayMarchShader.handle, "u_maxSteps");
    r_rayMarchShader.raysPerPixelLocation = glGetUniformLocation(r_rayMarchShader.handle, "u_raysPerPixel");
    r_rayMarchShader.distanceScaleLocation = glGetUniformLocation(r_rayMarchShader.handle, "u_distanceScale");
    r_rayMarchShader.emissiveScaleLocation = glGetUniformLocation(r_rayMarchShader.handle, "u_emissiveScale");

    r_drawTextureShader.handle = createShader(screenQuadVertex, copyTextureFragment);
    r_drawTextureShader.textureLocation = glGetUniformLocation(r_drawTextureShader.handle, "u_texture");

    r_gaussianBlurShader.handle = createShader(screenQuadVertex, gaussianFragment);
    r_gaussianBlurShader.blurScaleLocation = glGetUniformLocation(r_gaussianBlurShader.handle, "u_blurScale");
    r_gaussianBlurShader.textureResolutionLocation = glGetUniformLocation(r_gaussianBlurShader.handle, "u_textureResolution");
    r_gaussianBlurShader.textureLocation = glGetUniformLocation(r_gaussianBlurShader.handle, "u_texture");
}

void createTextures() {
    glGenTextures(1, &r_voronoiTexture);
    glBindTexture(GL_TEXTURE_2D, r_voronoiTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, r_rayMarchWidth, r_rayMarchWidth, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

    glGenFramebuffers(1, &r_voronoiFramebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, r_voronoiFramebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, r_voronoiTexture, 0);

    glGenTextures(1, &r_voronoiTexture2);
    glBindTexture(GL_TEXTURE_2D, r_voronoiTexture2);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, r_rayMarchWidth, r_rayMarchWidth, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

    glGenFramebuffers(1, &r_voronoiFramebuffer2);
    glBindFramebuffer(GL_FRAMEBUFFER, r_voronoiFramebuffer2);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, r_voronoiTexture2, 0);

    glGenTextures(1, &r_distanceFieldTexture);
    glBindTexture(GL_TEXTURE_2D, r_distanceFieldTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, r_rayMarchWidth, r_rayMarchWidth, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

    glGenFramebuffers(1, &r_distanceFieldFramebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, r_distanceFieldFramebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, r_distanceFieldTexture, 0);

    glGenTextures(1, &r_gDiffuseTexture);
    glBindTexture(GL_TEXTURE_2D, r_gDiffuseTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, r_rayMarchWidth, r_rayMarchWidth, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenTextures(1, &r_gNormalTexture);
    glBindTexture(GL_TEXTURE_2D, r_gNormalTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, r_rayMarchWidth, r_rayMarchWidth, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenTextures(1, &r_gDepthTexture);
    glBindTexture(GL_TEXTURE_2D, r_gDepthTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, r_rayMarchWidth, r_rayMarchWidth, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenFramebuffers(1, &r_gBufferFramebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, r_gBufferFramebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, r_gDiffuseTexture, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, r_gNormalTexture, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, r_voronoiTexture, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, r_gDepthTexture, 0);

    glGenTextures(1, &r_accumulationEmissiveTexture);
    glBindTexture(GL_TEXTURE_2D, r_accumulationEmissiveTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, r_rayMarchWidth, r_rayMarchWidth, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

    glGenTextures(1, &r_globalEmissiveTexture);
    glBindTexture(GL_TEXTURE_2D, r_globalEmissiveTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, r_rayMarchWidth, r_rayMarchWidth, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

    glGenFramebuffers(1, &r_globalEmissiveFramebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, r_globalEmissiveFramebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, r_globalEmissiveTexture, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, r_accumulationEmissiveTexture, 0);

    glGenTextures(1, &r_globalEmissiveTexture2);
    glBindTexture(GL_TEXTURE_2D, r_globalEmissiveTexture2);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, r_rayMarchWidth, r_rayMarchWidth, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

    glGenFramebuffers(1, &r_globalEmissiveFramebuffer2);
    glBindFramebuffer(GL_FRAMEBUFFER, r_globalEmissiveFramebuffer2);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, r_globalEmissiveTexture2, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, r_accumulationEmissiveTexture, 0);

    r_globalEmissiveRing = Ring<RenderTexturePair>({
        { r_globalEmissiveFramebuffer, r_globalEmissiveTexture },
        { r_globalEmissiveFramebuffer2, r_globalEmissiveTexture2 },
    });

    glGenTextures(1, &r_emissiveBlurredTexture);
    glBindTexture(GL_TEXTURE_2D, r_emissiveBlurredTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, r_rayMarchWidth, r_rayMarchWidth, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

    glGenFramebuffers(1, &r_emissiveBlurredFramebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, r_emissiveBlurredFramebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, r_emissiveBlurredTexture, 0);
}

void createGeometryBuffers() {
    QuadVertex quad[4] = {
        {{-1, -1}, {0, 0}},
        {{ 1, -1}, {1, 0}},
        {{ 1,  1}, {1, 1}},
        {{-1,  1}, {0, 1}},
    };

    uint32_t index[6] = {
        0, 1, 2,
        2, 3, 0,
    };

    glGenBuffers(1, &r_spriteInstanceMesh.vertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, r_spriteInstanceMesh.vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glGenBuffers(1, &r_spriteInstanceMesh.indexBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, r_spriteInstanceMesh.indexBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(index), index, GL_STATIC_DRAW);
    glGenVertexArrays(1, &r_spriteInstanceMesh.vertexArray);
    glBindVertexArray(r_spriteInstanceMesh.vertexArray);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, r_spriteInstanceMesh.indexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, r_spriteInstanceMesh.vertexBuffer);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(QuadVertex), (void*)offsetof(QuadVertex, pos));
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(QuadVertex), (void*)offsetof(QuadVertex, uv));
    glBindVertexArray(0);

    glGenBuffers(1, &r_screenQuadMesh.vertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, r_screenQuadMesh.vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glGenBuffers(1, &r_screenQuadMesh.indexBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, r_screenQuadMesh.indexBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(index), index, GL_STATIC_DRAW);
    glGenVertexArrays(1, &r_screenQuadMesh.vertexArray);
    glBindVertexArray(r_screenQuadMesh.vertexArray);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, r_screenQuadMesh.indexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, r_screenQuadMesh.vertexBuffer);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(QuadVertex), (void*)offsetof(QuadVertex, pos));
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(QuadVertex), (void*)offsetof(QuadVertex, uv));
    glBindVertexArray(0);
}

void renderCreate(int width, int height) {
    r_rayMarchWidth = width / 3;
    r_rayMarchHeight = height / 3;
    r_screenWidth = width;
    r_screenHeight = height;
    r_aspect = (float)width / (float)height;

    r_cameraHeight = 1.0f;
    r_cameraSize = vec2(r_aspect, 1.f) * r_cameraHeight;

    r_view = glm::mat4(1.0f);
    r_projection = glm::ortho(-r_cameraSize.x, r_cameraSize.x, -r_cameraSize.y, r_cameraSize.y, -1.0f, 1.0f);

    int pos = SDL_WINDOWPOS_CENTERED;
    int flags = SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE;

    SDL_Init(SDL_INIT_VIDEO);
    r_window = SDL_CreateWindow("Hello World!", pos, pos, width, height, flags);
    r_opengl = SDL_GL_CreateContext(r_window);
    gladLoadGLLoader(SDL_GL_GetProcAddress);
    SDL_GL_SetSwapInterval(1);

    ImGui::CreateContext();
    ImGui_ImplOpenGL3_Init("#version 330");
    ImGui_ImplSDL2_InitForOpenGL(r_window, r_opengl);

    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback(errorCallback, 0);

    createShaders();
    createTextures();
    createGeometryBuffers();
}

void renderDestroy() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    SDL_GL_DeleteContext(r_opengl);
    SDL_DestroyWindow(r_window);
    SDL_Quit();
}

void render() {
    
    //
    // Update state
    //

    r_clock.tick();

    // sort by texture usage
    std::sort(sprites.begin(), sprites.end(), [](const Sprite& a, const Sprite& b) {
        return a.diffuse < b.diffuse || a.normal < b.normal;
    });

    //
    // Render the G buffer
    //

    glViewport(0, 0, r_rayMarchWidth, r_rayMarchHeight);

    glBindFramebuffer(GL_FRAMEBUFFER, r_gBufferFramebuffer);
    {
        GLuint attachments[3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
        glDrawBuffers(3, attachments);
    }
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    
    glUseProgram(r_gBufferShader.handle);
    glUniformMatrix4fv(r_gBufferShader.viewLocation, 1, GL_FALSE, &r_view[0][0]);
    glUniformMatrix4fv(r_gBufferShader.projLocation, 1, GL_FALSE, &r_projection[0][0]);
    glUniform3fv(r_gBufferShader.lightPositionLocation, 1, &lightPosition[0]);
    glBindVertexArray(r_spriteInstanceMesh.vertexArray);

    GLuint lastDiffuse = 0;
    GLuint lastNormal = 0;

    for (const Sprite& sprite : sprites) {
        mat4 model = mat4(1.f);
        model = glm::translate(model, vec3(sprite.position.x, sprite.position.y, 0.0f));
        model = glm::scale(model, vec3(sprite.size.x, sprite.size.y, 1.0f));
        model = glm::rotate(model, sprite.rotation, vec3(0, 0, 1));

        glUniform4fv(r_gBufferShader.tintLocation, 1, &sprite.color[0]);
        glUniformMatrix4fv(r_gBufferShader.modelLocation, 1, GL_FALSE, &model[0][0]);

        if (sprite.diffuse != lastDiffuse) {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, sprite.diffuse);
            glUniform1i(r_gBufferShader.diffuseTextureLocation, 0);
            lastDiffuse = sprite.diffuse;
        }
        
        if (sprite.normal != lastNormal) {
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, sprite.normal);
            glUniform1i(r_gBufferShader.normalTextureLocation, 1);
            lastNormal = sprite.normal;
        }

        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    }
    glTextureBarrier();

    // 
    // 2D Global Illumination pass
    //

    // 1. Generate a voronoi texture from the screen positions

    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);

    glBindVertexArray(r_screenQuadMesh.vertexArray);
    
    glUseProgram(r_voronoiGeneratorShader.handle);
    glUniform2i(r_voronoiGeneratorShader.resolutionLocation, r_rayMarchWidth, r_rayMarchHeight);

    int passes = (int)ceil(log(max(r_rayMarchWidth, r_rayMarchHeight)) / log(2.0));

    Ring<RenderTexturePair> voronoiRing({
        { r_voronoiFramebuffer, r_voronoiTexture },
        { r_voronoiFramebuffer2, r_voronoiTexture2 },
    });

    {
        GLuint attachments[1] = { GL_COLOR_ATTACHMENT0 };
        glDrawBuffers(1, attachments);
    }

    glActiveTexture(GL_TEXTURE0);
    for (int i = 0; i < min(passes, max(limitPasses, 0)); i++) {
        int offset = (int)pow(2, passes - i - 1);
        glBindFramebuffer(GL_FRAMEBUFFER, voronoiRing.nextValue().framebuffer);
        glBindTexture(GL_TEXTURE_2D, voronoiRing.currentValue().texture);
        glUniform1i(r_voronoiGeneratorShader.inVoronoiLocation, 0);
        glUniform1i(r_voronoiGeneratorShader.offsetLocation, offset);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glTextureBarrier();
        voronoiRing.next();
    }

    GLuint finalVoronoi = voronoiRing.currentValue().texture;

    // 2. Generate a distance field from the voronoi texture

    glUseProgram(r_distanceFieldFromVoronoiShader.handle);
    glBindFramebuffer(GL_FRAMEBUFFER, r_distanceFieldFramebuffer);
    //glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, finalVoronoi);
    glUniform1i(r_distanceFieldFromVoronoiShader.inVoronoiLocation, 0);
    glUniform1f(r_distanceFieldFromVoronoiShader.distanceScaleLocation, distanceScale);
    glUniform2i(r_distanceFieldFromVoronoiShader.resolutionLocation, r_rayMarchWidth, r_rayMarchHeight);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glTextureBarrier();

    // 3. Trace rays through the distance field and get the color for a hit from color[voronoi[uv]]

    glUseProgram(r_rayMarchShader.handle);
    glBindFramebuffer(GL_FRAMEBUFFER, r_globalEmissiveRing.currentValue().framebuffer);

    {
        GLuint attachments[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
        glDrawBuffers(2, attachments);
    }

    //glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, r_distanceFieldTexture);
    glUniform1i(r_rayMarchShader.inDistanceFieldLocation, 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, r_gDiffuseTexture);
    glUniform1i(r_rayMarchShader.inSceneDiffuseLocation, 1);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, r_gNormalTexture);
    glUniform1i(r_rayMarchShader.inSceneNormalLocation, 2);

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, r_globalEmissiveRing.nextValue().texture);
    glUniform1i(r_rayMarchShader.inLastEmissiveTextureLocation, 3);

    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, r_accumulationEmissiveTexture);
    glUniform1i(r_rayMarchShader.inAccumulatedEmissiveTextureLocation, 4);

    glUniform2i(r_rayMarchShader.resolutionLocation, r_rayMarchWidth, r_rayMarchHeight);
    glUniform1f(r_rayMarchShader.timeLocation, r_clock.totalTime);
    glUniform1i(r_rayMarchShader.maxStepsLocation, maxSteps);
    glUniform1i(r_rayMarchShader.raysPerPixelLocation, raysPerPixel);
    glUniform1f(r_rayMarchShader.distanceScaleLocation, distanceScale);
    glUniform1f(r_rayMarchShader.emissiveScaleLocation, emissiveScale);
    glUniform1f(r_rayMarchShader.bounceLightEnabledLocation, bounceLightEnabled);
    glUniform1f(r_rayMarchShader.bounceLightDampeningLocation, clamp(1.f - r_clock.deltaTime *  bounceLightDampening, 0.f, 1.f));
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glTextureBarrier();

    r_globalEmissiveRing.next();

    // 4. Blur the global emissive texture

    glUseProgram(r_gaussianBlurShader.handle);
    glBindFramebuffer(GL_FRAMEBUFFER, r_emissiveBlurredFramebuffer);
    glUniform2f(r_gaussianBlurShader.textureResolutionLocation, (float)r_rayMarchWidth, (float)r_rayMarchHeight);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, r_accumulationEmissiveTexture);
    glUniform1i(r_gaussianBlurShader.textureLocation, 0);
    glUniform2f(r_gaussianBlurShader.blurScaleLocation, blurAmount.x, 0);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glTextureBarrier();
    glBindTexture(GL_TEXTURE_2D, r_emissiveBlurredTexture);
    glUniform1i(r_gaussianBlurShader.textureLocation, 0);
    glUniform2f(r_gaussianBlurShader.blurScaleLocation, 0, blurAmount.y);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glTextureBarrier();

    //
    // Composite (right now this is debug)
    //

    glViewport(0, 0, r_screenWidth, r_screenHeight);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(r_drawTextureShader.handle);
    GLuint layers[7] = {
        r_gDiffuseTexture,
        r_gNormalTexture,
        finalVoronoi,
        r_distanceFieldTexture,
        r_globalEmissiveTexture,
        r_accumulationEmissiveTexture,
        r_emissiveBlurredTexture
    };
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, layers[layerIndex]);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glUniform1i(r_drawTextureShader.textureLocation, 0);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL2_NewFrame();
    ImGui::NewFrame();

    if (ImGui::Begin("Ray marching settings")) {
        ImGui::SliderInt("Max steps", &maxSteps, 1, 100);
        ImGui::SliderInt("Rays per pixel", &raysPerPixel, 1, 100);
        ImGui::SliderFloat("Distance scale", &distanceScale, 0.1f, 10.0f);
        ImGui::SliderFloat("Emissive scale", &emissiveScale, 0.1f, 10.0f);
        ImGui::SliderInt("Draw layer", &layerIndex, 0, sizeof(layers) / sizeof(GLuint) - 1);
        ImGui::SliderInt("Limit passes", &limitPasses, 0, passes);
        ImGui::Checkbox("Use Bounce Light", &bounceLightEnabled);        
        ImGui::SliderFloat("Bounce light dampening", &bounceLightDampening, 0.f, 1000.f);
        ImGui::SliderFloat3("Light Position", &lightPosition.x, -5.0f, 5.0f);
        ImGui::SliderFloat2("Blur Amount", &blurAmount.x, 0.0f, 1.0f);
    }
    ImGui::End();
    
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    SDL_GL_SwapWindow(r_window);

    // sprites are only drawn if submitted
    // could retained
    sprites.clear();
}

void pollEvents(InputState* pInput) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        ImGui_ImplSDL2_ProcessEvent(&event);
        switch (event.type) {
            case SDL_QUIT: {
                pInput->running = false;
                break;
            }

            case SDL_WINDOWEVENT: {
                switch (event.window.event) {
                    case SDL_WINDOWEVENT_RESIZED: {
                        r_screenWidth = event.window.data1;
                        r_screenHeight = event.window.data2;
                        r_aspect = (float)r_screenWidth / (float)r_screenHeight;
                        r_cameraSize = vec2(r_cameraHeight * r_aspect, r_cameraHeight);
                        r_projection = glm::ortho(-r_cameraSize.x, r_cameraSize.x, -r_cameraSize.y, r_cameraSize.y, -1.0f, 1.0f);
                        break;
                    }
                }
                break;
            }

            case SDL_MOUSEMOTION: {
                vec2 mouse = vec2(event.motion.x, r_screenHeight - event.motion.y) / vec2(r_screenWidth, r_screenHeight) * 2.f - 1.f;
                pInput->mouseScreenPosition = mouse * r_cameraSize;
                pInput->isMousePressed = event.motion.state & SDL_BUTTON_LMASK;
                break;
            }

            case SDL_MOUSEBUTTONDOWN: {
                pInput->isMousePressed = event.button.button == SDL_BUTTON_LEFT;
                break;
            }

            case SDL_MOUSEBUTTONUP: {
                if (event.button.button == SDL_BUTTON_LEFT) {
                    pInput->isMousePressed = false;
                }
                break;
            }

            case SDL_MOUSEWHEEL: {
                pInput->mouseWheel += event.wheel.y / 10.f;
                break;
            }

            case SDL_KEYDOWN: {
                pInput->keysDown[event.key.keysym.scancode] = 1.f;
                break;
            }

            case SDL_KEYUP: {
                pInput->keysDown[event.key.keysym.scancode] = 0.f;
                break;
            }
        }
    }
}

void errorCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam) {
    if (severity != GL_DEBUG_SEVERITY_HIGH) {
        return;
    }
    
    printf("OpenGL error:\n");
    printf("  Source: ");
    switch (source) {
        case GL_DEBUG_SOURCE_API:             printf("API"); break;
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   printf("Window System"); break;
        case GL_DEBUG_SOURCE_SHADER_COMPILER: printf("Shader Compiler"); break;
        case GL_DEBUG_SOURCE_THIRD_PARTY:     printf("Third Party"); break;
        case GL_DEBUG_SOURCE_APPLICATION:     printf("Application"); break;
        case GL_DEBUG_SOURCE_OTHER:           printf("Other"); break;
    }
    printf("\n");

    printf("  Type: ");
    switch (type) {
        case GL_DEBUG_TYPE_ERROR:               printf("Error"); break;
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: printf("Deprecated Behavior"); break;
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  printf("Undefined Behavior"); break;
        case GL_DEBUG_TYPE_PORTABILITY:         printf("Portability"); break;
        case GL_DEBUG_TYPE_PERFORMANCE:         printf("Performance"); break;
        case GL_DEBUG_TYPE_MARKER:              printf("Marker"); break;
        case GL_DEBUG_TYPE_PUSH_GROUP:          printf("Push Group"); break;
        case GL_DEBUG_TYPE_POP_GROUP:           printf("Pop Group"); break;
        case GL_DEBUG_TYPE_OTHER:               printf("Other"); break;
    }
    printf("\n");

    printf("  ID: %u\n", id);
    printf("  Severity: ");
    switch (severity) {
        case GL_DEBUG_SEVERITY_HIGH:         printf("High"); break;
        case GL_DEBUG_SEVERITY_MEDIUM:       printf("Medium"); break;
        case GL_DEBUG_SEVERITY_LOW:          printf("Low"); break;
        case GL_DEBUG_SEVERITY_NOTIFICATION: printf("Notification"); break;
    }
    printf("\n");

    printf("  Message: %s\n", message);

    // throw nullptr;
}

void addSprite(const Sprite& sprite) {
    sprites.push_back(sprite);
}