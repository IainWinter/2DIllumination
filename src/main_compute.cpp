#include "gl/glad.h"
#include "SDL2/SDL.h"

#include "glm/vec2.hpp"
#include "glm/mat4x4.hpp"
#include "glm/gtc/matrix_transform.hpp"
using namespace glm;

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_sdl2.h"
#include "imgui/imgui_impl_opengl3.h"

#include <vector>
#include <chrono>

void errorCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam) {
    if (severity == GL_DEBUG_SEVERITY_NOTIFICATION) {
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

    throw nullptr;
}

struct Clock {
    std::chrono::steady_clock::time_point last = std::chrono::high_resolution_clock::now();
    float acc = 0.f;
    int accTicks = 0;
    int ticks = 0;

    float deltaTime = 0;
    float totalTime = 0;

    void tick() {
        using ms = std::chrono::duration<float, std::milli>;
        auto now = std::chrono::high_resolution_clock::now();
        deltaTime = std::chrono::duration_cast<ms>(now - last).count() / 1000.f;
        totalTime += deltaTime;

        acc += deltaTime;
        accTicks += 1;
        ticks += 1;

        last = now;
    }
};

struct Texture {
    GLuint handle;
    GLenum format;
};

struct Shader {
    GLuint handle;
};

Texture createTextureFromFile(const char* filename) {
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

    return Texture {
        .handle = texture,
        .format = format,
    };
}

Shader createShader(const char* vertex, const char* fragment) {
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
        printf("Vertex shader source: %s\n", vertex);
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
        printf("Fragment shader source: %s\n", fragment);
        free(log);
        throw nullptr;
    }

    glAttachShader(shader, vertexShader);
    glAttachShader(shader, fragmentShader);
    glLinkProgram(shader);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return Shader {
        .handle = shader,
    };
}

Shader createComputeShader(const char* source) {
    GLuint shader = glCreateProgram();
    GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(computeShader, 1, &source, NULL);
    glCompileShader(computeShader);

    // check status
    GLint status;
    glGetShaderiv(computeShader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE) {
        GLint length;
        glGetShaderiv(computeShader, GL_INFO_LOG_LENGTH, &length);
        char* log = (char*)malloc(length);
        glGetShaderInfoLog(computeShader, length, &length, log);
        printf("Compute shader error: %s\n", log);
        printf("Compute shader source: %s\n", source);
        free(log);
        throw nullptr;
    }

    glAttachShader(shader, computeShader);
    glLinkProgram(shader);
    glDeleteShader(computeShader);

    return Shader {
        .handle = shader,
    };
}

struct SpriteShader {
    Shader program;

    GLuint viewLocation;
    GLuint projLocation;
    GLuint modelLocation;
    GLuint diffuseTextureLocation;
    GLuint normalTextureLocation;
    GLuint lightPositionLocation;
};

SpriteShader createSpriteShader() {
    Shader shader = createShader(
        R"(
            #version 330 core
            
            layout(location = 0) in vec2 vert_pos;
            layout(location = 1) in vec2 vert_uv;

            out vec2 frag_uv;
            out vec3 frag_position;

            uniform mat4 u_view;
            uniform mat4 u_proj;
            uniform mat4 u_model;

            void main() {
                vec4 pos = u_model * vec4(vert_pos, 0.0, 1.0);

                frag_uv = vert_uv;
                frag_position = pos.xyz;
                gl_Position = u_proj * u_view * pos;
            }
        )", 
        R"(
            #version 330 core

            in vec2 frag_uv;
            in vec3 frag_position;

            out vec4 final_color;

            uniform sampler2D u_diffuse;
            uniform sampler2D u_normal;
            uniform vec3 u_lightPosition;

            void main() {
                vec4 color = texture(u_diffuse, frag_uv);
                vec3 normal = texture(u_normal, frag_uv).rgb * 2.0 - 1.0;

                normal += vec3(0, 0, 1); // bias to pointing up
                normal = normalize(normal);

                vec3 lightDir = normalize(u_lightPosition - frag_position);
                vec3 viewDir = vec3(0, 0, 1);
                vec3 halfDir = normalize(lightDir + viewDir);

                float intensity = max(dot(normal, halfDir), 0.0);

                if (color.a == 0) {
                    discard;
                }

                final_color = vec4(color.rgb * intensity, color.a);
            }
        )"
    );

    GLuint viewLocation = glGetUniformLocation(shader.handle, "u_view");
    GLuint projLocation = glGetUniformLocation(shader.handle, "u_proj");
    GLuint modelLocation = glGetUniformLocation(shader.handle, "u_model");
    GLuint diffuseTextureLocation = glGetUniformLocation(shader.handle, "u_diffuse");
    GLuint normalTextureLocation = glGetUniformLocation(shader.handle, "u_normal");
    GLuint lightPositionLocation = glGetUniformLocation(shader.handle, "u_lightPosition");

    return SpriteShader {
        .program = shader,
        .viewLocation = viewLocation,
        .projLocation = projLocation,
        .modelLocation = modelLocation,
        .diffuseTextureLocation = diffuseTextureLocation,
        .normalTextureLocation = normalTextureLocation,
        .lightPositionLocation = lightPositionLocation,
    };
}

struct VoronoiGeneratorSeedShader {
    Shader program;

    GLuint viewLocation;
    GLuint projLocation;
    GLuint modelLocation;
    GLuint diffuseTextureLocation;
};

VoronoiGeneratorSeedShader createVoronoiGeneratorSeedShader() {
    Shader shader = createShader(
        R"(
            #version 330 core
            
            layout(location = 0) in vec2 vert_pos;
            layout(location = 1) in vec2 vert_uv;

            out vec2 frag_uv;
            out vec2 frag_screenUv;

            uniform mat4 u_view;
            uniform mat4 u_proj;
            uniform mat4 u_model;

            void main() {
                vec4 pos = u_model * vec4(vert_pos, 0.0, 1.0);
                vec4 clipPos = u_proj * u_view * pos;

                frag_uv = vert_uv;
                frag_screenUv = (clipPos.xy / clipPos.w + 1.0) / 2.0;
                gl_Position = clipPos;
            }
        )", 
        R"(
            #version 330 core

            in vec2 frag_uv;
            in vec2 frag_screenUv;

            out vec4 final_color;

            uniform sampler2D u_diffuse;

            void main() {
                vec4 color = texture(u_diffuse, frag_uv);
                
                if (color.a == 0) {
                    discard;
                }

                final_color = vec4(frag_screenUv, 0, 1);
            }
        )"
    );

    GLuint viewLocation = glGetUniformLocation(shader.handle, "u_view");
    GLuint projLocation = glGetUniformLocation(shader.handle, "u_proj");
    GLuint modelLocation = glGetUniformLocation(shader.handle, "u_model");
    GLuint diffuseTextureLocation = glGetUniformLocation(shader.handle, "u_diffuse");

    return VoronoiGeneratorSeedShader {
        .program = shader,
        .viewLocation = viewLocation,
        .projLocation = projLocation,
        .modelLocation = modelLocation,
        .diffuseTextureLocation = diffuseTextureLocation,
    };
}

struct VoronoiGeneratorShader {
    Shader program;

    GLuint inVoronoiLocation;
    GLuint outVoronoiLocation;
    GLuint offsetLocation;
    GLuint resolutionLocation;
};

VoronoiGeneratorShader createVoronoiGeneratorShader() {
    Shader program = createComputeShader(
        R"(
            #version 430 core

            layout(local_size_x = 16, local_size_y = 16) in;

            layout(rgba32f, binding = 0) uniform readonly image2D u_in;
            layout(rgba32f, binding = 1) uniform writeonly image2D u_out;

            uniform int u_offset;
            uniform ivec2 u_resolution;

            void main() {
                float closestDist = 9999999.0;
                vec2 closestPoint = vec2(0.0);

                ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
                vec2 screenUv = vec2(uv) / vec2(u_resolution);

                for (int x = -1; x <= 1; x++) {
                for (int y = -1; y <= 1; y++) {
                    ivec2 offset = uv + ivec2(x, y) * u_offset;

                    vec2 objScreenUv = imageLoad(u_in, offset).xy;
                    float dist = distance(screenUv, objScreenUv);

                    if (objScreenUv.x != 0.0 && objScreenUv.y != 0 && dist < closestDist) {
                        closestDist = dist;
                        closestPoint = objScreenUv;
                    }
                }
                }

                imageStore(u_out, uv, vec4(closestPoint, 0, 1));
            }
        )"
    );

    GLuint inVoronoiLocation = glGetUniformLocation(program.handle, "u_in");
    GLuint outVoronoiLocation = glGetUniformLocation(program.handle, "u_out"); 
    GLuint offsetLocation = glGetUniformLocation(program.handle, "u_offset");
    GLuint resolutionLocation = glGetUniformLocation(program.handle, "u_resolution");

    return VoronoiGeneratorShader {
        .program = program,
        .inVoronoiLocation = inVoronoiLocation,
        .outVoronoiLocation = outVoronoiLocation,
        .offsetLocation = offsetLocation,
        .resolutionLocation = resolutionLocation,
    };
}

struct DistanceFieldFromVoronoiShader {
    Shader program;

    GLuint inVoronoiLocation;
    GLuint outDistanceFieldLocation;
    GLuint resolutionLocation;
    GLuint distanceScaleLocation;
};

DistanceFieldFromVoronoiShader createDistanceFieldFromVoronoiShader() {
    Shader program = createComputeShader(
        R"(
            #version 430 core

            layout(local_size_x = 16, local_size_y = 16) in;

            layout(rgba32f, binding = 0) uniform readonly image2D u_inVoronoi;
            layout(rgba32f, binding = 1) uniform writeonly image2D u_outDistanceField;

            uniform ivec2 u_resolution;
            uniform float u_distanceScale;

            void main() {
                ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
                vec2 screenUv = vec2(uv) / vec2(u_resolution);

                vec2 fragScreenUv = imageLoad(u_inVoronoi, uv).xy;
                float dist = distance(screenUv, fragScreenUv);
                float mapped = clamp(dist * u_distanceScale, 0.0, 1.0);

                imageStore(u_outDistanceField, uv, vec4(vec3(mapped), 1.0));
            }
        )");

    GLuint inVoronoiLocation = glGetUniformLocation(program.handle, "u_in");
    GLuint outDistanceFieldLocation = glGetUniformLocation(program.handle, "u_out");
    GLuint resolutionLocation = glGetUniformLocation(program.handle, "u_resolution");
    GLuint distanceScaleLocation = glGetUniformLocation(program.handle, "u_distanceScale");

    return DistanceFieldFromVoronoiShader {
        .program = program,
        .inVoronoiLocation = inVoronoiLocation,
        .outDistanceFieldLocation = outDistanceFieldLocation,
        .resolutionLocation = resolutionLocation,
        .distanceScaleLocation = distanceScaleLocation,
    };
}

struct RayMarchShader {
    Shader program;
    
    GLuint inDistanceFieldLocation;
    GLuint inSceneTextureLocation;
    GLuint outEmissiveLocation;

    GLuint resolutionLocation;
    GLuint timeLocation;

    GLuint maxStepsLocation;
    GLuint raysPerPixelLocation;

    GLuint distanceScaleLocation;
    GLuint emissiveScaleLocation;

    GLuint angleDebugLocation;
    GLuint offsetDebugLocation;
};

RayMarchShader createRayMarchShader() {
    Shader program = createComputeShader(
        R"(
            #version 430 core

            layout(local_size_x = 16, local_size_y = 16) in;

            layout(rgba32f, binding = 0) uniform readonly image2D u_inDistanceField;
            layout(rgba8, binding = 1) uniform readonly image2D u_inSceneTexture;
            layout(rgba8, binding = 2) uniform writeonly image2D u_outEmissive;

            uniform ivec2 u_resolution;
            uniform float u_time;
            uniform int u_maxSteps;
            uniform int u_raysPerPixel;
            uniform float u_distanceScale;
            uniform float u_emissiveScale;
            uniform float u_angleDebug;
            uniform vec2 u_offsetDebug;

            bool rayMarch(vec2 origin, vec2 dir, out vec2 hitPos) {
                float currentDistance = 0.0;
                for (int i = 0; i < u_maxSteps; i++) {
                    vec2 pv = origin + dir * currentDistance * u_resolution; // dist is in screen space
                    ivec2 point = ivec2(round(pv.x), round(pv.y)); // maybe round properly?

                    if (point.x < 0 || point.x >= u_resolution.x || point.y < 0 || point.y >= u_resolution.y) {
                        return false;
                    }

                    float distance = imageLoad(u_inDistanceField, point).r / u_distanceScale;

                    if (distance < 0.001) {
                        hitPos = vec2(point);
                        return true;
                    }

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
                ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
                vec2 screenPos = vec2(uv) / vec2(u_resolution);

                float pixelEmissive = 0.0;
                vec3 pixelColor = vec3(0.0);

                //float PI = 3.141596;

                //float rand2PI = random(screenPos * vec2(u_time, -u_time)) * 2.0 * PI;
                //float goldenAngle = PI * 0.7639320225; // magic number that gives us a good ray distribution.

                //for (int i = 0; i < u_raysPerPixel; i++) {
                    //float angle = rand2PI + goldenAngle * float(i);
                    float angle = u_angleDebug;
                    vec2 dir = vec2(cos(angle), sin(angle));
                    vec2 origin = uv;

                    vec2 outHitPos;
                    bool hit = rayMarch(origin, dir, outHitPos);
                    if (hit) {
                        vec4 pixel = imageLoad(u_inSceneTexture, ivec2(outHitPos));
                        pixelEmissive += max(pixel.r, max(pixel.g, pixel.b)) * u_emissiveScale;
                        pixelColor += pixel.rgb;
                    }
                //}

                pixelColor /= pixelEmissive;             // note order
                pixelEmissive /= float(u_raysPerPixel);

                vec4 color = vec4(lin_to_srgb(pixelColor * pixelEmissive), 1.0);
                imageStore(u_outEmissive, uv, color);
            }
        )");

    GLuint inDistanceFieldLocation = glGetUniformLocation(program.handle, "u_inDistanceField");
    GLuint inSceneTextureLocation = glGetUniformLocation(program.handle, "u_inSceneTexture");
    GLuint outEmissiveLocation = glGetUniformLocation(program.handle, "u_outEmissive");
    GLuint resolutionLocation = glGetUniformLocation(program.handle, "u_resolution");
    GLuint timeLocation = glGetUniformLocation(program.handle, "u_time");
    GLuint maxStepsLocation = glGetUniformLocation(program.handle, "u_maxSteps");
    GLuint raysPerPixelLocation = glGetUniformLocation(program.handle, "u_raysPerPixel");
    GLuint distanceScaleLocation = glGetUniformLocation(program.handle, "u_distanceScale");
    GLuint emissiveScaleLocation = glGetUniformLocation(program.handle, "u_emissiveScale");
    GLuint angleDebugLocation = glGetUniformLocation(program.handle, "u_angleDebug");
    GLuint offsetDebugLocation = glGetUniformLocation(program.handle, "u_offsetDebug");

    return RayMarchShader {
        .program = program,
        .inDistanceFieldLocation = inDistanceFieldLocation,
        .outEmissiveLocation = outEmissiveLocation,
        .resolutionLocation = resolutionLocation,
        .timeLocation = timeLocation,
        .maxStepsLocation = maxStepsLocation,
        .raysPerPixelLocation = raysPerPixelLocation,
        .distanceScaleLocation = distanceScaleLocation,
        .emissiveScaleLocation = emissiveScaleLocation,
        .angleDebugLocation = angleDebugLocation,
        .offsetDebugLocation = offsetDebugLocation,
    };
}

struct DrawTextureShader {
    Shader program;

    GLuint textureLocation;
};

DrawTextureShader createDrawTextureShader() {
    Shader shader = createShader(
        R"(
            #version 330 core
            
            layout(location = 0) in vec2 vert_pos;
            layout(location = 1) in vec2 vert_uv;

            out vec2 frag_uv;

            void main() {
                frag_uv = vert_uv;
                gl_Position = vec4(vert_pos, 0.0, 1.0);
            }
        )", 
        R"(
            #version 330 core

            in vec2 frag_uv;

            out vec4 final_color;

            uniform sampler2D u_texture;

            void main() {
                final_color = texture(u_texture, frag_uv);
            }
        )"
    );

    GLuint textureLocation = glGetUniformLocation(shader.handle, "u_texture");

    return DrawTextureShader {
        .program = shader,
        .textureLocation = textureLocation,
    };
}

int main() {
    const int initialWidth = 1024;
    const int initialHeight = 1024;

    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow("Hello World!", 100, 100, initialWidth, initialHeight, SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    SDL_GLContext glContext = SDL_GL_CreateContext(window);
    gladLoadGLLoader(SDL_GL_GetProcAddress);
    SDL_GL_SetSwapInterval(1);

    ImGui::CreateContext();
    ImGui_ImplOpenGL3_Init("#version 330");
    ImGui_ImplSDL2_InitForOpenGL(window, glContext);

    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback(errorCallback, 0);

    // create a shader and a quad with a texture

    struct Vertex {
        vec2 pos;
        vec2 uv;
    };

    Vertex quad[4] = {
        {{-1, -1}, {0, 1}},
        {{ 1, -1}, {1, 1}},
        {{ 1,  1}, {1, 0}},
        {{-1,  1}, {0, 0}},
    };

    uint32_t index[6] = {
        0, 1, 2,
        2, 3, 0,
    };

    GLuint quadVertexBuffer;
    glGenBuffers(1, &quadVertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, quadVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);

    GLuint quadIndexBuffer;
    glGenBuffers(1, &quadIndexBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quadIndexBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(index), index, GL_STATIC_DRAW);

    GLuint quadVertexArray;
    glGenVertexArrays(1, &quadVertexArray);
    glBindVertexArray(quadVertexArray);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quadIndexBuffer);
    
    glBindBuffer(GL_ARRAY_BUFFER, quadVertexBuffer);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, pos));
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, uv));
    
    glBindVertexArray(0);

    SpriteShader spriteShader = createSpriteShader();
    VoronoiGeneratorSeedShader voronoiGeneratorSeedShader = createVoronoiGeneratorSeedShader();
    VoronoiGeneratorShader voronoiGeneratorShader = createVoronoiGeneratorShader();
    DistanceFieldFromVoronoiShader distanceFieldFromVoronoiShader = createDistanceFieldFromVoronoiShader();
    DrawTextureShader drawTextureShader = createDrawTextureShader();
    RayMarchShader rayMarchShader = createRayMarchShader();

    Texture diffuse = createTextureFromFile("C:/dev/src/simplest/circle.png");
    Texture normal = createTextureFromFile("C:/dev/src/simplest/circle_normal.png");

    // create voronoi render target

    GLuint voronoiTexture;
    glGenTextures(1, &voronoiTexture);
    glBindTexture(GL_TEXTURE_2D, voronoiTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, initialWidth, initialHeight, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    GLuint voronoiFramebuffer;
    glGenFramebuffers(1, &voronoiFramebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, voronoiFramebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, voronoiTexture, 0);

    // only need framebuffer for seeding pass

    GLuint voronoiTexture2;
    glGenTextures(1, &voronoiTexture2);
    glBindTexture(GL_TEXTURE_2D, voronoiTexture2);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, initialWidth, initialHeight, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    // texture for the distance field

    GLuint distanceFieldTexture;
    glGenTextures(1, &distanceFieldTexture);
    glBindTexture(GL_TEXTURE_2D, distanceFieldTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, initialWidth, initialHeight, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

    // texture for the main scene
    // the voronoi texture will map to this, so this one should store colors and material props
    // create a framebuffer for the main scene

    GLuint sceneTexture;
    glGenTextures(1, &sceneTexture);
    glBindTexture(GL_TEXTURE_2D, sceneTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, initialWidth, initialHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    GLuint sceneFramebuffer;
    glGenFramebuffers(1, &sceneFramebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, sceneFramebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, sceneTexture, 0);

    // raymarching emissive texture

    GLuint globalEmissiveTexture;
    glGenTextures(1, &globalEmissiveTexture);
    glBindTexture(GL_TEXTURE_2D, globalEmissiveTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, initialWidth, initialHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    float aspect = (float)initialWidth / (float)initialHeight;
    float cameraHeight = 1.f;
    vec2 cameraSize = cameraSize = vec2(cameraHeight * aspect, cameraHeight);
    int screenWidth = initialWidth;
    int screenHeight = initialHeight;
    mat4 view = glm::mat4(1.0f);
    mat4 projection = projection = glm::ortho(-cameraSize.x, cameraSize.x, -cameraSize.y, cameraSize.y, -1.0f, 1.0f);
    vec2 mouseScreenPosition = vec2(0.0f);
    int limitPasses = 10;
    bool running = true;
    std::vector<mat4> models;

    int maxSteps = 2;
    int raysPerPixel = 1;
    float distanceScale = 1.0f;
    float emissiveScale = 1.0f;
    int layerIndex = 3;

    float angleDebug = 0.0;
    vec2 offsetDebug = vec2(1, 1);

    for (int i = 0; i < 1; i++) {
        float x = (float)(rand() % 400) / 400.f * 2.f - 1.f;
        float y = (float)(rand() % 400) / 400.f * 2.f - 1.f;
        float scale = (float)(rand() % 1000) / 1000.f * 0.1f + 0.05f;

        mat4 model = mat4(1.f);
        model = glm::translate(model, vec3(x, y, 0.0f));
        model = glm::scale(model, vec3(scale, scale, 1.0f));

        models.push_back(model);
    }

    Clock clock;

    while (running) {
        clock.tick();

        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL2_ProcessEvent(&event);
            switch (event.type) {
                case SDL_QUIT: {
                    running = false;
                    break;
                }
                case SDL_WINDOWEVENT: {
                    switch (event.window.event) {
                        case SDL_WINDOWEVENT_RESIZED: {
                            screenWidth = event.window.data1;
                            screenHeight = event.window.data2;
                            aspect = (float)screenWidth / (float)screenHeight;
                            cameraSize = vec2(cameraHeight * aspect, cameraHeight);
                            projection = glm::ortho(-cameraSize.x, cameraSize.x, -cameraSize.y, cameraSize.y, -1.0f, 1.0f);

                            break;
                        }
                    }
                    break;
                }
                case SDL_MOUSEMOTION: {
                    vec2 mouse = vec2(event.motion.x, screenHeight-event.motion.y) / vec2(screenWidth, screenHeight) * 2.f - 1.f;
                    mouseScreenPosition = mouse * cameraSize;

                    break;
                }

                // when press + increment limitPasses and - decrement
                case SDL_KEYDOWN: {
                    if (event.key.keysym.sym == SDLK_EQUALS) {
                        limitPasses++;
                    } else if (event.key.keysym.sym == SDLK_MINUS) {
                        limitPasses--;
                    }

                    printf("limitPasses: %d\n", limitPasses);
                    break;
                }
            }
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        if (ImGui::Begin("Ray marching settings")) {
            int passes = (int)ceil(log(max(initialWidth, initialHeight)) / log(2.0));

            ImGui::SliderInt("Max steps", &maxSteps, 1, 100);
            ImGui::SliderInt("Rays per pixel", &raysPerPixel, 1, 100);
            ImGui::SliderFloat("Distance scale", &distanceScale, 0.1f, 10.0f);
            ImGui::SliderFloat("Emissive scale", &emissiveScale, 0.1f, 10.0f);
            ImGui::SliderInt("Draw layer", &layerIndex, 0, 3);
            ImGui::SliderInt("Limit passes", &limitPasses, 0, passes);
            ImGui::SliderFloat("Angle debug", &angleDebug, 0.0f, 6.28f);
            ImGui::SliderFloat2("Offset debug", &offsetDebug.x, -2.0f, 2.0f);
        }
        ImGui::End();

        mat4 mouseMover = mat4(1.f);
        //mouseMover = glm::translate(mouseMover, vec3(mouseScreenPosition.x, -mouseScreenPosition.y, 0.0f));
        mouseMover = glm::translate(mouseMover, vec3(cos(clock.totalTime)/100, sin(clock.totalTime)/100, 0.0f));
        mouseMover = glm::scale(mouseMover, vec3(0.1f, 0.1f, 1.0f));

        // render seeds to voronoi
        
        glBindFramebuffer(GL_FRAMEBUFFER, voronoiFramebuffer);
        glViewport(0, 0, initialWidth, initialHeight);
        glClear(GL_COLOR_BUFFER_BIT);
        
        glUseProgram(voronoiGeneratorSeedShader.program.handle);
        glUniformMatrix4fv(voronoiGeneratorSeedShader.viewLocation, 1, GL_FALSE, &view[0][0]);
        glUniformMatrix4fv(voronoiGeneratorSeedShader.projLocation, 1, GL_FALSE, &projection[0][0]);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, diffuse.handle);
        glUniform1i(voronoiGeneratorSeedShader.diffuseTextureLocation, 0);

        glBindVertexArray(quadVertexArray);

        for (int i = 0; i < models.size(); i++) {
            glUniformMatrix4fv(voronoiGeneratorSeedShader.modelLocation, 1, GL_FALSE, &models[i][0][0]);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        }

        glUniformMatrix4fv(voronoiGeneratorSeedShader.modelLocation, 1, GL_FALSE, &mouseMover[0][0]);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        // generate voronoi
        int passes = (int)ceil(log(max(initialWidth, initialHeight)) / log(2.0));
        int bufferIndex = 0;
        GLuint textures[2] = { voronoiTexture, voronoiTexture2 };
        GLuint finalVoronoi = textures[0]; // can calculate

        glUseProgram(voronoiGeneratorShader.program.handle);
        glUniform2i(voronoiGeneratorShader.resolutionLocation, initialWidth, initialHeight);

        for (int i = 0; i < min(passes, max(limitPasses, 0)); i++) {
            int readIndex = bufferIndex;
            int writeIndex = (bufferIndex + 1) % 2;
            GLuint writeTexture = textures[writeIndex];
            GLuint readTexture = textures[readIndex];
            int offset = (int)pow(2, passes - i - 1);

            glBindImageTexture(0, readTexture, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
            glBindImageTexture(1, writeTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
            glUniform1i(voronoiGeneratorShader.offsetLocation, offset);
            glDispatchCompute(initialWidth/16, initialHeight/16, 1);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);

            bufferIndex = writeIndex;
            finalVoronoi = writeTexture;
        }

        // generate distance field
        glUseProgram(distanceFieldFromVoronoiShader.program.handle);
        glBindImageTexture(0, finalVoronoi, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
        glBindImageTexture(1, distanceFieldTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
        glUniform1f(distanceFieldFromVoronoiShader.distanceScaleLocation, distanceScale);
        glUniform2i(distanceFieldFromVoronoiShader.resolutionLocation, initialWidth, initialHeight);
        glDispatchCompute(initialWidth/16, initialHeight/16, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);

        // draw the scene
        glBindFramebuffer(GL_FRAMEBUFFER, sceneFramebuffer);
        glViewport(0, 0, initialWidth, initialHeight);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(spriteShader.program.handle);
        glUniformMatrix4fv(spriteShader.viewLocation, 1, GL_FALSE, &view[0][0]);
        glUniformMatrix4fv(spriteShader.projLocation, 1, GL_FALSE, &projection[0][0]);
        glUniform3f(spriteShader.lightPositionLocation, mouseScreenPosition.x, -mouseScreenPosition.y, 1.0f);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, diffuse.handle);
        glUniform1i(spriteShader.diffuseTextureLocation, 0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, normal.handle);
        glUniform1i(spriteShader.normalTextureLocation, 1);

        glBindVertexArray(quadVertexArray);

        for (int i = 0; i < models.size(); i++) {
            glUniformMatrix4fv(spriteShader.modelLocation, 1, GL_FALSE, &models[i][0][0]);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        }

        glUniformMatrix4fv(spriteShader.modelLocation, 1, GL_FALSE, &mouseMover[0][0]);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        glMemoryBarrier(GL_ALL_BARRIER_BITS);

        // calc the ray marching

        glUseProgram(rayMarchShader.program.handle);
        glBindImageTexture(0, distanceFieldTexture, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
        glBindImageTexture(1, sceneTexture, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA8);
        glBindImageTexture(2, globalEmissiveTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
        glUniform2i(rayMarchShader.resolutionLocation, initialWidth, initialHeight);
        //glUniform1f(rayMarchShader.timeLocation, clock.totalTime);
        glUniform1i(rayMarchShader.maxStepsLocation, maxSteps);
        glUniform1i(rayMarchShader.raysPerPixelLocation, raysPerPixel);
        glUniform1f(rayMarchShader.distanceScaleLocation, distanceScale);
        glUniform1f(rayMarchShader.emissiveScaleLocation, emissiveScale);
        glUniform1f(rayMarchShader.angleDebugLocation, angleDebug);
        glUniform2f(rayMarchShader.offsetDebugLocation, offsetDebug.x, offsetDebug.y);
        glDispatchCompute(initialWidth/16, initialHeight/16, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);

        // debug draw layers

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, screenWidth, screenHeight);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(drawTextureShader.program.handle);

        GLuint layers[4] = {
            sceneTexture,
            finalVoronoi,
            distanceFieldTexture,
            globalEmissiveTexture
        };

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, layers[layerIndex]);
        glUniform1i(drawTextureShader.textureLocation, 0);

        glBindVertexArray(quadVertexArray);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        // final pass

        // glBindFramebuffer(GL_FRAMEBUFFER, 0);
        // glViewport(0, 0, screenWidth, screenHeight);
        // glClear(GL_COLOR_BUFFER_BIT);

        // glUseProgram(spriteShader.program.handle);
        // glUniform3f(spriteShader.lightPositionLocation, mouseScreenPosition.x, mouseScreenPosition.y, 1.0f);
        // glUniformMatrix4fv(spriteShader.viewLocation, 1, GL_FALSE, &view[0][0]);
        // glUniformMatrix4fv(spriteShader.projLocation, 1, GL_FALSE, &projection[0][0]);
        // glActiveTexture(GL_TEXTURE0);
        // glBindTexture(GL_TEXTURE_2D, diffuse.handle);
        // glUniform1i(spriteShader.diffuseTextureLocation, 0);
        // glActiveTexture(GL_TEXTURE1);
        // glBindTexture(GL_TEXTURE_2D, normal.handle);
        // glUniform1i(spriteShader.normalTextureLocation, 1);
        
        // glBindVertexArray(quadVertexArray);
        // glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        SDL_GL_SwapWindow(window);
    }

    // should clean gl resources :)

    ImGui_ImplSDL2_Shutdown();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui::DestroyContext();

    SDL_GL_DeleteContext(glContext);
    SDL_DestroyWindow(window);
    SDL_Quit();
}