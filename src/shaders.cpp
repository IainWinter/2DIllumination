#include "shaders.h"

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

static const char* screenVertexShaderSource = R"(
    #version 330 core

    layout(location = 0) in vec2 vert_pos;
    layout(location = 1) in vec2 vert_uv;

    out vec2 frag_uv;

    void main() {
        frag_uv = vert_uv;
        gl_Position = vec4(vert_pos, 0.0, 1.0);
    }
)";

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

            uniform vec4 u_tint;

            void main() {
                vec4 color = u_tint * texture(u_diffuse, frag_uv);
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
    GLuint tintLocation = glGetUniformLocation(shader.handle, "u_tint");

    return SpriteShader {
        .program = shader,
        .viewLocation = viewLocation,
        .projLocation = projLocation,
        .modelLocation = modelLocation,
        .diffuseTextureLocation = diffuseTextureLocation,
        .normalTextureLocation = normalTextureLocation,
        .lightPositionLocation = lightPositionLocation,
        .tintLocation = tintLocation,
    };
}

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

VoronoiGeneratorShader createVoronoiGeneratorShader() {
    Shader program = createShader(screenVertexShaderSource,
        R"(
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
        )"
    );

    GLuint inVoronoiLocation = glGetUniformLocation(program.handle, "u_in");
    GLuint offsetLocation = glGetUniformLocation(program.handle, "u_offset");
    GLuint resolutionLocation = glGetUniformLocation(program.handle, "u_resolution");

    return VoronoiGeneratorShader {
        .program = program,
        .inVoronoiLocation = inVoronoiLocation,
        .offsetLocation = offsetLocation,
        .resolutionLocation = resolutionLocation,
    };
}

DistanceFieldFromVoronoiShader createDistanceFieldFromVoronoiShader() {
    Shader program = createShader(screenVertexShaderSource,
        R"(
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
        )");

    GLuint inVoronoiLocation = glGetUniformLocation(program.handle, "u_in");
    GLuint resolutionLocation = glGetUniformLocation(program.handle, "u_resolution");
    GLuint distanceScaleLocation = glGetUniformLocation(program.handle, "u_distanceScale");

    return DistanceFieldFromVoronoiShader {
        .program = program,
        .inVoronoiLocation = inVoronoiLocation,
        .resolutionLocation = resolutionLocation,
        .distanceScaleLocation = distanceScaleLocation,
    };
}

RayMarchShader createRayMarchShader() {
    Shader program = createShader(screenVertexShaderSource,
        R"(
            #version 330 core

            in vec2 frag_uv;
            out vec4 final_color;

            uniform sampler2D u_inDistanceField;
            uniform sampler2D u_inSceneTexture;

            uniform ivec2 u_resolution;
            uniform float u_time;
            uniform int u_maxSteps;
            uniform int u_raysPerPixel;
            uniform float u_distanceScale;
            uniform float u_emissiveScale;
            uniform float u_angleDebug;
            uniform vec2 u_offsetDebug;

            float epsilon() {
                return 0.5 / max(u_resolution.x, u_resolution.y);
            }

            bool rayMarch(vec2 origin, vec2 dir, out vec2 hitPos) {
                float currentDistance = 0.0;
                for (int i = 0; i < u_maxSteps; i++) {
                    vec2 point = origin + dir * currentDistance;

                    if (point.x < 0 || point.x >= 1 || point.y < 0 || point.y >= 1) {
                        return false;
                    }

                    float distance = texture(u_inDistanceField, point).r / u_distanceScale;

                    if (distance < epsilon()) {
                        hitPos = point;
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
                float pixelEmissive = 0.0;
                vec3 pixelColor = vec3(0.0);

                float PI = 3.141596;

                float rand2PI = random(frag_uv * vec2(u_time, -u_time)) * 2.0 * PI;
                float goldenAngle = PI * 0.7639320225; // magic number that gives us a good ray distribution.

                for (int i = 0; i < u_raysPerPixel; i++) {
                    float angle = rand2PI + goldenAngle * float(i);
                    //float angle = u_angleDebug;
                    vec2 dir = vec2(cos(angle), sin(angle));
                    vec2 origin = frag_uv;

                    vec2 outHitPos;
                    bool hit = rayMarch(origin, dir, outHitPos);
                    if (hit) {
                        // I keep getting bad results with a single pixel
                        // the voronoi texture seems always off by half a pixel or something...
                        
                        vec2 delta = 1.0 / vec2(u_resolution);
                        vec4 p0 = texture(u_inSceneTexture, outHitPos + vec2(-delta.x, -delta.y));
                        vec4 p1 = texture(u_inSceneTexture, outHitPos + vec2( delta.x, -delta.y));
                        vec4 p2 = texture(u_inSceneTexture, outHitPos + vec2(-delta.x,  delta.y));
                        vec4 p3 = texture(u_inSceneTexture, outHitPos + vec2( delta.x,  delta.y));

                        vec4 pixel = max(max(p0, p1), max(p2, p3));

                        //vec4 pixel = texture(u_inSceneTexture, outHitPos);

                        pixelEmissive += max(pixel.r, max(pixel.g, pixel.b)) * u_emissiveScale;
                        pixelColor += pixel.rgb;
                        //pixelColor.r = 1;
                    }
                }

                pixelColor /= pixelEmissive;             // note order
                pixelEmissive /= float(u_raysPerPixel);

                vec4 color = vec4(lin_to_srgb(pixelColor * pixelEmissive), 1.0);
                final_color = color;
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
        .inSceneTextureLocation = inSceneTextureLocation,
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

DrawTextureShader createDrawTextureShader() {
    Shader shader = createShader(screenVertexShaderSource,
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