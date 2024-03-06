#include "r.h"

#include "render/async_buffer.h"

#include "SDL2/SDL.h"

#define print(...) printf(__VA_ARGS__)
//#define print(...)

float randf() {
    return (float)rand() / RAND_MAX;
}

class SpriteMesh {
public:
    struct Vertex {
        vec2 position;
        vec2 uv;
    };

    struct Instance {
        vec2 position;
        float rotation;
        vec2 scale;
        vec4 tint;
    };

public:
    SpriteMesh(size_t batchSize) {
        m_instances = new Instance[batchSize];
        m_batchSize = batchSize;
        m_instanceCount = 0;

        Vertex vertices[4] = {
            {{ -1, -1 }, { 0, 0 }},
            {{ -1,  1 }, { 0, 1 }},
            {{  1,  1 }, { 1, 1 }},
            {{  1, -1 }, { 1, 0 }}
        };

        int indices[6] = {
            0, 1, 2,
            2, 3, 0,
        };

        glGenBuffers(1, &m_indexBuffer);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexBuffer);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

        glGenBuffers(1, &m_vertexBuffer);
        glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        for (int i = 0; i < 100; i++) {
            Instance inst;
            inst.position = { randf()*2 - 1, randf()*2 - 1 };
            inst.rotation = randf() * 3.14 * 2.f;
            inst.scale = { randf()*.1, randf()*.1 };
            inst.tint = { randf()*.8 + .2, randf()*.8 + .2, randf()*.8 + .2, randf()*.8 + .2 } ;
            m_instances[m_instanceCount++] = inst;
        }
    }

    ~SpriteMesh() {
        for (InstanceBuffer& buffer : m_instanceBuffers) {
            glDeleteVertexArrays(1, &buffer.vertexArray);
            glDeleteBuffers(1, &buffer.instanceBuffer);
        }

        glDeleteBuffers(1, &m_vertexBuffer);
        glDeleteBuffers(1, &m_indexBuffer);
    }

    // no moves no copies
    SpriteMesh(const SpriteMesh&) = delete;
    SpriteMesh& operator=(const SpriteMesh&) = delete;
    SpriteMesh(SpriteMesh&&) = delete;
    SpriteMesh& operator=(SpriteMesh&&) = delete;

    void draw(int tick) {
        // update the instances on cpu
        // if (m_instanceCount < m_batchSize) {
        //     Instance i;
        //     i.position = { randf()*2 - 1, randf()*2 - 1 };
        //     i.rotation = randf() * 3.14 * 2.f;
        //     i.scale = { randf()*.1, randf()*.1 };
        //     i.tint = { randf()*.8 + .2, randf()*.8 + .2, randf()*.8 + .2, randf()*.8 + .2 } ;
        //     m_instances[m_instanceCount++] = i;
        // }

        // write to the first available instance buffer
        // or create a new one if there are none

        // print("Fence state (tick %d)\n", tick);
        // for (size_t i = 0; i < m_instanceBuffers.size(); i++) {
        //     InstanceBuffer& buffer = m_instanceBuffers.at(i);

        //     GLenum result = glClientWaitSync(buffer.transferFence, 0, 0);
        //     GLenum result2 = glClientWaitSync(buffer.renderFence, 0, 0);

        //     const char* names[] = { 
        //         "-", // "Unsignaled      ",
        //         "+", // "Signaled        ",
        //         "+", // "Already signaled",
        //         "-", // "Timeout expired ",
        //         "+", // "Satisfied       ",
        //         "x", // "Wait failed     "
        //     };

        //     const char* r0 = names[(int)result - (int)GL_UNSIGNALED];
        //     const char* r1 = names[(int)result2 - (int)GL_UNSIGNALED];

        //     print("\tbuffer: %d fence state: %s %s created: %d last r: %d t: %d\n", i, r0, r1, buffer.tickCreated, buffer.tickRenderer, buffer.tickTransferred);
        // }

        // cull old transfers
        // this is a big hack should figure out why they dont get touched

        // for (int i = 0; i < m_instanceBuffers.size(); i++) {
        //     InstanceBuffer& buffer = m_instanceBuffers.at(i);

        //     if (buffer.tickTransferred != -1 && tick - buffer.tickTransferred > 100) {
        //         print("Culling buffer %d\n", i);
        //         glDeleteVertexArrays(1, &buffer.vertexArray);
        //         glDeleteBuffers(1, &buffer.instanceBuffer);
        //         m_instanceBuffers.erase(m_instanceBuffers.begin() + i);
        //         i--;
        //     }
        // }

        print("Transferring\n");

        size_t writeToIndex = -1;

        // find the last buffer in the list which is idle
        for (int i = m_instanceBuffers.size() - 1; i >= 0 ; i--) {
            InstanceBuffer& buffer = m_instanceBuffers.at(i);

            // If this buffer is idle, write to it

            GLenum result = glClientWaitSync(buffer.transferFence, 0, 0);
            GLenum result2 = glClientWaitSync(buffer.renderFence, 0, 0);

            const char* names[] = { 
                "Unsignaled      ",
                "Signaled        ",
                "Already signaled",
                "Timeout expired ",
                "Satisfied       ",
                "Wait failed     "
            };

            int index = (int)result - (int)GL_UNSIGNALED;
            int index2 = (int)result2 - (int)GL_UNSIGNALED;

            //print("\tbuffer %d. tf: %d rf: %d transfer: %s render: %s\n", i, buffer.transferFence, buffer.renderFence, names[index], names[index2]);

            if (   (result == GL_ALREADY_SIGNALED || result == GL_CONDITION_SATISFIED)
                && (result2 == GL_ALREADY_SIGNALED || result2 == GL_CONDITION_SATISFIED))
            {
                writeToIndex = i;
                break;
            }
        }

        // If one is found write to it
        if (writeToIndex != -1) {
            print("transfer %d\n", writeToIndex);

            InstanceBuffer& buffer = m_instanceBuffers.at(writeToIndex);

            // could write only ones that exist, and then zero the range
            // which was removed from the buffer that would be ezpz
            glBindBuffer(GL_ARRAY_BUFFER, buffer.instanceBuffer);
            void* mapped = glMapBufferRange(GL_ARRAY_BUFFER, 0, m_instanceCount * sizeof(Instance), GL_MAP_WRITE_BIT);
            
            for (int i = 0; i < 100; i++) {
                m_instances[i].rotation += 0.01f;
            }
            
            memcpy(mapped, m_instances, m_instanceCount * sizeof(Instance));
            glUnmapBuffer(GL_ARRAY_BUFFER);

            // Create a new fence to track this transfer
            glDeleteSync(buffer.transferFence);
            buffer.transferFence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);

            buffer.tickTransferred = tick;

            // move this buffer to the front
            // is this needed? https://www.youtube.com/watch?v=YNFaOnhaaso
            std::swap(m_instanceBuffers.front(), buffer);
        }

        // If there are no buffers create a new one
        else {
            print("\tCreating new buffer\n");

            GLuint instanceBuffer;
            glGenBuffers(1, &instanceBuffer);
            glBindBuffer(GL_ARRAY_BUFFER, instanceBuffer);
            glBufferStorage(GL_ARRAY_BUFFER, m_batchSize * sizeof(Instance), m_instances, GL_MAP_WRITE_BIT);

            ///void* pinned = glMapBufferRange(GL_ARRAY_BUFFER, 0, m_batchSize * sizeof(Instance), GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);

            // Create a new VOA
            GLuint vertexArray;
            glGenVertexArrays(1, &vertexArray);
            glBindVertexArray(vertexArray);

            glEnableVertexAttribArray(0);
            glEnableVertexAttribArray(1);
            glEnableVertexAttribArray(2);
            glEnableVertexAttribArray(3);
            glEnableVertexAttribArray(4);
            glEnableVertexAttribArray(5);

            glVertexAttribDivisor(2, 1);
            glVertexAttribDivisor(3, 1);
            glVertexAttribDivisor(4, 1);
            glVertexAttribDivisor(5, 1);

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexBuffer);
            
            glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffer);
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (const void*)offsetof(Vertex, position));
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (const void*)offsetof(Vertex, uv));

            glBindBuffer(GL_ARRAY_BUFFER, instanceBuffer);
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Instance), (const void*)offsetof(Instance, position));
            glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(Instance), (const void*)offsetof(Instance, rotation));
            glVertexAttribPointer(4, 2, GL_FLOAT, GL_FALSE, sizeof(Instance), (const void*)offsetof(Instance, scale));
            glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(Instance), (const void*)offsetof(Instance, tint));

            GLsync transferFence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
            GLsync renderFence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);

            InstanceBuffer buffer = {
                .vertexArray = vertexArray,
                .instanceBuffer = instanceBuffer,
                .transferFence = transferFence,
                .renderFence = renderFence,
                //.pinned = pinned,
                .tickCreated = tick,
                .tickRenderer = -1,
                .tickTransferred = -1
            };
            
            // also move this to the front
            m_instanceBuffers.push_back(buffer);
            std::swap(m_instanceBuffers.front(), m_instanceBuffers.back());
        }

        // find the first mesh which is not transferring

        size_t renderIndex = -1;

        print("Rendering\n");

        for (size_t i = 0; i < m_instanceBuffers.size(); i++) {
            InstanceBuffer& buffer = m_instanceBuffers.at(i);

            const char* names[] = { 
                "Unsignaled      ",
                "Signaled        ",
                "Already signaled",
                "Timeout expired ",
                "Satisfied       ",
                "Wait failed     "
            };

            GLenum result = glClientWaitSync(buffer.transferFence, 0, 0);

            //print("\tbuffer %d. tf: %d transfer: %s\n", i, buffer.transferFence, names[(int)result - (int)GL_UNSIGNALED]);

            if (result == GL_ALREADY_SIGNALED || result == GL_CONDITION_SATISFIED) {
                renderIndex = i;
                break;
            }
        }

        // there are no ready buffers to render
        if (renderIndex == -1) {
            print(" skipped\n");
            return;
        }

        print(" rendering %d\n", renderIndex);
        //print("async buffer count: %d\n", m_instanceBuffers.size());

        InstanceBuffer& buffer = m_instanceBuffers.at(renderIndex);

        // draw
        glBindVertexArray(buffer.vertexArray);
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr, m_instanceCount);

        // create a new fence to track this render
        glDeleteSync(buffer.renderFence);
        buffer.renderFence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);

        buffer.tickRenderer = tick;

        print("Done\n\n\n");
    }

    size_t m_instanceCount;

private:
    Instance* m_instances;
    size_t m_batchSize;

    // These are static buffers so can be shared
    GLuint m_indexBuffer;
    GLuint m_vertexBuffer;

    struct InstanceBuffer {
        // store a vao for each instance buffer
        // so that we dont have to rebind every time
        // may solve sync also
        GLuint vertexArray;

        GLuint instanceBuffer;

        GLsync transferFence;
        GLsync renderFence;

        void* pinned;

        int tickCreated;
        int tickRenderer;
        int tickTransferred;
    };

    std::vector<InstanceBuffer> m_instanceBuffers;
};

void errorCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam) {
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
}


namespace render {
    struct Context {
        SDL_Window* window;
        SDL_GLContext glContext;

        // Simplest sprite shader
        Shader* shaderHandle;

        SpriteMesh* spriteMesh;
    };

    static Context s_context;

    void create() {
        SDL_Init(SDL_INIT_VIDEO);

        SDL_Window* window = SDL_CreateWindow("Hello World!", 100, 100, 640, 480, SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL);
        SDL_GLContext glContext = SDL_GL_CreateContext(window);

        gladLoadGLLoader(SDL_GL_GetProcAddress);

        SDL_GL_SetSwapInterval(0);

        glEnable(GL_DEBUG_CALLBACK_FUNCTION);
        glDebugMessageCallback(errorCallback, nullptr);

        Shader* shaderHandle = shaderCreate(
            R"(
                #version 330 core

                layout (location = 0) in vec2 vert_position;
                layout (location = 1) in vec2 vert_uv;

                layout (location = 2) in vec2 inst_position;
                layout (location = 3) in float inst_rotation;
                layout (location = 4) in vec2 inst_scale;
                layout (location = 5) in vec4 inst_tint;

                out vec2 frag_uv;
                out vec4 frag_tint;

                void main() {
                    float c = cos(inst_rotation);
                    float s = sin(inst_rotation);

                    vec2 world_position = vec2(
                        vert_position.x * inst_scale.x * c - vert_position.y * inst_scale.y * s + inst_position.x,
                        vert_position.x * inst_scale.x * s + vert_position.y * inst_scale.y * c + inst_position.y
                    );

                    gl_Position = vec4(world_position, 0.0, 1.0);
                    frag_uv = vert_uv;
                    frag_tint = inst_tint;
                }
            )",
            R"(
                #version 330 core

                //uniform sampler2D uniform_texture;

                in vec2 frag_uv;
                in vec4 frag_tint;

                out vec4 final_color;

                void main() {
                    //vec4 color = texture(uniform_texture, frag_uv);

                    final_color = frag_tint;// * color;
                }
            )"
        );

        s_context.spriteMesh = new SpriteMesh(100000);

        s_context.window = window;
        s_context.glContext = glContext;
        s_context.shaderHandle = shaderHandle;
    }

    void destroy() {
        shaderDestroy(s_context.shaderHandle);

        SDL_GL_DeleteContext(s_context.glContext);
        SDL_DestroyWindow(s_context.window);
        SDL_Quit();
    }

    void pollEvents() {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                exit(0);
            }
        }
    }

    void draw(int tick) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(s_context.shaderHandle->handle);

        s_context.spriteMesh->draw(tick);

        SDL_GL_SwapWindow(s_context.window);
    }

    int instCount() {
        return s_context.spriteMesh->m_instanceCount;
    }

    Shader* shaderCreate(const char* vertexSource, const char* fragmentSource) {
        GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &vertexSource, nullptr);
        glCompileShader(vertexShader);

        // check for shader compile errors

        int success;
        glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
        if (!success) {
            char infoLog[2024];
            glGetShaderInfoLog(vertexShader, 2024, nullptr, infoLog);
            printf("Failed to compile vertex shader\n%s", infoLog);
        }

        GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &fragmentSource, nullptr);
        glCompileShader(fragmentShader);

        glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
        if (!success) {
            char infoLog[2024];
            glGetShaderInfoLog(fragmentShader, 2024, nullptr, infoLog);
            printf("Failed to compile fragment shader\n%s", infoLog);
        }

        GLuint program = glCreateProgram();
        glAttachShader(program, vertexShader);
        glAttachShader(program, fragmentShader);
        glLinkProgram(program);

        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        Shader* shader = new Shader();
        shader->handle = program;

        return shader;
    }

    void shaderDestroy(Shader* shader) {
        glDeleteProgram(shader->handle);
        delete shader;
    }
}