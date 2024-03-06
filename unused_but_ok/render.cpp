#include "render.h"

#include "gl/glad.h"
#include "SDL2/SDL.h"
#include "SDL2/SDL_opengl.h"

#include <vector>

namespace PixelRender {

struct Texture {
    uint32_t id;
    std::string name;

    uint32_t width;
    uint32_t height;

    char* pixels;

    GLuint handle;
};

struct Context {
    SDL_Window* window;
    SDL_GLContext opengl;

    float pixelSize;
    vec4 clearColor;

    std::vector<Texture> textures;
    std::vector<Sprite*> sprites;
};

static Context r;

void create() {
    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window* window = SDL_CreateWindow("Hello World!", 100, 100, 640, 480, SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL);
    SDL_GLContext glContext = SDL_GL_CreateContext(window);

    gladLoadGLLoader(SDL_GL_GetProcAddress);

    SDL_GL_SetSwapInterval(1);

    r.window = window;
    r.opengl = glContext;
}

void destroy() {
    SDL_GL_DeleteContext(r.opengl);
    SDL_DestroyWindow(r.window);
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

void draw() {
    glClearColor(r.clearColor.r, r.clearColor.g, r.clearColor.b, r.clearColor.a);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);



    SDL_GL_SwapWindow(r.window);
}

void setPixelSize(float pixelSize) {
    r.pixelSize = pixelSize;
}

void setClearColor(vec4 clearColor) {
    r.clearColor = clearColor;
}

void loadTexturesFromDirectory(const std::string& directory) {

}

Sprite* createSprite(const std::string& textureName);
void destroySprite(Sprite* sprite);

Trail* createTrail();
void destroyTrail(Trail* trail);

void spawnParticle(ParticleCreateInfo info) {

}

}