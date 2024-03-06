#include "render.h"

#include "SDL2/SDL.h"
#include "gl/glad.h"

#include "gl/buffer.h"

#include <vector>

struct Render {
    SDL_Window* window;
    SDL_GLContext opengl;

    // store all static meshes in a list
    std::vector<Mesh> staticMeshes;

    // make one large buffer for all static meshes to use
    BufferAllocation staticMeshes;
};

struct Game {
    bool isRunning;
};

static Render s_render;
static Game s_game;

void renderCreate() {
    SDL_InitSubSystem(SDL_INIT_VIDEO);

    s_render.window = SDL_CreateWindow("Game", 
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 
        640, 480, 
        SDL_WINDOW_RESIZABLE | SDL_WINDOW_OPENGL);

    s_render.opengl = SDL_GL_CreateContext(s_render.window);

    gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress);

    SDL_GL_SetSwapInterval(1);
}

void renderDestroy() {
    SDL_QuitSubSystem(SDL_INIT_VIDEO);
    SDL_DestroyWindow(s_render.window);
    SDL_GL_DeleteContext(s_render.opengl);
}

bool renderBeginFrame() {
    return true;
}

void renderEndFrame() {
    SDL_GL_SwapWindow(s_render.window);
}

void gameCreate() {
    s_game.isRunning = true;
}

void gameDestroy() {
}

void gamePollEvents() {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        switch (event.type) {
            case SDL_QUIT: {
                s_game.isRunning = false;
                break;
            }
        }
    }
}

bool gameBeginTick() {
    return s_game.isRunning;
}
