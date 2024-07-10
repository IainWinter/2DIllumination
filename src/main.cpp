#include "gl/glad.h"
#include "SDL2/SDL.h"

#include "glm/vec2.hpp"
#include "glm/mat4x4.hpp"
#include "glm/gtc/matrix_transform.hpp"
using namespace glm;

#include <vector>

#include "render.h"
#include "clock.h"

extern std::vector<Sprite> sprites;

int main() {
    InputState input{};
    input.running = true;

    renderCreate(1280, 1280);

    GLuint circleDiffuse = createTextureFromFile("C:/dev/src/simplest/circle.png");
    GLuint circleNormal = createTextureFromFile("C:/dev/src/simplest/circle_normal.png");

    Sprite circle{};
    circle.diffuse = createTextureFromFile("C:/dev/src/simplest/flashlight.png");
    circle.normal = createTextureFromFile("C:/dev/src/simplest/flashlight_normal.png");
    circle.size = vec2(0.03f, 0.03f);
    circle.position = vec2(0, 0);
    circle.color = vec4(1.f);

    Sprite cave{};
    cave.diffuse = createTextureFromFile("C:/dev/src/simplest/cave.png");
    cave.normal = createTextureFromFile("C:/dev/src/simplest/cave.png");
    cave.size = vec2(1.f, 1.f);
    cave.position = vec2(0, 0);
    cave.color = vec4(1.f);

    std::vector<Sprite> bullets;

    Clock clock;

    while (input.running) {
        clock.tick();

        pollEvents(&input);

        // update

        float speed = input.keysDown[SDL_SCANCODE_LSHIFT] == 0 ? .3f : .8f;
        vec2 delta = vec2(cos(circle.rotation), sin(circle.rotation));

        circle.position += input.keysDown[SDL_SCANCODE_W] * delta * clock.deltaTime * speed;
        circle.rotation += (input.keysDown[SDL_SCANCODE_Q] - input.keysDown[SDL_SCANCODE_E]) * clock.deltaTime * 2.f;

        if (input.isMousePressed) {
            Sprite bullet{};
            bullet.diffuse = circleDiffuse;
            bullet.normal = circleNormal;
            bullet.size = vec2(0.005f, 0.005f);
            bullet.position = circle.position;
            bullet.velocity = delta * .8f;
            bullet.life = 2.f;
            bullet.color = vec4(1, 0, 0, .1);
            bullets.push_back(bullet);
        }

        for (int i = 0; i < bullets.size(); i++) {
            Sprite& bullet = bullets[i];
            bullet.position += bullet.velocity * clock.deltaTime;
            bullet.life -= clock.deltaTime;

            // This has to be last
            if (bullet.life < 0) {
                bullets.erase(bullets.begin() + i);
                i--;
            }
        }

        // render

        addSprite(circle);
        addSprite(cave);

        for (Sprite& bullet : bullets) {
            addSprite(bullet);
        }

        render();
    }

    renderDestroy();

    return 0;
}