#include "gl/glad.h"
#include "SDL2/SDL.h"

#include "glm/vec2.hpp"
#include "glm/mat4x4.hpp"
#include "glm/gtc/matrix_transform.hpp"
using namespace glm;

#include <vector>

#include "render.h"

extern std::vector<Sprite> sprites;

int main() {
    InputState input{};
    input.running = true;

    renderCreate(1280, 1280);

    Sprite circle;
    circle.diffuse = createTextureFromFile("C:/dev/src/simplest/circle.png");
    circle.normal = createTextureFromFile("C:/dev/src/simplest/circle_normal.png");
    circle.size = vec2(0.01f, 0.01f);
    circle.position = vec2(0, 0);
    circle.color = vec4(1.f);

    Sprite cave;
    cave.diffuse = createTextureFromFile("C:/dev/src/simplest/cave.png");
    cave.normal = createTextureFromFile("C:/dev/src/simplest/cave.png");
    cave.size = vec2(1.f, 1.f);
    cave.position = vec2(0, 0);
    cave.color = vec4(1.f);

    while (input.running) {
        pollEvents(&input);

        circle.position = input.mouseScreenPosition;
        addSprite(circle);

        addSprite(cave);

        render();

        // if (isMousePressed) {
        //     lastMousePressedScreenPosition = mouseScreenPosition;

        //     if (!isMousePressedLast) {
        //         Bullet bullet;
        //         bullet.position = mouseScreenPosition;
        //         bullet.velocity = normalize(vec2(rand() / (float)RAND_MAX, 0)) / 10.f;
        //         bullet.color = vec4(rand() / (float)RAND_MAX, rand() / (float)RAND_MAX, rand() / (float)RAND_MAX, 1);
        //         bullets.push_back(bullet);
        //     }
        // }

        // mat4 mouseMover = mat4(1.f);
        // mouseMover = glm::translate(mouseMover, vec3(lastMousePressedScreenPosition.x, lastMousePressedScreenPosition.y, 0.0f));
        // mouseMover = glm::scale(mouseMover, vec3(0.1f, 0.1f, 1.0f));
        // mouseMover = glm::rotate(mouseMover, mouseWheelPos, vec3(0, 0, 1));

        // for (int i = 0; i < bullets.size(); i++) {
        //     bullets[i].position += bullets[i].velocity * clock.deltaTime * 10.f;

        //     if (bullets[i].position.x < -cameraSize.x || bullets[i].position.x > cameraSize.x
        //         || bullets[i].position.y < -cameraSize.y || bullets[i].position.y > cameraSize.y)
        //     {
        //         bullets.erase(bullets.begin() + i);
        //         i--;
        //     }
        // }
    }

    renderDestroy();
}