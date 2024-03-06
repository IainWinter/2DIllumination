#include <iostream>
#include <vector>

#include "r.h"

#include <chrono>

int main() {
    render::create();

    auto last = std::chrono::high_resolution_clock::now();
    float acc = 0.f;
    int accTicks = 0;
    int ticks = 0;

    // Texture* texture = render::textureCreateEmpty(32, 32);
    // TextureView* view = render::textureViewCreate(texture, vec2(0, 0), vec2(1, 1), ivec2(0, 0), ivec2(32, 32));
    // Sprite* sprite = render::spriteCreate(view, vec4(1, 1, 1, 1), Transform2D { vec2(0, 0), 0, vec2(1, 1)});

    while (1) {
        using ms = std::chrono::duration<float, std::milli>;
        auto now = std::chrono::high_resolution_clock::now();
        auto deltaTime = std::chrono::duration_cast<ms>(now - last).count() / 1000.f;

        acc += deltaTime;
        accTicks += 1;
        ticks += 1;

        // if (acc > .1f) {
        //     return 0;
        // }

        if (acc > 1.f) {
            float avgDeltaTime = acc / accTicks;

            printf("ms: %f fps: %f insts: %d\n", avgDeltaTime, 1.f / avgDeltaTime, render::instCount());
            acc = 0.f;
            accTicks = 0;
        }

        render::pollEvents();

        //sprite->transform.rotation += 0.01f;

        render::draw(ticks);

        last = now;
    }

    render::destroy();

    return 0;
}

// class Scene {
// public:
// private:
//     std::vector<Sprite*> m_sprites;
// };

// void setup() {

// }

// void tick() {

// }

// void draw() {

// }

// int main() {
//     renderCreate();
//     gameCreate();

//     setup();

//     while (gameBeginTick()) {
//         gamePollEvents();
//         tick();
//         if (renderBeginFrame()) {
//             draw();
//             renderEndFrame();
//         }
//     }

//     gameDestroy();
//     renderDestroy();
// }