#pragma once

#include <chrono>

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