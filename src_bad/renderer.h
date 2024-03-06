#pragma once

#include "frontend.h"
#include <vector>

struct Sprite {
    float x, y, rotation;
    Texture* texture;
};

class DynamicTextureSpriteRenderer {
public:
    Sprite* spriteCreate(Texture* texture) {
        Sprite* sprite = new Sprite();
        sprite->texture = texture;
        return sprite;
    }

    void tick() {
        
    }

private:
    Device* device;

    std::vector<Sprite*> sprites;
};