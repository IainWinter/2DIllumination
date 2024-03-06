#pragma once

struct Texture {
    int width;
    int height;
};

// I think that this still needs to be an interface...
class Device {
public:
    virtual ~Device() = default;

    // Textures

    virtual Texture* textureCreateEmpty(int width, int height) = 0;
    virtual Texture* textureCreate(int width, int height, const char* data) = 0;
    
    virtual void textureDestroy(Texture* texture) = 0;
};