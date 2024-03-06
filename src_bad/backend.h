#pragma once

#include "frontend.h"
#include "gl/glad.h"

#include <vector>

struct TextureBackend : Texture {
    GLuint handle;
};

class RendererBackend {
public:
    Texture* createTexture(int width, int height) {
        Texture* texture = new Texture();
        texture->width = width;
        texture->height = height;

        GLuint handle;
        glGenTextures(1, &handle);
        glBindTexture(GL_TEXTURE_2D, handle);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

        TextureBackend backend;
        backend.external = texture;
        backend.handle = handle;

        m_textures.push_back(backend);
    }

    void destroyTexture(Texture* texture) {

    }

private:
    std::vector<TextureBackend> m_textures;
};