#pragma once

#include "../gl/glad.h"
#include <vector>

// Store a fixed sized buffer which creates enough copies of it self
// to never stall either side of the pipeline when writing
class AsyncBuffer {
public:
    AsyncBuffer();
    AsyncBuffer(GLenum type, size_t size, const void* data);
    ~AsyncBuffer();

    // no copy or move
    AsyncBuffer(const AsyncBuffer&) = delete;
    AsyncBuffer(AsyncBuffer&&) = delete;
    AsyncBuffer& operator=(const AsyncBuffer&) = delete;
    AsyncBuffer& operator=(AsyncBuffer&&) = delete;

    void write(const void* data);

    void bind();

private:
    struct Buffer {
        GLuint handle;

        GLsync transferFence;
        GLsync renderFence;

        void* pinned;
    };

    GLenum m_type;
    GLuint m_flags;
    size_t m_size;

    std::vector<Buffer> m_buffers;
};