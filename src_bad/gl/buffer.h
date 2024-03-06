#pragma once

#include "glad.h"
#include <vector>

// Layer 0 (l0) is the lowest level of the rendering system
// It is responsible for creating the OpenGL objects and collecting some of their settings
// into structs

namespace l0 {

struct Buffer {
    GLuint handle;
    GLenum type;
    size_t size;
};

struct Texture {
    GLuint handle;
    int width, height;
};

struct Shader {
    GLuint handle;
};

Buffer createBuffer(GLenum type, size_t size);
void destroyBuffer(Buffer buffer);

void writeBuffer(Buffer buffer, const void* data);
void writeBufferRegion(Buffer buffer, size_t offset, size_t size, const void* data);

Texture createTexture(int width, int height, const void* data);
void destroyTexture(Texture texture);

void writeTexture(Texture texture, const char* data);
void writeTextureRegion(Texture texture, int x, int y, int width, int height, const char* data);



}

// // A chunk of data on the device
// // Can be used as a vertex buffer, index buffer, uniform buffer, etc.
// class Buffer {
// public:
//     Buffer(GLenum type, size_t size);
//     ~Buffer();

//     // no copies
//     Buffer(const Buffer&) = delete;
//     Buffer& operator=(const Buffer&) = delete;

//     // only moves
//     Buffer(Buffer&& other);
//     Buffer& operator=(Buffer&& other);

//     void write(size_t offset, size_t size, const void* data);

//     void bind();

//     GLenum type() const { return m_type; }
//     GLuint handle() const { return m_handle; }
//     size_t size() const { return m_size; }

// private:
//     GLenum m_type;
//     size_t m_size;

//     GLuint m_handle;
// };

// // A single allocation in a buffer allocator
// struct BufferAllocation {
//     size_t offset;
//     size_t size;
//     GLenum handle;
// };

// // A simple allocator for a buffer
// // Can be used to allocate small chunks of data from a buffer
// class BufferAllocator {
// public:
//     BufferAllocator(GLenum type, size_t size);
//     ~BufferAllocator();

//     // no copies
//     BufferAllocator(const BufferAllocator&) = delete;
//     BufferAllocator& operator=(const BufferAllocator&) = delete;

//     // only moves
//     BufferAllocator(BufferAllocator&& other);
//     BufferAllocator& operator=(BufferAllocator&& other);

//     BufferAllocation allocate(size_t size, const void* data);
//     void free(BufferAllocation allocation);

// private:
//     Buffer m_buffer;
//     std::vector<BufferAllocation> m_allocations;
// };