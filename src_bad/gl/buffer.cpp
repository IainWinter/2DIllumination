#pragma once

#include "glad.h"
#include <vector>

class Buffer {
public:
    Buffer(GLenum type, size_t size);
    ~Buffer();

    // no copies
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    // only moves
    Buffer(Buffer&& other);
    Buffer& operator=(Buffer&& other);

    void write(size_t offset, size_t size, const void* data);

    void bind();

    GLenum type() const { return m_type; }
    GLuint handle() const { return m_handle; }
    size_t size() const { return m_size; }

private:
    GLenum m_type;
    size_t m_size;

    GLuint m_handle;
};

struct BufferAllocation {
    size_t offset;
    size_t size;
    GLenum handle;
};

class BufferAllocator {
public:
    BufferAllocator(GLenum type, size_t size);
    ~BufferAllocator();

    // no copies
    BufferAllocator(const BufferAllocator&) = delete;
    BufferAllocator& operator=(const BufferAllocator&) = delete;

    // only moves
    BufferAllocator(BufferAllocator&& other);
    BufferAllocator& operator=(BufferAllocator&& other);

    BufferAllocation allocate(size_t size, const void* data);
    void free(BufferAllocation allocation);

private:
    Buffer m_buffer;
    std::vector<BufferAllocation> m_allocations;
};