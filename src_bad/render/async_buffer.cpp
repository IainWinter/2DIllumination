#include "async_buffer.h"

AsyncBuffer::AsyncBuffer() 
    : m_type  (GL_INVALID_ENUM)
    , m_flags (0)
    , m_size  (0)
{}

AsyncBuffer::AsyncBuffer(GLenum type, size_t size, const void* data) 
    : m_type (type)
    , m_size (size)
{
    m_flags = GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
    write(data);
}

AsyncBuffer::~AsyncBuffer() {
    for (Buffer& buffer : m_buffers) {
        glBindBuffer(m_type, buffer.handle);
        glUnmapBuffer(m_type);
        glDeleteBuffers(1, &buffer.handle);
        glDeleteSync(buffer.transferFence);
        glDeleteSync(buffer.renderFence);
    }
}

void AsyncBuffer::write(const void* data) {
    for (size_t i = 0; i < m_buffers.size(); i++) {
        Buffer& buffer = m_buffers.at(i);

        // If this buffer is idle, write to it
        
        if (   glClientWaitSync(buffer.transferFence, 0, 0) == GL_CONDITION_SATISFIED
            && glClientWaitSync(buffer.renderFence, 0, 0) == GL_CONDITION_SATISFIED)
        {
            memcpy(buffer.pinned, data, m_size);

            // Create a new fence to track this transfer
            glDeleteSync(buffer.transferFence);
            buffer.transferFence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);

            // move this buffer to the front?
            // https://www.youtube.com/watch?v=YNFaOnhaaso

            // exit after writing to the buffer
            return;
        }
    }

    // If there are no buffers create a new one

    GLuint handle;
    glGenBuffers(1, &handle);
    glBindBuffer(m_type, handle);
    glBufferStorage(m_type, m_size, data, m_flags);

    void* pinned = glMapBufferRange(m_type, 0, m_size, m_flags);

    GLsync transferFence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    GLsync renderFence = 0;

    Buffer buffer = {
        .handle = handle,
        .transferFence = transferFence,
        .renderFence = renderFence,
        .pinned = pinned,
    };
    
    m_buffers.push_back(buffer);
}

void AsyncBuffer::bind() {
    for (Buffer& buffer : m_buffers) {
        if (glClientWaitSync(buffer.transferFence, 0, 0) == GL_CONDITION_SATISFIED) {
            glBindBuffer(m_type, buffer.handle);
            return;
        }
    }
}