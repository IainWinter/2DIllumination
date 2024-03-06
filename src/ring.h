#pragma once

#include <vector>

template<typename _t>
class Ring {
public:
    Ring() 
        : m_index (0)
    {}

    Ring(const std::vector<_t>& values)
        : m_ring  (values)
        , m_index (0)
    {}

    const _t& currentValue() const {
        return m_ring[m_index];
    }

    const _t& nextValue() const {
        return m_ring[nextIndex()];
    }

    size_t nextIndex() const {
        return (m_index + 1) % m_ring.size();
    }

    void next() {
        m_index = nextIndex();
    }

private:
    std::vector<_t> m_ring;
    size_t m_index;
};