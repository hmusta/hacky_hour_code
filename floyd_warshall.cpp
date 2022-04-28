#include <cstdint>
#include <cstddef>
#include <vector>
#include <iostream>
#include <x86intrin.h>

#include <boost/align/aligned_allocator.hpp>

#include "timer.h"

template <typename T>
using AlignedVector = std::vector<T, boost::alignment::aligned_allocator<T, 512>>;

uint32_t floyd_warshall(std::vector<AlignedVector<uint32_t>> &d) {
    size_t dsize = d.size();
    for (size_t k = 0; k < dsize; ++k) {
        for (size_t i = 0; i < dsize; ++i) {
            #pragma omp simd
            for (size_t j = 0; j < dsize; ++j) {
                d[i][j] = std::min(d[i][j], d[i][k] + d[k][j]);
            }
        }
    }
    return d.back().back();
}

uint32_t floyd_warshall2(std::vector<AlignedVector<uint32_t>> &d) {
    size_t dsize = d.size();
    for (size_t k = 0; k < dsize; ++k) {
        for (size_t i = 0; i < dsize; ++i) {
            for (size_t j = 0; j < dsize; ++j) {
                d[i][j] = std::min(d[i][j], d[i][k] + d[k][j]);
            }
        }
    }
    return d.back().back();
}

uint32_t floyd_warshall3(std::vector<AlignedVector<uint32_t>> &d) {
    size_t dsize = d.size();
    for (size_t k = 0; k < dsize; ++k) {
        for (size_t i = 0; i < dsize; ++i) {
            size_t j = 0;
#if 0
#if __AVX512F__
            // This is measurably faster, but underclocks the CPU...
            {
                __m512i interm_1 = _mm512_set1_epi32(d[i][k]);
                for ( ; j + 16 <= dsize; j += 16) {
                    __m512i old = _mm512_load_si512((__m512i*)&d[i][j]);
                    __m512i interm_2 = _mm512_load_si512((__m512i*)&d[k][j]);
                    __m512i next = _mm512_add_epi32(interm_1, interm_2);
                    __m512i val = _mm512_min_epi32(old, next);
                    _mm512_store_si512((__m512i*)&d[i][j], val);
                }
            }
#endif
#endif
#if __AVX2__
            {
                __m256i interm_1 = _mm256_set1_epi32(d[i][k]);
                for ( ; j + 8 <= dsize; j += 8) {
                    __m256i old = _mm256_load_si256((__m256i*)&d[i][j]);
                    __m256i interm_2 = _mm256_load_si256((__m256i*)&d[k][j]);
                    __m256i next = _mm256_add_epi32(interm_1, interm_2);
                    __m256i val = _mm256_min_epi32(old, next);
                    _mm256_store_si256((__m256i*)&d[i][j], val);
                }
                _mm256_zeroupper();
            }
#endif
            {
                __m128i interm_1 = _mm_set1_epi32(d[i][k]);
                for ( ; j + 4 <= dsize; j += 4) {
                    __m128i old = _mm_load_si128((__m128i*)&d[i][j]);
                    __m128i interm_2 = _mm_load_si128((__m128i*)&d[k][j]);
                    __m128i next = _mm_add_epi32(interm_1, interm_2);
                    __m128i val = _mm_min_epi32(old, next);
                    _mm_store_si128((__m128i*)&d[i][j], val);
                }
            }
            for ( ; j < dsize; ++j) {
                d[i][j] = std::min(d[i][j], d[i][k] + d[k][j]);
            }
        }
    }
    return d.back().back();
}

int main(void) {
    std::vector<AlignedVector<uint32_t>> dist(2023, AlignedVector<uint32_t>(2023));
    timed_run("Pragma", floyd_warshall, dist);
    timed_run("Autovec", floyd_warshall2, dist);
    timed_run("Expvec", floyd_warshall3, dist);
    return 0;
}
