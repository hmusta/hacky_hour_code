#include <omp.h>
#include <random>
#include <cstdint>
#include <mutex>
#include <atomic>

#include "timer.h"

uint64_t fill_queue_omp(uint64_t* begin, size_t num_inserts, size_t num_threads) {
    const uint64_t *start = begin;
    #pragma omp parallel for num_threads(num_threads)
    for (size_t i = 0; i < num_inserts; ++i) {
        std::mt19937 gen(i);
        std::uniform_int_distribution<> distrib(1, 6);
        uint64_t roll = distrib(gen);

        #pragma omp critical
        {
        *begin = roll;
        ++begin;
        }
    }

    return std::accumulate(start, start + num_inserts, uint64_t{ 0 });
}

ssize_t fill_queue_lock(uint64_t* begin, size_t num_inserts, size_t num_threads) {
    const uint64_t *start = begin;
    std::mutex mu;
    #pragma omp parallel for num_threads(num_threads)
    for (size_t i = 0; i < num_inserts; ++i) {
        std::mt19937 gen(i);
        std::uniform_int_distribution<> distrib(1, 6);
        uint64_t roll = distrib(gen);

        std::lock_guard<std::mutex> lock(mu);
        *begin = roll;
        ++begin;
    }

    return std::accumulate(start, start + num_inserts, uint64_t{ 0 });
}

ssize_t fill_queue_atomic(uint64_t* begin, size_t num_inserts, size_t num_threads) {
    std::atomic<size_t> j { 0 };
    #pragma omp parallel for num_threads(num_threads)
    for (size_t i = 0; i < num_inserts; ++i) {
        std::mt19937 gen(i);
        std::uniform_int_distribution<> distrib(1, 6);
        uint64_t roll = distrib(gen);
        begin[j.fetch_add(1, std::memory_order_relaxed)] = roll;
    }

    return std::accumulate(begin, begin + num_inserts, uint64_t{ 0 });
}

ssize_t fill_queue_atomic_batches(uint64_t* begin, size_t num_inserts, size_t num_threads) {
    std::atomic<size_t> i { 0 };
    #pragma omp parallel num_threads(num_threads) shared(i) shared(begin)
    #pragma omp single
    while (i.load(std::memory_order_acquire) < num_inserts) {
        #pragma omp task
        {
            std::mt19937 gen(0);
            std::uniform_int_distribution<> distrib(1, num_inserts/num_threads);
            uint64_t count = distrib(gen);
            size_t prev_i = i.fetch_add(count, std::memory_order_release);
            size_t i = std::min(i + count, num_inserts);
            for (size_t j = prev_i; j < i; ++j) {
                std::mt19937 gen(j);
                std::uniform_int_distribution<> distrib(1, 6);
                begin[j] = distrib(gen);
            }
        }
    }

    return std::accumulate(begin, begin + num_inserts, uint64_t{ 0 });
}

ssize_t fill_queue_ideal(uint64_t* begin, size_t num_inserts, size_t num_threads) {
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (size_t i = 0; i < num_inserts; ++i) {
        std::mt19937 gen(i);
        std::uniform_int_distribution<> distrib(1, 6);
        uint64_t roll = distrib(gen);
        begin[i] = roll;
    }

    return std::accumulate(begin, begin + num_inserts, uint64_t{ 0 });
}

int main(int argc, char **argv) {
    size_t num_threads = std::atol(argv[1]);
    size_t num_iterations = std::atol(argv[2]);
    std::vector<uint64_t> storage(num_iterations);
    size_t num_inserts = num_iterations * 0.8;
    std::cout << num_threads << "\t";
    timed_run("Mutex_Locking", fill_queue_lock, storage.data(), num_inserts, num_threads);

    std::cout << num_threads << "\t";
    timed_run("OMP_Locking", fill_queue_omp, storage.data(), num_inserts, num_threads);

    std::cout << num_threads << "\t";
    timed_run("Ideal", fill_queue_ideal, storage.data(), num_inserts, num_threads);

    std::cout << num_threads << "\t";
    timed_run("Atomic_Batches", fill_queue_atomic_batches, storage.data(), num_inserts, num_threads);

    std::cout << num_threads << "\t";
    timed_run("Atomic", fill_queue_atomic, storage.data(), num_inserts, num_threads);

    return 0;
}
