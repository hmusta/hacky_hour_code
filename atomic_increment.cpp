#include <cstdint>
#include <omp.h>
#include <atomic>
#include <mutex>
#include <cstdio>
#include <random>
#include <iostream>

#include "timer.h"

uint64_t mu_locking_increment(size_t num_iterations, size_t num_threads) {
    uint64_t counter = 0;
    std::mutex mu;
    #pragma omp parallel for num_threads(num_threads)
    for (size_t i = 0; i < num_iterations; ++i) {
        std::mt19937 gen(i);
        std::uniform_int_distribution<> distrib(1, 6);
        uint64_t roll = distrib(gen);

        std::lock_guard<std::mutex> lock(mu);
        counter += roll;
    }
    return counter;
}

uint64_t omp_locking_increment(size_t num_iterations, size_t num_threads) {
    uint64_t counter = 0;
    #pragma omp parallel for num_threads(num_threads)
    for (size_t i = 0; i < num_iterations; ++i) {
        std::mt19937 gen(i);
        std::uniform_int_distribution<> distrib(1, 6);
        uint64_t roll = distrib(gen);

        #pragma omp critical
        counter += roll;
    }
    return counter;
}

uint64_t omp_reduction_increment(size_t num_iterations, size_t num_threads) {
    uint64_t counter = 0;
    #pragma omp parallel for num_threads(num_threads) reduction(+: counter)
    for (size_t i = 0; i < num_iterations; ++i) {
        std::mt19937 gen(i);
        std::uniform_int_distribution<> distrib(1, 6);
        uint64_t roll = distrib(gen);
        counter += roll;
    }
    return counter;
}

uint64_t atomic_increment_seq_cst(size_t num_iterations, size_t num_threads) {
    std::atomic<uint64_t> counter { 0 };
    #pragma omp parallel for num_threads(num_threads)
    for (size_t i = 0; i < num_iterations; ++i) {
        std::mt19937 gen(i);
        std::uniform_int_distribution<> distrib(1, 6);
        uint64_t roll = distrib(gen);
        counter.fetch_add(roll, std::memory_order_seq_cst);
    }
    return counter;
}

uint64_t atomic_increment_relaxed(size_t num_iterations, size_t num_threads) {
    std::atomic<uint64_t> counter { 0 };
    #pragma omp parallel for num_threads(num_threads)
    for (size_t i = 0; i < num_iterations; ++i) {
        std::mt19937 gen(i);
        std::uniform_int_distribution<> distrib(1, 6);
        uint64_t roll = distrib(gen);
        counter.fetch_add(roll, std::memory_order_relaxed);
    }
    return counter;
}

uint64_t atomic_increment_cas(size_t num_iterations, size_t num_threads) {
    std::atomic<uint64_t> counter { 0 };
    #pragma omp parallel for num_threads(num_threads)
    for (size_t i = 0; i < num_iterations; ++i) {
        std::mt19937 gen(i);
        std::uniform_int_distribution<> distrib(1, 6);
        uint64_t roll = distrib(gen);
        uint64_t expected = counter.load(std::memory_order_relaxed);
        while (!counter.compare_exchange_strong(expected, expected + roll,
                                                std::memory_order_release,
                                                std::memory_order_relaxed));
    }
    return counter;
}

int main(int argc, char **argv) {
    size_t num_threads = std::atol(argv[1]);
    size_t num_iterations = std::atol(argv[2]);
    std::cout << num_threads << "\t";
    timed_run("Mutex_Locking", mu_locking_increment, num_iterations, num_threads);

    std::cout << num_threads << "\t";
    timed_run("OMP_Locking", omp_locking_increment, num_iterations, num_threads);

    std::cout << num_threads << "\t";
    timed_run("OMP_Reduction", omp_reduction_increment, num_iterations, num_threads);

    std::cout << num_threads << "\t";
    timed_run("Atomic_seq_cst", atomic_increment_seq_cst, num_iterations, num_threads);

    std::cout << num_threads << "\t";
    timed_run("Atomic_relaxed", atomic_increment_relaxed, num_iterations, num_threads);

    std::cout << num_threads << "\t";
    timed_run("Atomic_cas", atomic_increment_cas, num_iterations, num_threads);

    return 0;
}
