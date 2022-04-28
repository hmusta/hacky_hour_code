#include <functional>
#include <chrono>
#include <string>
#include <iostream>

template <class F, typename... Args>
void timed_run(const std::string &name, F&& f, Args&&... args) {
    auto start = std::chrono::high_resolution_clock::now();
    auto result = std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << name << "\t" << result << "\t" << duration.count() << "\n";
}
