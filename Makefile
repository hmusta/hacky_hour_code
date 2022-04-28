CXXCMD=g++ -O3 --std=c++17 -march=native -fopenmp -DNDEBUG -g -fopenmp-simd

all: atomic_inc floyd fixed
atomic_inc: atomic_increment.cpp
	$(CXXCMD) -o atomic_increment atomic_increment.cpp
floyd: floyd_warshall.cpp
	$(CXXCMD) -o floyd_warshall floyd_warshall.cpp
fixed: fixed_size_queue.cpp
	$(CXXCMD) -o fixed_size_queue fixed_size_queue.cpp
