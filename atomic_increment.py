from threading import Thread
from threading import Lock
import itertools
import sys
from numpy import random

class ThreadSafeCounter():
    def __init__(self):
        self._counter = 0
        self._lock = Lock()

    def increment(self, v):
        with self._lock:
            self._counter += v

    def value(self):
        with self._lock:
            return self._counter

class FastWriteCounter(object):
    def __init__(self):
        self._number_of_read = 0
        self._counter = itertools.count()
        self._read_lock = Lock()

    def increment(self, val):
        next(itertools.zip_longest(*[self._counter]*val))

    def value(self):
        with self._read_lock:
            value = next(self._counter) - self._number_of_read
            self._number_of_read += 1
        return value

num_threads = int(sys.argv[1])
num_iterations = int(sys.argv[2])
def task(counter,i):
    for j in range(i, i + int(num_iterations / num_threads)):
        rng = random.default_rng(j)
        num = rng.integers(low=1,high=6,size=1)[0]
        counter.increment(num)

for ctype in [FastWriteCounter, ThreadSafeCounter]:
    counter = ctype()
    threads = [Thread(target=task, args=(counter,i,)) for i in range(num_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print(ctype,counter.value())
