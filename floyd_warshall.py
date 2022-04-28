import numpy as np
from timeit import timeit
from numba import jit

# Too slow, don't bother
#def floyd_warshall():
#    dsize = 200
#    d = np.zeros((dsize, dsize), dtype=np.int)
#    for k in range(dsize):
#        for i in range(dsize):
#            for j in range(dsize):
#                d[i][j] = max(d[i][j], d[i][k] + d[k][j])
#    return d[-1][-1]

def floyd_warshall2():
    dsize = 2023
    d = np.zeros((dsize, dsize), dtype=np.int32)
    for k in range(dsize):
        for i in range(dsize):
            d[i] = np.maximum(d[i], d[k] + d[i][k])
    return d[-1][-1]

@jit(nopython=True)
def floyd_warshall3():
    dsize = 2023
    d = np.zeros((dsize, dsize), dtype=np.int32)
    for k in range(dsize):
        for i in range(dsize):
            d[i] = np.maximum(d[i], d[k] + d[i][k])
    return d[-1][-1]

factor = 1000000
print(f'NumPy\t0\t{timeit(lambda: floyd_warshall2(), number=1)*factor}')
print(f'Numba\t0\t{timeit(lambda: floyd_warshall3(), number=1)*factor}')
