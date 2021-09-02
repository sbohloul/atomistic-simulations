import numpy as np
import matplotlib.pyplot as plt
import timeit

# def memoized_chebyshev(n, D={}):
#     if n in D.keys():
#         return D[n]
#     else:
#         if n == 0:
#             result = lambda x: 1
#         elif n == 1:
#             result = lambda x: x
#         else: 
#             y_2 = lambda x: memoized_chebyshev(n-2)(x)
#             y_1 = lambda x: 2 * x * memoized_chebyshev(n-1)(x)
#             result = lambda x: y_1(x) - y_2(x)
#     D[n] = result
#     return result 

# def chebyshev(n):    
#     if n == 0:
#         return lambda x: x
#     elif n == 1:
#         return lambda h, x: np.dot(h, x)
#     else: 
#         y_2 = lambda x: memoized_chebyshev(n-2)(x)
#         y_1 = lambda x: 2 * x * memoized_chebyshev(n-1)(x)
#         return lambda h, x: 2 * np.dot(h, chebyshev(n-1)(h, x)) - chebyshev(n-2)(h, x)
def memoized_chebyshev_filter(n, D={}): 
    if n in D.keys():   
        return D[n]
    else:
        if n == 0:
            results = lambda h, x: x
        elif n == 1:
            results = lambda h, x: np.dot(h, x)
        else: 
            results = lambda h, x: 2 * np.dot(h, chebyshev_filter(n-1)(h, x)) - chebyshev_filter(n-2)(h, x)
    D[n] = results
    return results

def chebyshev_filter(n):    
    if n == 0:
        return lambda h, x: x
    elif n == 1:
        return lambda h, x: np.dot(h, x)
    else: 
        return lambda h, x: 2 * np.dot(h, chebyshev_filter(n-1)(h, x)) - chebyshev_filter(n-2)(h, x)


seed = 100
np.random.seed(seed=seed)

# h = np.random.rand(4, 4)
dim = 1000
x = np.diag(np.ones(dim))
h = np.random.rand(dim, dim)
print(x)
print(h)
n = 15

start_time = timeit.default_timer()
t_n = chebyshev_filter(n)
t_n(h, x)
print('Total time = {}'.format(timeit.default_timer() - start_time))

start_time = timeit.default_timer()
t_n = memoized_chebyshev_filter(n)
t_n(h, x)
print('Total time = {}'.format(timeit.default_timer() - start_time))

# x = 2
# t_0 = memoized_chebyshev(0)
# t_1 = memoized_chebyshev(1)
# t_2 = memoized_chebyshev(2)
# print(t_2(x))
# t_n = lambda x: 2 * x * t_1(x) - t_0(x)
# print(t_n(x))

# start_time = timeit.default_timer()
# t_15 = memoized_chebyshev(15)
# print(t_15(2))
# print('Total time = {}'.format(timeit.default_timer() - start_time))
# start_time = timeit.default_timer()
# c_15 = chebyshev(15)
# print(c_15(2))
# print('Total time = {}'.format(timeit.default_timer() - start_time))


# x = np.linspace(-1.01, 1.01, 1000)
# plt.figure()
# plt.box(True)
# plt.grid(True)
# for n in range(2, 8):
#     t_n = memoized_chebyshev(n)
#     y = t_n(x)
#     plt.plot(x, y)
# plt.show()

# def my_x(x):
#     return x

# y = lambda x: x * my_x(x)
# print(y(3))