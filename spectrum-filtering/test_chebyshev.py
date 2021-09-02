from spfilter import chebyshev, memoized_chebyshev
from spfilter import chebyshev_filter, memoized_chebyshev_filter, chebyshev_filter_shifted
from spfilter import linear_map
import numpy as np
from numpy import linalg as LA 


# =====================
# Chebyshev polynomials
# =====================
# t_n - 2 * x * t_n_1 + t_n_2 = 0
print("\nTest: t_n - 2 * x * t_n_1 + t_n_2 = 0")
for n in range(2, 10):
    t_n = chebyshev(n)
    t_n_1 = chebyshev(n-1)
    t_n_2 = chebyshev(n-2)
    x = 10
    y = t_n(x) - 2 * x * t_n_1(x) + t_n_2(x)
    print('n = {}, t_n - 2 * x * t_n_1 + t_n_2 = {}'.format(n , y))
    assert y == 0, 'Test one failed'

# t_n(1) = 1, t_n(1) = +1 or -1
print("\nTest: t_n(1) = 1, t_n(-1) = +1 or -1")
for n in range(10):
    t_n = chebyshev(n)
    print('n = {:2d}, t_n(-1) = {:2d}, t_n(1) = {:2d}'.format(n, t_n(-1), t_n(1)))

# =================
# Chebyshev filters
# =================
# T_n(h) |psi_i> = T_n(epsilon_i) |psi_i>
seed = 1000
np.random.seed(seed=seed)
dim = 10
h = np.random.rand(dim, dim)
h = .5 * (h + h.T)
eigvals, eigvecs = LA.eigh(h)
i = 1
x_i = eigvecs[:, i]
e_i = eigvals[i]

print("\nTest: tf_n(h, x_i) - t_n(e_i) * x_i = 0")
for n in range(10):
    t_n = chebyshev(n)
    tf_n = chebyshev_filter(n)
    y = tf_n(h, x_i) - t_n(e_i) * x_i 
    # print("n = {:2d}, tf_n(h, x_i) - t_n(e_i) * x_i = 0 is {}".format(n, np.allclose(y, 0)))
    print("n = {:2d}, tf_n(h, x_i) - t_n(e_i) * x_i = {:.4e}".format(n, np.max(y)))


# ==========
# linear map
# ==========
# T_n(L(h)) |psi_i> = T_n(L(epsilon_i)) |psi_i>
n_occ = 4
b = eigvals[-1]
a = eigvals[n_occ]
L = linear_map(a, b)

print("\nTest (shifted): tf_n(L(h), x_i) - t_n(L(e_i)) * x_i = 0")
for n in range(10):
    t_n = chebyshev(n)
    tf_n = chebyshev_filter_shifted(n, a, b)
    y = tf_n(h, x_i) - t_n(2 / (b - a) * e_i - (b + a) / (b - a)) * x_i
    # print("n = {:2d}, tf_n(L(h), x_i) - t_n(L(e_i)) * x_i = 0 is {}".format(n, np.allclose(y, 0)))
    print("n = {:2d}, tf_n(L(h), x_i) - t_n(L(e_i)) * x_i = {:.4e}".format(n, np.max(y)))

print("\nTest : tf_n(L(h), x_i) - t_n(L(e_i)) * x_i = 0")
for n in range(10):
    t_n = chebyshev(n)
    tf_n = chebyshev_filter(n)
    y = tf_n(L(h), x_i) - t_n(2 / (b - a) * e_i - (b + a) / (b - a)) * x_i
    # print("n = {:2d}, tf_n(L(h), x_i) - t_n(L(e_i)) * x_i = 0 is {}".format(n, np.allclose(y, 0)))
    print("n = {:2d}, tf_n(L(h), x_i) - t_n(L(e_i)) * x_i = {:.4e}".format(n, np.max(y)))

