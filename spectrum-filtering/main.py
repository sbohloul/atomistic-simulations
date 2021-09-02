import numpy as np
import scipy.linalg as LA
from scipy.stats import ortho_group


seed = 233423
np.random.seed(seed=seed)

# ==========
# parameters
# ==========
min_val = -1
max_val = 1
num_val = 10
dim = num_val
# 
sub_dim = 5
n_span = 5
# 
n_power = 10
# 
n_chebyshev = 10
lower_b = 1
upper_b = 


# ========
# spectrum
# ========
spectrum = np.linspace(min_val, max_val, num_val)
ham_diag = np.diag(spectrum)
print('eigvals = ', spectrum)


# ===============
# transformations
# ===============
# orthogonal transformation matrix
trans_ortho = ortho_group.rvs(dim)


# =========
# rotations
# =========
ham_ortho = np.dot(ham_diag, trans_ortho.T)
ham_ortho = np.dot(trans_ortho, ham_ortho)


# ===========
# diagonalize
# ===========
ham_mat = .5 * (ham_ortho + ham_ortho.T)
eigvals_ortho, eigvecs_ortho = LA.eigh(ham_mat)
print('eigvals_ortho = ', eigvals_ortho)
# print('eigvecs_ortho = ', eigvecs_ortho)


# ==============
# subspace basis
# ==============
a_coeff = np.random.rand(n_span, sub_dim)
sub_basis = np.dot(eigvecs_ortho[:, :n_span], a_coeff)
print('sub_basis.shape = ', sub_basis.shape)

sub_basis = LA.orth(sub_basis)
print('sub_basis.shape = ', sub_basis.shape)
# print(np.dot(sub_basis.T, sub_basis))

ham_sub = np.dot(ham_ortho, sub_basis)
ham_sub = np.dot(sub_basis.T, ham_sub)

ham_sub = .5 * (ham_sub + ham_sub.T)
print(LA.eigvalsh(ham_sub))


# ====================
# power filtered basis
# ====================
def power_n(n, h, x):
    if n != 1:
        return np.dot(h, power_n(n-1, h, x))
    else:
        return np.dot(h, x)
n = 5
rand_basis = np.random.rand(dim, sub_dim)
rand_basis = power_n(n, ham_ortho, rand_basis)

print('sub_basis.shape = ', rand_basis.shape)

rand_basis = LA.orth(rand_basis)
print('sub_basis.shape = ', rand_basis.shape)
# print(np.dot(rand_basis.T, rand_basis))

ham_sub = np.dot(ham_ortho, rand_basis)
ham_sub = np.dot(rand_basis.T, ham_sub)
ham_sub = .5 * (ham_sub + ham_sub.T)
print(LA.eigvalsh(ham_sub))


# ========================
# chebyshev filtered basis
# ========================
def chebyshev_n(n, h, x, a, b):
    if n != 0 and n != 1:
        return 4 / (a + b) * ( np.dot(h, chebyshev_n(n-1, h, x, a, b)) - (b - a) / 2 * chebyshev_n(n-1, h, x, a, b) ) - \
                    chebyshev_n(n-2, h, x, a, b) 
    elif n == 1:
        return 2 / (b + a) * ( np.dot(h, x) - (b - a) / 2 * x ) 
    elif n == 0:
        return x

a = spectrum[sub_dim]
b = spectrum[-1]
n = 15
rand_basis = np.random.rand(dim, sub_dim)
rand_basis = chebyshev_n(n, ham_ortho, rand_basis, a, b)

print('sub_basis.shape = ', rand_basis.shape)

rand_basis = LA.orth(rand_basis)
print('sub_basis.shape = ', rand_basis.shape)
# print(np.dot(rand_basis.T, rand_basis))

ham_sub = np.dot(ham_ortho, rand_basis)
ham_sub = np.dot(rand_basis.T, ham_sub)
ham_sub = .5 * (ham_sub + ham_sub.T)
print(LA.eigvalsh(ham_sub))