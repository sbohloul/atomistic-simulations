import numpy as np
from numpy.polynomial.legendre import leggauss
import scipy.linalg as LA
from spfilter import Spectrum, chebyshev_filter, gen_random_vec, linear_map, subspace_proj

rand_seed = 1000
np.set_printoptions(precision=4)

# Hamilotnian and parameters
spectrum = np.linspace(-1, 1, 10)
s = Spectrum(spectrum, seed=rand_seed)
h = s.orth_rep
eigvals, eigvecs = LA.eigh(h)

n_occ = 5
lbound_ind = 5
subpace_dim = [h.shape[0], n_occ]

print('n_occ = {}'.format(n_occ))
print('lbound_ind = {}'.format(lbound_ind))
print('subspace subpace_dim = {}'.format(subpace_dim))


# ===========================
# Chebyshev filtered subspace
# ===========================
print('\n===========================')
print('Chebyshev filtered subspace')
print('===========================')

chebpoly_n = 15
print('chebpoly_n = ', chebpoly_n)
t_n = chebyshev_filter(chebpoly_n)
a = s.spectrum[:lbound_ind][-1]
b = s.spectrum[-1]
L = linear_map(a, b)
x0 = gen_random_vec(subpace_dim, seed=rand_seed)

# generate orthonormal basis for subspace
cheb_basis = t_n(L(h), x0)
print('cheb_basis.shape (filt) = {}'.format(cheb_basis.shape))
cheb_basis = LA.orth(cheb_basis)
print('cheb_basis.shape (orth) = {}'.format(cheb_basis.shape))
assert cheb_basis.shape[1] == n_occ, "Number of filtered basis smaller not equal to requested (n_occ) "

# Subspace projection
h_sub_cheb = subspace_proj(h, cheb_basis)
eigvals_sub_cheb, eigvecs_sub_cheb = LA.eigh(h_sub_cheb)
print("eigvals_sub_cheb = ", eigvals_sub_cheb)
print("eigvals_sub_cheb (acc abs) = ", eigvals_sub_cheb - s.spectrum[:n_occ])
print("eigvals_sub_cheb (acc rel) = ", abs(eigvals_sub_cheb - s.spectrum[:n_occ]) / abs(s.spectrum[:n_occ]) * 1e2)


# ========================
# Custom filtered subspace
# ========================
print("\n========================")
print("Custom filtered subspace")
print("========================")
coef0 = gen_random_vec([lbound_ind, n_occ], seed=rand_seed)
print('coef0.shape = ', coef0.shape)
cust_basis = np.dot(eigvecs[:, :lbound_ind], coef0)
print('cust_basis.shape (filt) = {}'.format(cust_basis.shape))
cust_basis_orth = LA.orth(cust_basis)
print('cust_basis.shape (orth) = {}'.format(cust_basis.shape))
h_sub_cust = subspace_proj(h, cust_basis)
h_sub_cust_roth = subspace_proj(h, cust_basis_orth)

B = np.dot(cust_basis.T, cust_basis)
eigvals_sub_cust, eigvecs_sub_cust = LA.eigh(h_sub_cust, B)
print("eigvals_sub_cust = ", eigvals_sub_cust)
print("eigvals_sub_cust (acc abs) = ", eigvals_sub_cust - s.spectrum[:n_occ])
print("eigvals_sub_cust (acc rel) = ", abs(eigvals_sub_cust - s.spectrum[:n_occ]) / abs(s.spectrum[:n_occ]) * 1e2)

eigvals_sub_cust_orth, eigvecs_sub_cust_orth = LA.eigh(h_sub_cust_roth)
print("eigvals_sub_cust = ", eigvals_sub_cust_orth)
print("eigvals_sub_cust_orth (acc abs) = ", eigvals_sub_cust_orth - s.spectrum[:n_occ])
print("eigvals_sub_cust_orth (acc rel) = ", abs(eigvals_sub_cust_orth - s.spectrum[:n_occ]) / abs(s.spectrum[:n_occ]) * 1e2)


# =========================
# Contour filtered subspace
# =========================
print("\n========================")
print("Contour filtered subspace")
print("========================")

n_gauss = 15
gauss_x0, gauss_w0 = leggauss(n_gauss)
a = s.spectrum[0] -.5
b = s.spectrum[:lbound_ind][-1] + 0.02
r = (b - a) / 2

Q = 0 * x0
for x_i, w_i in zip(gauss_x0, gauss_w0):

    theta = np.pi / 2 * (1 - x_i)
    Z = (b + a) / 2 + r * np.exp(1j * theta)
    print("theta = {}, Z = {}".format(theta, Z))

    A = Z * np.eye(h.shape[0], h.shape[1]) - h
    X = LA.solve(A, x0)
    XT = LA.solve(A.T, x0)

    tmp = np.exp(1j * theta) * X 
    tmp = tmp + np.exp(-1j * theta) * XT
    Q = Q - w_i * tmp

cont_basis = 1 / 4 * r * Q
print('cont_basis.shape = {}'.format(cont_basis.shape))
# cont_basis = LA.orth(cont_basis)
# print('cont_basis.shape (orth) = {}'.format(cont_basis.shape))
h_sub_cont = subspace_proj(h, cont_basis)
B = np.dot(np.conjugate(cont_basis.T), cont_basis)
eigvals_sub_cont, eigvecs_sub_cont = LA.eigh(h_sub_cont, B)
print("eigvals_sub_cont = ", eigvals_sub_cont)
print("eigvals_sub_cont (acc abs) = ", eigvals_sub_cont - s.spectrum[:n_occ])
print("eigvals_sub_cont (acc rel) = ", abs(eigvals_sub_cont - s.spectrum[:n_occ]) / abs(s.spectrum[:n_occ]) * 1e2)
    






# import matplotlib.pyplot as plt
# # plt.figure()
# plt.matshow(np.diag(eigvals))
# plt.matshow(h)
# plt.matshow(eigvecs)
# plt.show()