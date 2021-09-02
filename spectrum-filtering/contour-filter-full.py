import numpy as np
from numpy.polynomial.legendre import leggauss
import scipy.linalg as LA
from spfilter import Spectrum, chebyshev_filter, gen_random_vec, linear_map, subspace_proj

rand_seed = 1000
np.set_printoptions(precision=4)

# Hamiltonian and parameters
spectrum = np.linspace(-1, 1, 10)
# spectrum = np.arange(-5, 6)
s = Spectrum(spectrum, seed=rand_seed)
h = s.orth_rep
eigvals, eigvecs = LA.eigh(h)
# print(h)

n_occ = 5
lbound_ind = 5
subpace_dim = [h.shape[0], n_occ]

print('n_occ = {}'.format(n_occ))
print('lbound_ind = {}'.format(lbound_ind))
print('subspace subpace_dim = {}'.format(subpace_dim))
print('spectrum = ', spectrum)


x0 = gen_random_vec(subpace_dim, seed=rand_seed) 
# x0 = eigvecs[:,:n_occ]

# 
for n_gauss in range(1, 20):
    print("n_gauss = {}".format(n_gauss))
    gauss_x0, gauss_w0 = leggauss(n_gauss)
    # a = s.spectrum[0]
    # b = s.spectrum[:lbound_ind][-1]
    a = -1.5
    b = 0
    r = (b - a) / 2
    print("a = {}, b = {}, r = {}".format(a, b, r))

    Q = 0 * x0
    for x_i, w_i in zip(gauss_x0, gauss_w0):

        theta = np.pi / 2 * (1 - x_i)
        Z = (b + a) / 2 + r * np.exp(1j * theta)
        # print("theta = {}, Z = {}".format(theta, Z))

        A = Z * np.eye(h.shape[0], h.shape[1]) - h
        X = LA.solve(A, x0)

        tmp = np.real(np.exp(1j * theta) * X)
        Q = Q - w_i * tmp

    cont_basis = 1 / 2 * r * Q
    print('cont_basis.shape = {}'.format(cont_basis.shape))
    cont_basis = LA.orth(cont_basis)
    print('cont_basis.shape (orth) = {}'.format(cont_basis.shape))

    h_sub_cont = np.dot(h, cont_basis)
    h_sub_cont = np.dot(np.conjugate(cont_basis.T), h_sub_cont)
    h_sub_cont = .5 * (h_sub_cont + np.conjugate(h_sub_cont.T))
    # print(h_sub_cont)
    
    B = None
    # B = np.dot(np.conjugate(cont_basis.T), cont_basis)
    # B = .5 * (B + np.conjugate(B.T))
    if B is not None:
        eigvals_sub_cont, eigvecs_sub_cont = LA.eigh(h_sub_cont, B)
    else:
        eigvals_sub_cont, eigvecs_sub_cont = LA.eigh(h_sub_cont)

    print("eigvals_sub_cont = ", eigvals_sub_cont)
    print("eigvals_sub_cont (acc abs) = ", eigvals_sub_cont - s.spectrum[:n_occ])
    print("eigvals_sub_cont (acc rel) = ", abs(eigvals_sub_cont - s.spectrum[:n_occ]) / abs(s.spectrum[:n_occ]) * 1e2)
    
