import numpy as np
from numpy.polynomial.legendre import leggauss
import scipy.linalg as LA
from spfilter import Spectrum, chebyshev_filter, gen_random_vec, linear_map, subspace_proj, contour_filter

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
print(f'spectrum = {spectrum[:]}')

x0 = gen_random_vec(subpace_dim, seed=rand_seed) 
# x0 = eigvecs[:,:n_occ]

x_min = -1.5
x_max = spectrum[n_occ-1] + spectrum[n_occ]
x_max = x_max / 2 
r = (x_max - x_min) / 2
print(f"x_min = {x_min}, x_max = {x_max}, r = {r}")
for n in range(1, 21, 5):
    print("n = {}".format(n))
    x_lg, w_lg = leggauss(n)

    int_lim_1 = 0
    int_lim_2 = 2 * np.pi
    z = lambda theta: (x_max + x_min) / 2 + r * np.exp(1j * theta)
    A = lambda theta: z(theta) * np.eye(h.shape[0], h.shape[1]) - h
    fz = lambda theta: 1 / 2 / np.pi / 1j * LA.solve(A(theta), x0)
    dz = lambda theta: 1j * r * np.exp(1j * theta)

    Q = contour_filter(n,int_lim_1, int_lim_2, dz, fz)
    cont_basis = Q
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
    
