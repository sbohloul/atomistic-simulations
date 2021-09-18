import numpy as np
from numpy.core.arrayprint import printoptions
from numpy.linalg import eig
import spfilter as sf
import scipy.linalg as LA


rand_seed = 19840808
np.set_printoptions(precision=4)

# Spectrum parameters
dim, eigval_min, eigval_max = 10, -1, 1
spectrum = np.linspace(eigval_min, eigval_max, dim)
s = sf.Spectrum(spectrum=spectrum, seed=rand_seed)
h = s.orth_rep
eigvals, eigvecs = LA.eigh(h)
print(f'eigvals = {eigvals}')
# print(f"eigvecs = {eigvecs}")

# Subspace parameters
sub_rank = 5
sub_dim = 5


# =========================
# Contour filtered subspace
# =========================
x_rand = sf.gen_random_vec(dim=(dim, sub_rank), seed=rand_seed)

# contour parameters
x_min = -1.5
x_max = spectrum[sub_rank-1] + spectrum[sub_rank]
x_max = x_max / 2
radius = (x_max - x_min) / 2
print(f"Contour parameters:") 
print(f"x_min = {x_min}, x_max = {x_max}, radius = {radius}")

# leggauss intergation parameters
n_leggauss = [30]
print(f"n_leggauss = {n_leggauss}")

# perform the contour integration
for n in n_leggauss:
    print(f"n = {n}")

    # Integral parameters
    int_lim_1 = 0
    int_lim_2 = 2 * np.pi
    z = lambda theta: (x_max + x_min) / 2 + radius * np.exp(1j * theta)
    dz = lambda theta: 1j * radius * np.exp(1j * theta)
    A = lambda theta: z(theta) * np.eye(dim, dim) - h
    # 
    fz = lambda theta: 1 / 2j / np.pi * LA.solve(A(theta), x_rand) 
    Q = sf.contour_filter(n, int_lim_1, int_lim_2, dz, fz)
    sub_basis = Q
    # 
    B = sub_basis.conjugate().T @ sub_basis
    h_sub = sf.subspace_proj(h, sub_basis)
    eigvals_sub, eigvecs_sub = LA.eigh(h_sub, B)
    abs_acc = abs(eigvals_sub - spectrum[:sub_rank])
    rel_acc = abs_acc / abs(spectrum[:sub_rank]) * 1e2
    print(f"eigvals_sub (abs acc) = {abs_acc}")
    print(f"eigvals_sub (rel acc) = {rel_acc} %")

# =================
# Perturbed problem
# =================
pert_para = 1e-5
delta_h = np.random.rand(dim, dim)
delta_h = .5 * delta_h
if np.any(np.iscomplex(delta_h)):
    delta_h += delta_h.conjugate().T
else:
    delta_h += delta_h.T
delta_h = pert_para * delta_h
# 
h_pert = h + delta_h
eigvals_pert, eigvecs_pert = LA.eigh(h_pert)
print(f"eigvals **** = {eigvals}")
print(f"eigvals_pert = {eigvals_pert}")
abs_acc = abs(eigvals_pert - eigvals)
rel_acc = abs_acc / abs(eigvals) * 1e2
print(f"|eigvals_pert - eigvals| (abs) = {abs_acc}")
print(f"|eigvals_pert - eigvals| (rel) = {rel_acc} %")

# Sternheimer
alpha = 1
proj_c = lambda x: x - np.dot(eigvecs[:,:sub_rank], np.dot(eigvecs[:,:sub_rank].conjugate().T, x))
pc = np.eye(dim, dim) - eigvecs[:,:sub_rank] @ eigvecs[:,:sub_rank].conjugate().T
pv = np.eye(dim, dim) - pc
# print(eigvecs.T @ proj_c(eigvecs))

# psi_0 = eigvecs[:,0]
# e_0 = eigvals[0]
# A = h - e_0 * np.eye(dim, dim) - alpha * pv
# b = delta_h @ psi_0
# b = -proj_c(psi_0)
# d_psi = LA.solve(A, b)
# d_dens = psi_0 * d_psi.conjugate().T
# d_dens = d_dens + psi_0.conjugate().T * d_psi
# d_dens_exact = eigvecs_pert[:,0] * eigvecs_pert[:,0].conjugate().T
# d_dens_exact = d_dens_exact - psi_0 * psi_0.conjugate().T

# print("d_dens")
# print(d_dens)
# print(d_dens_exact)

# print("psi_0")
# print(psi_0)
# print(d_psi)
# print(eigvecs_pert[0] - psi_0)

# tmp = eigvecs.conjugate().T @ delta_h
# tmp = tmp @ eigvecs
# print(np.diag(tmp))
# print(abs_acc)

delta_eigvecs = []
if True:
    for epsilon, vec in zip(spectrum[:sub_rank], eigvecs[:,:sub_rank].T):
        print(epsilon)
        A = h - epsilon * np.eye(dim, dim) - alpha * pv        
        b = delta_h @ vec
        b = -proj_c(b)
        # print(eigvecs.T @ b)
        delta_eigvecs.append(LA.solve(A, b, assume_a='sym'))
# print(delta_eigvecs)        
delta_eigvecs = np.stack(delta_eigvecs, axis=1)
# print(np.stack(delta_eigvecs, axis=0) - delta_eigvecs[0])
# print(delta_eigvecs)        

# delta_eigvecs_exact = eigvecs_pert - eigvecs
# delta_eigvecs_exact = proj_c(delta_eigvecs_exact)
# print(delta_eigvecs - delta_eigvecs_exact[:,:sub_rank])
# print(delta_eigvecs)
# print(delta_eigvecs_exact[:,:sub_rank])

d_dens = np.sum(eigvecs[:,:sub_rank] * delta_eigvecs.conjugate(), 1)
d_dens = d_dens + np.sum(delta_eigvecs * eigvecs[:,:sub_rank].conjugate(), 1)
d_dens_exact = np.sum(eigvecs_pert[:,:sub_rank] * eigvecs_pert[:,:sub_rank].conjugate(), 1)
d_dens_exact = d_dens_exact -  np.sum(eigvecs[:,:sub_rank] *  eigvecs[:,:sub_rank].conjugate(), 1)

print("d_dens")
print(d_dens)
print(d_dens_exact)
print(abs(d_dens - d_dens_exact))
