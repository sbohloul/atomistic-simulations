import numpy as np
import scipy.linalg as LA
from spfilter import Spectrum, subspace_proj



rand_seed = 1000
np.set_printoptions(precision=4)

# Hamiltonian and parameters
spectrum = np.linspace(-1, 1, 10)
# spectrum = np.arange(-5, 6)
s = Spectrum(spectrum, seed=rand_seed)
h = s.orth_rep
eigvals, eigvecs = LA.eigh(h)

print(spectrum)
print(f" |eigvals - ref|  = {np.abs(eigvals - spectrum)}")
# print(np.diag(np.dot(eigvecs.T, eigvecs)))

# ========
# Subspace
# ========
dim_sub = 7
rank_sub = 7
coef = np.random.rand(rank_sub, dim_sub)
sub_span = eigvecs[:, :rank_sub]
sub_basis = np.dot(sub_span, coef)
sub_basis = LA.orth(sub_basis)
print(f"sub_basis^d * sub_basis = {np.diag(np.dot(sub_basis.T, sub_basis))}")

# Project into subspace
h_sub = subspace_proj(h, sub_basis)
o_sub = np.dot(np.conjugate(sub_basis.T), sub_basis)
o_sub = .5 * (np.conjugate(o_sub.T) + o_sub)
# print(h_sub)
# print(o_sub)
# Generalize eigenvalue problem in subspace
eigvals_sub, eigvecs_sub = LA.eigh(h_sub, o_sub)
print(f"eigvals_sub - ref = {np.abs(eigvals_sub - spectrum[:dim_sub])}")

print(f"eigvec_sub^d * eigvecs_sub = {np.diag(np.dot(eigvecs_sub.T.conjugate(), eigvecs_sub))}")
exp_itheta = eigvecs[:, :rank_sub] / (sub_basis @ eigvecs_sub)
tmp = eigvecs[:, :rank_sub] @ np.diag(exp_itheta[0, :]) - sub_basis @ eigvecs_sub[:, :rank_sub]
print(f"norm(eigvecs - sub_basis * eigvec_sub) = {np.linalg.norm(tmp, np.inf)}")
# ==============
# Fermi-Dirac fu
# ==============
mu = .4
ind_mu = 5
# beta = 100 #  k * 300 kelvin in meV
beta = 1e8 # meV

fd_fun = lambda x, beta, mu: 1 / (1 + np.exp(beta * (x - mu)))
fd_diag = np.diag(fd_fun(spectrum, beta, mu))
print(f"fd_diag = {np.diag(fd_diag)}")

fd_orth = np.dot(fd_diag, np.conjugate(eigvecs.T))
fd_orth = np.dot(eigvecs, fd_orth)
# print(f"fd_orth = {fd_orth}")
tmp = np.dot(fd_orth, eigvecs) - np.dot(eigvecs, fd_diag)
print(f"nomr(f(H) psi - f(e_) psi) = {np.linalg.norm(tmp, np.inf)}")
print(f"trace(fd_diag = {np.trace(fd_diag)}")
print(f"trace(fd_orth = {np.trace(fd_orth)}")

# ==============
# Density matrix
# ==============
den_mat_orth = eigvecs @ fd_diag
den_mat_orth = den_mat_orth @ np.conjugate(eigvecs.T)
# print(den_mat_orth - fd_orth)
print(f"trace(den_mat__orth = {np.trace(den_mat_orth)}")

fd_diag_sub = np.diag(fd_fun(spectrum[:rank_sub], beta, mu))
print(f"fd_diag_sub = {fd_diag_sub}")
den_mat_sub = np.dot(eigvecs_sub, fd_diag_sub)
den_mat_sub = np.dot(den_mat_sub, eigvecs_sub.T.conjugate())
print(f"trace(den_mat_sub = {np.trace(den_mat_sub)}")
print(f"trace(h = {np.trace(h)}")
print(f"trace(h_sub = {np.trace(h_sub)}")
print(f"den_mat_sub = {den_mat_sub}")

tmp = (sub_basis @ den_mat_sub) @ sub_basis.T.conjugate()
print(f"norm(dens_mat_orth - phi * dens_mat_sub * phi) = {np.linalg.norm(den_mat_orth - tmp, np.inf)}")
tmp = np.eye(dim_sub, dim_sub) + eigvecs_sub[:,ind_mu:] @ (fd_diag_sub[ind_mu:, ind_mu:] - np.eye(2, 2)) @ eigvecs_sub[:,ind_mu:].T.conjugate()
print(tmp - den_mat_sub)
# print(den_mat_orth)
