import numpy as np
from numpy.core.arrayprint import printoptions
from numpy.linalg import eig
import spfilter as sf
import scipy.linalg as LA


def print_section(title):
    n_char = len(title)
    print("\n#", "=" * n_char)
    print("#", title)
    print("#", "=" * n_char)

def print_line(title, array):
    print(title, "=", ["{:.4e}".format(x) for x in array.tolist()])

rand_seed = 19840808
np.set_printoptions(precision=4)

# Spectrum parameters
print_section("Unperturbed spectrum")
dim, eigval_min, eigval_max = 10, -1, 1
spectrum = np.linspace(eigval_min, eigval_max, dim)
s = sf.Spectrum(spectrum=spectrum, seed=rand_seed)
h = s.orth_rep
eigvals, eigvecs = LA.eigh(h)
# print(f'eigvals = {eigvals}')
# print(f"eigvecs = {eigvecs}")
print_line("eigvals", eigvals)

# Subspace parameters
print_section("Subspace parameters")
sub_rank = 5
sub_dim = 5
print(f"sub_rank = {sub_rank}, sub_dim = {sub_dim}")

# =========================
# Contour filtered subspace
# =========================
print_section("Unperturbed contour")
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
    # print(f"eigvals_sub (abs acc) = {abs_acc}")
    # print(f"eigvals_sub (rel acc) = {rel_acc} %")
    print_line("eigvals_sub (abs acc)", abs_acc)
    print_line("eigvals_sub (rel acc)", rel_acc)

# =================
# Perturbed problem
# =================
print_section("Perturbed spectrum")

pert_para = 1e-5
print(f"pert_para = {pert_para}")
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
# print(f"eigvals **** = {eigvals}", sep=',')
# print(f"eigvals_pert = {eigvals_pert}")
print_line("eigvals unpe", eigvals)
print_line("eigvals_pert", eigvals_pert)

abs_acc = abs(eigvals_pert - eigvals)
rel_acc = abs_acc / abs(eigvals) * 1e2
# print(f"|eigvals_pert - eigvals| (abs) = {abs_acc}")
# print(f"|eigvals_pert - eigvals| (rel) = {rel_acc} %")
print_line("|eigvals_pert - eigvals| (abs)", abs_acc)
print_line("|eigvals_pert - eigvals| (rel)", rel_acc)

# Sternheimer
print_section("Sternheimer")
alpha = 1
print(f"alpha = {alpha}")
proj_c = lambda x: x - np.dot(eigvecs[:,:sub_rank], np.dot(eigvecs[:,:sub_rank].conjugate().T, x))
pv = eigvecs[:,:sub_rank] @ eigvecs[:,:sub_rank].conjugate().T
pc = np.eye(dim, dim) - pv
# print(eigvecs.T @ proj_c(eigvecs))

delta_eigvecs = []
if True:
    for epsilon, vec in zip(spectrum[:sub_rank], eigvecs[:,:sub_rank].T):
        # print(epsilon)
        A = h - epsilon * np.eye(dim, dim) - alpha * pv        
        b = delta_h @ vec
        b = -proj_c(b)
        # print(eigvecs.T @ b)
        delta_eigvecs.append(LA.solve(A, b, assume_a='sym'))
# print(delta_eigvecs)        
delta_eigvecs = np.stack(delta_eigvecs, axis=1)
# print(np.stack(delta_eigvecs, axis=0) - delta_eigvecs[0])
# print(delta_eigvecs)        

delta_dens_stern = np.sum(eigvecs[:,:sub_rank] * delta_eigvecs.conjugate(), 1)
delta_dens_stern = delta_dens_stern + np.sum(delta_eigvecs * eigvecs[:,:sub_rank].conjugate(), 1)
delta_dens_exact = np.sum(eigvecs_pert[:,:sub_rank] * eigvecs_pert[:,:sub_rank].conjugate(), 1)
delta_dens_exact = delta_dens_exact -  np.sum(eigvecs[:,:sub_rank] *  eigvecs[:,:sub_rank].conjugate(), 1)

print_line("delta_dens_exact", delta_dens_exact)
print_line("delta_dens_stern", delta_dens_stern)
print_line("|delta_den| (abs)", abs(delta_dens_exact - delta_dens_stern))
print_line("|delta_den| (rel)", abs(delta_dens_exact - delta_dens_stern) / delta_dens_exact )

# ========================
# Perturbed contour filter
# ========================
print_section("Perturbed contour filter")

x_min = -1.5
x_max = spectrum[sub_rank-1] + spectrum[sub_rank]
x_max = x_max / 2
radius = (x_max - x_min) / 2
print(f"Contour parameters:") 
print(f"x_min = {x_min}, x_max = {x_max}, radius = {radius}")

# A term
psi_n = eigvecs[:,:sub_rank]
# leggauss intergation parameters
n_leggauss = [30]
print(f"n_leggauss = {n_leggauss}")


print("\n----------------------------------------------------------")
print("Check A term: Q = int(zI - H)^-1 dz |psi-n> = 2pi*i* |psi_n>")
print("------------------------------------------------------------")
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
    fz = lambda theta: 1 / 2j / np.pi * LA.solve(A(theta), psi_n) 
    Q = sf.contour_filter(n, int_lim_1, int_lim_2, dz, fz)

    abs_acc = Q - psi_n   
    abs_acc = np.linalg.norm(abs_acc, np.inf, axis=0)
    print_line("norm(Q - psi_n, inf)", abs_acc)
    # rel_acc = abs_acc / abs(spectrum[:sub_rank]) * 1e2
    # print(f"eigvals_sub (abs acc) = {abs_acc}")
    # print(f"eigvals_sub (rel acc) = {rel_acc} %")

# Binomial Expansion
print_section("Binomial expansion")
x_rand = sf.gen_random_vec(dim=(dim, sub_rank), seed=rand_seed)
A_pert = lambda theta: z(theta) * np.eye(dim, dim) - h_pert  
A_h = lambda theta: z(theta) * np.eye(dim, dim) - h

n_theta = 5
theta_list = [x * 2 * np.pi / n_theta for x in range(n_theta+1)]

print("\n------------------------------------------------------------")
print("Factorization: zI - H - dH = (zI - H)*(I - deltaH*(zI-H)^-1)")
print("------------------------------------------------------------")
print("b1 = (zI - h_pert) * X")
print("b2 = (I - delta_h * (zI - H)^-1) * (zI - h) * X")
print("b3 = (zI - h) * (I - (zI - H)^-1 * delta_h) * X")

for theta in theta_list:
    print(f"\ntheta = {theta}, z = {z(theta)}")
    A_pert_sym = A_pert(theta)
    A_pert_sym = .5 * (A_pert_sym.conjugate().T + A_pert_sym)
    # (zI - h_pert) @ X
    b1 = A_pert_sym @ x_rand
    A_h_sym = A_h(theta)
    A_h_sym = .5 * (A_h_sym.conjugate().T + A_h_sym)
    # (I - delta_h @ (zI - H)^-1) * (zI - h)  * X
    b2 = A_h_sym @ x_rand
    b2 = b2 - delta_h @ LA.solve(A_h_sym, b2, assume_a='sym')
    # (zI - h) * (I - (zI - H)^-1 * delta_h) * X 
    b3 = x_rand - LA.solve(A_h_sym, delta_h @ x_rand)
    b3 = A_h_sym @ b3
    abs_acc = b1 - b2   
    abs_acc = np.linalg.norm(abs_acc, np.inf, axis=0)
    print_line("norm(b1, b2, inf)", abs_acc)
    abs_acc = b1 - b3   
    abs_acc = np.linalg.norm(abs_acc, np.inf, axis=0)
    print_line("norm(b1, b3, inf)", abs_acc)

print("\n-------")
print("Expansion")
print("---------")
print("b1 = (zI - h_pert) * X")
print("b2 = (I - delta_h * (zI - H)^-1) * (zI - h) * X")
print("b3 = (zI - h) * (I - (zI - H)^-1 * delta_h) * X")
for theta in theta_list:
    print(f"\ntheta = {theta}, z = {z(theta)}")
    A_pert_sym = A_pert(theta)
    A_pert_sym = .5 * (A_pert_sym.conjugate().T + A_pert_sym)
    # (zI - h_pert)^-1 @ X
    b1 = LA.solve(A_pert_sym, x_rand, assume_a='sym')
    A_h_sym = A_h(theta)
    A_h_sym = .5 * (A_h_sym.conjugate().T + A_h_sym)  
    # (I + (zI - H)^-1 * delta_h) * (zI - h)^-1 @ X
    b2 = LA.solve(A_h_sym, x_rand, assume_a='sym')     
    b2 = b2 + LA.solve(A_h_sym, delta_h @ b2, assume_a='sym')
    # * (zI - h)^-1 * (I - delta_H * (zI - H)^-1)^-1 * X
    b3 = LA.inv(np.eye(dim, dim) - delta_h @ LA.inv(A_h_sym)) @ x_rand
    b3 = LA.solve(A_h_sym, b3)
    abs_acc = b1 - b2   
    abs_acc = np.linalg.norm(abs_acc, np.inf, axis=0)
    print(abs_acc)
    abs_acc = b1 - b3
    abs_acc = np.linalg.norm(abs_acc, np.inf, axis=0)
    print(abs_acc)    