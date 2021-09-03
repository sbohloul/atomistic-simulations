import numpy as np
from scipy import linalg
from scipy.stats import ortho_group
from numpy.polynomial.legendre import leggauss

default_seed = 1000

# =================
# Chebyshev filters
# =================
def chebyshev_filter(n):
    if n == 0:
        return lambda h, x: x
    elif n == 1:
        return lambda h, x: np.dot(h, x)
    else:
        return lambda h, x: 2 * np.dot(h, chebyshev_filter(n-1)(h, x)) - chebyshev_filter(n-2)(h, x)

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

def chebyshev_filter_shifted(n, a, b):
    if n == 0:
        return lambda h, x: x
    elif n == 1:
        return lambda h, x: 2 / (b - a) * np.dot(h, x) - (b + a) / (b - a) * x
    else:        
        return lambda h, x: 4 / (b - a) * np.dot(h, chebyshev_filter_shifted(n-1, a, b)(h, x)) - 2 * (b + a) / (b - a) * chebyshev_filter_shifted(n-1, a, b)(h, x) - chebyshev_filter_shifted(n-2, a, b)(h, x)



# =====================
# Chebyshev polynomials
# =====================
def chebyshev(n):
    if n == 0:
        return lambda x: 1
    elif n == 1:
        return lambda x: x
    else:
        return lambda x: 2 * x * chebyshev(n-1)(x) - chebyshev(n-2)(x)    

def memoized_chebyshev(n, D={}):
    if n in D.keys():
        return D[n]
    else:
        if n == 0:
            results = lambda x: 1
        elif n == 1:
            results = lambda x: x
        else:
            results = lambda x: 2 * x * chebyshev(n-1)(x) - chebyshev(n-2)(x)            
    D[n] = results
    return D


# ============
# power filter
# ============
def power_filter(n):
    if n == 1:
        return lambda h, x: np.dot(h, x)
    else:
        return lambda h, x: np.dot(h, power_filter(n-1)(x))

def memoized_power_filter(n, D={}):
    if n in D.keys():
        return D[n]
    else:
        if n == 1:
            results = lambda h, x: np.dot(h, x)
        else:
            results = lambda h, x: np.dot(h, power_filter(n-1)(x))        
    D[n] = results
    return D


# ==============
# Contour filter
# ==============
def contour_filter(n, int_lim_1, int_lim_2, dz, fz):
    x_lg, w_lg = leggauss(n)
    int_val = 0
    for x_i, w_i in zip(x_lg, w_lg):
        t_i = (int_lim_2 - int_lim_1) / 2 * x_i + (int_lim_2 + int_lim_1) / 2
        fz_i = fz(t_i)
        dz_i = dz(t_i) * (int_lim_2 - int_lim_1) / 2

        int_val += w_i * fz_i * dz_i
    return int_val

# =======================
# Transformation matrices
# =======================
def gen_ortho_trans(dim, seed):
    np.random.seed(seed)
    return ortho_group.rvs(dim)


# ==============
# random vectors
# ==============
def gen_random_vec(dim, seed=default_seed):
    np.random.seed(seed=seed)
    return np.random.rand(dim[0], dim[1])

# ==========
# Linear map
# ==========
def linear_map(lower_b, upper_b):
    return lambda x: 2 / (upper_b - lower_b) * x - (upper_b + lower_b) / (upper_b - lower_b) * np.eye(x.shape[0], x.shape[1])


# ===================
# Subspace projection
# ===================
def subspace_proj(h, basis):
    h = np.dot(h, basis)
    if np.any(np.iscomplex(basis)):
        print("IS COMPLEX")
        basis = np.conjugate(basis)
    h = np.dot(basis.T, h)
    if np.any(np.iscomplex(h)):
        print("IS COMPLEX")
        return .5 * (h + np.conjugate(h.T))
    else:
        return .5 * (h + h.T)

# =======
# Classes
# =======
class Spectrum:

    def __init__(self, spectrum=None, seed=default_seed):

        self.spectrum = spectrum
        self.spectrum_size = np.size(self.spectrum)        
        self.diag_rep = np.diag(spectrum)     
        self.random_seed = seed
        
        self.orth_rep = self.ortho_trans()   

    def ortho_trans(self):
        ortho_trans = gen_ortho_trans(self.spectrum_size, seed=self.random_seed)
        self.ortho_trans = ortho_trans
        a = self.diag_rep
        a = np.dot(a, ortho_trans.T)
        a =  np.dot(ortho_trans, a)
        return .5 * (a + a.T)

    def gen_initial_vec(self, dim):
        self.initial_vec = gen_random_vec(dim, self.random_seed)




         

# spectrum = np.linspace(-1, 1, 3)
# ham = GenHamiltonian(spectrum=spectrum)
# print(ham.diag_rep)
# print(ham.orth_rep)
