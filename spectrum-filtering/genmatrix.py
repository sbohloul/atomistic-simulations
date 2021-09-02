import numpy as np
from scipy import linalg
from scipy.stats import ortho_group



default_seed = 19840808

def gen_ortho_trans(dim, seed=default_seed):
    np.random.seed(seed=seed)
    return ortho_group.rvs(dim)


def chebyshev(n):
    if n == 0:
        return lambda x: 1
    elif n == 1:
        return lambda x: x
    else:
        return lambda x: 2 * x * chebyshev(n-1) - chebyshev(n-2)


class GenHamiltonian:

    def __init__(self, spectrum=None, seed=default_seed):

        self.random_seed = seed
        self.spectrum = spectrum
        self.spectrum_size = np.size(self.spectrum)        
        self.diag_representation = np.diag(spectrum)     
        self.orth_representation = self.ortho_trans()   

    def ortho_trans(self):
        trans_ortho = gen_ortho_trans(self.spectrum_size, seed=self.random_seed)
        self.trans_ortho = trans_ortho
        a = self.diag_representation
        a = np.dot(a, trans_ortho.T)
        a =  np.dot(trans_ortho, a)
        return .5 * (a + a.T)



         

# spectrum = np.linspace(-1, 1, 3)
# ham = GenHamiltonian(spectrum=spectrum)
# print(ham.diag_representation)
# print(ham.orth_representation)
