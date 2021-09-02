import numpy as np
import matplotlib.pyplot as plt


# physical constants
k = 0.1
epsilon_0 = 1.0

# real-space grid
lat_para = 1.0
n_points = 5
grid_points = np.linspace( -lat_para/2, lat_para/2, n_points)

# atoms
n_atoms = 1
atoms_position = [0.0 for i_atom in range(n_atoms)]
atoms_charge = [1.0 for i_atom in range(n_atoms)]
atoms_sigma = [0.5 for i_atom in range(n_atoms)]

# rho atom
rho_atom = []
for pos, chg, sigma in zip(atoms_position, atoms_charge, atoms_sigma):

    # print('pos = {}'.format(pos))
    # print('chg = {}'.format(chg))

    dist = grid_points - pos
    # print(dist)

    rho = - chg / np.sqrt(2 * np.pi * sigma**2)
    rho = rho * np.exp(-.5 * (dist / sigma)**2 )
    rho_atom.append(rho)


minx = lambda x, y: np.abs(y-x) - np.round(np.abs(y-x) / lat_para) * lat_para

print(grid_points)
for i_point in range(n_points):
    for j_point in range(n_points):

        print(minx(grid_points[i_point], grid_points[j_point]))

# fig = plt.figure()
# for rho in rho_atom:
#     plt.plot(grid_points, rho)


# plt.show()
