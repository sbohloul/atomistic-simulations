import numpy as np
from numpy.polynomial.legendre import leggauss
from spfilter import contour_filter

if False:
    n = 5
    gauss_x0, gauss_w0 = leggauss(n)

    for x, w in zip(gauss_x0, gauss_w0):
        print("x0 = {:16.12f}, w0 = {:16.12f}".format(x, w))

    y = lambda x: np.exp(x)
    lin_map = lambda a, b, x: (b - a) / 2 * x + (b + a) / 2

    int_lim = [-3, 3]
    x1 = int_lim[0]
    x2 = int_lim[1]
    fx = y(lin_map(x1, x2, gauss_x0))
    int_y = (x2 - x1) / 2 * gauss_w0 * fx

    # integral_{-3}^{3} exp(x) = sum_{i=1}^{5} w_i exp(t_i) = 20.036
    print(np.sum(int_y))

if False:
    x_max = 1
    x_min = -1
    r = (x_max - x_min) / 2    
    print("r = ", r)
    for n in range(1, 21, 4):
        x_gl, w_gl = leggauss(n)
        int_val = 0
        
        # 0 to pi
        for x_i, w_i in zip(x_gl, w_gl):
            theta = np.pi / 2 * (x_i + 1)
            Z = (x_min + x_max) / 2 + r * np.exp(1j * theta)
            f_xi = 1j * np.pi / 2 * r * np.exp(1j * theta) / Z
            # print("f_xi = ", f_xi)
            int_val += w_i * f_xi

        # pi to 2*pi
        for x_i, w_i in zip(x_gl, w_gl):
            theta = np.pi / 2 * (x_i + 3)
            Z = (x_min + x_max) / 2 + r * np.exp(1j * theta)            
            f_xi = 1j * np.pi / 2 * r * np.exp(1j * theta) / Z
            int_val += w_i * f_xi            

        print("n = {}, int_val = {:.4f}, int_val - 2pi = {:.4f}".format(n, int_val, np.abs(int_val) - 2 * np.pi))

if False:
    fz = lambda z: 1 / (z**2 + 1) / (z**2 + 1)
    ref_val = np.pi / 2
    x_max = .5
    x_min = -.5
    r = (x_max - x_min) / 2  
    Z0 = 1j * 1
    print("r = ", r)
    for n in range(1, 30, 4):
        x_gl, w_gl = leggauss(n)
        
        # 0 to pi
        int_val_1 = 0
        for x_i, w_i in zip(x_gl, w_gl):
            theta = np.pi / 2 * (x_i + 1)
            Z = Z0 + (x_min + x_max) / 2 + r * np.exp(1j * theta)
            f_xi = 1j * np.pi / 2 * r * np.exp(1j * theta) * fz(Z)
            # print("f_xi = ", f_xi)
            int_val_1 += w_i * f_xi

        # pi to 2*pi
        int_val_2 = 0
        for x_i, w_i in zip(x_gl, w_gl):
            theta = np.pi / 2 * (x_i + 3)
            Z = Z0 + (x_min + x_max) / 2 + r * np.exp(1j * theta)            
            f_xi = 1j * np.pi / 2 * r * np.exp(1j * theta) * fz(Z)
            int_val_2 += w_i * f_xi            

        int_val = int_val_1 + int_val_2
        print("n = {}, int_val = {:.4f}, int_val - ref = {:.4f}".format(n, int_val, np.abs(int_val) - ref_val))        

if True:
    fz = lambda z: 1 / (z**2 + 1) / (z**2 + 1)
    ref_val = np.pi / 2
    x_max = 2.1
    x_min = -2.1
    r = (x_max - x_min) / 2  
    print("r = ", r)
    for n in range(1, 30, 5):
        x_gl, w_gl = leggauss(n)
        
        # 0 to pi
        int_val_1 = 0
        for x_i, w_i in zip(x_gl, w_gl):
            theta = np.pi / 2 * (x_i + 1)
            Z = (x_min + x_max) / 2 + r * np.exp(1j * theta)
            f_xi = 1j * np.pi / 2 * r * np.exp(1j * theta) * fz(Z)
            # print("f_xi = ", f_xi)
            int_val_1 += w_i * f_xi

        # -a to a
        int_val_2 = 0
        for x_i, w_i in zip(x_gl, w_gl):

            Z = (x_max + x_min) / 2 + (x_max - x_min) / 2 * x_i
            f_xi = (x_max - x_min) / 2 * fz(Z)
            int_val_2 += w_i * f_xi            

        int_val = int_val_1 + int_val_2
        print("n = {}, int_val = {:.4f}, int_val - ref = {:.4f}".format(n, int_val, np.abs(int_val) - ref_val))            


if True:
    ref_val = np.pi / 2
    x_min = -2.1
    x_max =  2.1
    r = (x_max - x_min) / 2

    for n in range(1, 30, 5):
        # 0 to pi
        int_lim_1 = 0
        int_lim_2 = np.pi        
        z = lambda theta: (x_max + x_min) / 2 + r * np.exp(1j * theta)
        fz = lambda theta: 1 / (z(theta)**2 + 1) / (z(theta)**2 + 1)
        dz = lambda theta: 1j * r * np.exp(1j * theta)        
        int_val_1 = contour_filter(n, int_lim_1, int_lim_2, dz, fz)
        # -a to a
        int_lim_1 = x_min
        int_lim_2 = x_max       
        z = lambda x: x
        fz = lambda x: 1 / (z(x)**2 + 1) / (z(x)**2 + 1)
        dz = lambda x: 1
        int_val_2 = contour_filter(n, int_lim_1, int_lim_2, dz, fz)

        int_val = int_val_1 + int_val_2
        print("n = {}, int_val = {:.4f}, int_val - ref = {:.4f}".format(n, int_val, np.abs(int_val - ref_val)))            





