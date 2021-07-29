import numpy as np
import matplotlib.pyplot as plt

import sys
import os
import copy

from utils import *
from TLFSD import *
import douglas_peucker as dp
import screen_capture_rev2 as scr2

import matplotlib as mpl
mpl.rc('font',family='Times New Roman')

def mc_main():
    s_demos = []
    f_demos = []
    fnames = '../h5 files/multi_coordinate_demos2.h5'
    num_demos = 6
    n_pts_resample = 100
    plt.figure()
    for n in range(num_demos):
        print(n)
        [[sm_t, sm_x, sm_y], [unsm_t, unsm_x, unsm_y], [norm_t, norm_x, norm_y]] = scr2.read_demo_h5(fnames, n)
        data = np.hstack((np.reshape(norm_x, (len(norm_x), 1)), np.reshape(-norm_y, (len(norm_y), 1))))
        data = data - data[-1, :]
        data = data * 100
        traj = dp.DouglasPeuckerPoints(data, n_pts_resample)
        if (n >= 3):
            s_demos.append(np.transpose(traj))
            s_demo, = plt.plot(traj[:, 0], traj[:, 1], 'g', lw=3, alpha=0.4)
        else:
            f_demos.append(np.transpose(traj))
            f_demo, = plt.plot(traj[:, 0], traj[:, 1], 'r', lw=3, alpha=0.4)
    
    inds = [0, 65, n_pts_resample - 1]
    consts = f_demos[1][:, inds]
    
    K = 100000.0
    
    obj_cart = TLFSD(copy.copy(s_demos), copy.copy(f_demos))
    obj_cart.encode_GMMs(6)
    traj_cart = obj_cart.get_successful_reproduction(K, inds, consts) #K = 100000.0 #cart_demo.txt
    
    K = 900.0
    
    obj_mult = TLFSD(copy.copy(s_demos), copy.copy(f_demos))
    obj_mult.encode_GMMs(6, True)
    obj_mult.set_params(0.001, 0.009, 0.99)
    traj_mult = obj_mult.get_successful_reproduction(K, inds, consts) #K = 0.0 #mult_demo2.txt
    
    cart, = plt.plot(traj_cart[0, :], traj_cart[1, :], 'k--', lw=5, ms=6)
    mult, = plt.plot(traj_mult[0, :], traj_mult[1, :], 'k-', lw=5, ms=6)
    
    init, = plt.plot(consts[0, 0], consts[1, 0], 'ko', ms=15, mew=5)
    viap, = plt.plot(consts[0, 1], consts[1, 1], 'k*', ms=20, mew=1, markerfacecolor='c', markeredgecolor='k')
    endp, = plt.plot(consts[0, 2], consts[1, 2], 'kx', ms=15, mew=5)
    plt.xticks([])
    plt.yticks([])
    
    plt.legend((s_demo, f_demo, cart, mult, init, viap, endp), ('Successful Set', 'Failed Set', 'Single Coordinate', 'Multi-Coordinate', 'Initial Constraint', 'Via-point Constraint', 'Endpoint Constraint'), fontsize='x-large', handlelength=3.0)# bbox_to_anchor=(1, 0.5))
    plt.show()
	
	
if __name__ == '__main__':
    mc_main()