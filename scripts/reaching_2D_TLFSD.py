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

def fig1_new_main():
    s_demos = []
    f_demos = []
    fnames = '../h5 files/box2.h5'
    num_demos = 4
    n_pts_resample = 50
    plt.figure()
    for n in range(num_demos):
        print(n)
        [[sm_t, sm_x, sm_y], [unsm_t, unsm_x, unsm_y], [norm_t, norm_x, norm_y]] = scr2.read_demo_h5(fnames, n)
        data = np.hstack((np.reshape(norm_x, (len(norm_x), 1)), np.reshape(norm_y, (len(norm_y), 1))))
        data = data - data[-1, :]
        data = data * 100
        traj = dp.DouglasPeuckerPoints(data, n_pts_resample)
        if (n < 2):
            s_demos.append(np.transpose(traj))
            s_demo, = plt.plot(traj[:, 0], traj[:, 1], 'g', lw=3.5, alpha=0.4)
        else:
            f_demos.append(np.transpose(traj))
            f_demo, = plt.plot(traj[:, 0], traj[:, 1], 'r', lw=3.5, alpha=0.4)
    
    inds = [0, n_pts_resample - 1]
    consts = s_demos[0][:, inds]
    consts[0, 0] = consts[0, 0] + 0.1
    
    from matplotlib.patches import Rectangle
    rect = Rectangle((-45, -7), 14, 14, facecolor="black", alpha=0.3)
    
    K = 100.0
    
    obj_s = TLFSD(copy.copy(s_demos))
    obj_s.encode_GMMs(3)
    traj_s = obj_s.get_successful_reproduction(K, inds, consts) #K = 100.0 
    #obj_s.plot_results(mode='show')
    
    K = 10.0
    
    obj_b = TLFSD(copy.copy(s_demos), copy.copy(f_demos))
    obj_b.encode_GMMs(3)
    traj_b = obj_b.get_successful_reproduction(K, inds, consts) #K = 1000.0 
    #obj_b.plot_results(mode='show')
    
    d = 1
    if (obj_b.num_s > 0):
        gmr_ms, = plt.plot(obj_b.mu_s[d, :], obj_b.mu_s[d + 1, :], 'g', lw=6, alpha=0.8)
    if (obj_b.num_f > 0):
        gmr_mf, = plt.plot(obj_b.mu_f[d, :], obj_b.mu_f[d + 1, :], 'r', lw=6, alpha=0.8)
        
    succ, = plt.plot(traj_s[0, :], traj_s[1, :], 'k--', lw=6, ms=6)
    both, = plt.plot(traj_b[0, :], traj_b[1, :], 'k-', lw=6)
    
    plt.plot(consts[0, 0], consts[1, 0], 'ko', ms=20, mew=6)
    plt.plot(consts[0, 1], consts[1, 1], 'kx', ms=20, mew=6)
    
    ax = plt.gca()
    pc = copy.copy(rect)
    ax.add_patch(pc)
    plt.xticks([])
    plt.yticks([])
    
    #ax.legend((s_demo, f_demo, succ, both), ('Successful Set', 'Failed Set', 'Successful Only', 'Failed and Successful'), fontsize='x-large', handlelength=2.5, loc='lower left')# bbox_to_anchor=(1, 0.5))
    plt.show()
	
if __name__ == '__main__':
    fig1_new_main()