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

def fs_main():
    s_demos = []
    f_demos = []
    fnames = '../h5 files/curves1.h5'
    num_demos = 4
    n_pts_resample = 50
    plt.figure()
    
    for n in range(num_demos):
        print(n)
        [[sm_t, sm_x, sm_y], [unsm_t, unsm_x, unsm_y], [norm_t, norm_x, norm_y]] = scr2.read_demo_h5(fnames, n)
        data = np.hstack((np.reshape(norm_x, (len(norm_x), 1)), np.reshape(-norm_y, (len(norm_y), 1))))
        if (n == 0):
            data = data[350:, :]
        data = data - data[-1, :]
        data = data * 100
        traj = dp.DouglasPeuckerPoints(data, n_pts_resample)
        if (n < 2):
            s_demos.append(np.transpose(traj))
            s_demo, = plt.plot(traj[:, 0], traj[:, 1], 'g', lw=3, alpha=0.4)
        else:
            f_demos.append(np.transpose(traj))
            f_demo, = plt.plot(traj[:, 0], traj[:, 1], 'r', lw=3, alpha=0.4)
    
    inds = [0, 25, n_pts_resample - 1]
    consts = s_demos[1][:, inds]
    consts[:, 0] = np.array(([(f_demos[0][0, 0] + f_demos[1][0, 0]) / 2, (f_demos[0][1, 0] + f_demos[1][1, 0]) / 2])) 
    
    K = 10000.0
    
    obj_s = TLFSD(copy.copy(s_demos), [])
    obj_s.encode_GMMs(4)
    traj_s = obj_s.get_successful_reproduction(K, inds, consts) #K = 1200.0 
    #obj_s.plot_results(mode='show')
    
    K = 1000.0 
    
    obj_f = TLFSD([], copy.copy(f_demos))
    obj_f.encode_GMMs(4)
    traj_f = obj_f.get_successful_reproduction(K, inds, consts) #K = 1000.0 
    #obj_f.plot_results(mode='show')
    
    obj_b = TLFSD(copy.copy(s_demos), copy.copy(f_demos))
    obj_b.encode_GMMs(4)
    traj_b = obj_b.get_successful_reproduction(K, inds, consts) #K = 1000.0 
    #obj_b.plot_results(mode='show')
    
    d = 1
    if (obj_b.num_s > 0):
        gmr_ms, = plt.plot(obj_b.mu_s[d, :], obj_b.mu_s[d + 1, :], 'g', lw=5, alpha=0.8)
    if (obj_b.num_f > 0):
        gmr_mf, = plt.plot(obj_b.mu_f[d, :], obj_b.mu_f[d + 1, :], 'r', lw=5, alpha=0.8)
        
    fail, = plt.plot(traj_f[0, :], traj_f[1, :], 'k.', lw=5, ms=6)
    both, = plt.plot(traj_b[0, :], traj_b[1, :], 'k-', lw=5, ms=6)
    succ, = plt.plot(traj_s[0, :], traj_s[1, :], 'k--', lw=5, ms=6)
    
    init, = plt.plot(consts[0, 0], consts[1, 0], 'ko', ms=15, mew=5)
    viap, = plt.plot(consts[0, 1], consts[1, 1], 'k*', ms=20, mew=1, markerfacecolor='c', markeredgecolor='k')
    endp, = plt.plot(consts[0, 2], consts[1, 2], 'kx', ms=15, mew=5)
    plt.xticks([])
    plt.yticks([])
    
    #plt.legend((s_demo, f_demo, gmr_ms, gmr_mf, succ, fail, both, init, viap, endp), ('Successful Set', 'Failed Set', 'Successful GMR Mean', 'Failed GMR Mean', 'Successful Only', 'Failed Only', 'Failed and Successful', 'Initial Constraint', 'Via-point Constraint', 'Endpoint Constraint'), fontsize='x-large', handlelength=3.0)# bbox_to_anchor=(1, 0.5))
    plt.show()
    
	
if __name__ == '__main__':
    fs_main()