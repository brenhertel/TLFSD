import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

import sys
import os
import copy

from utils import *
from TLFSD import *
import douglas_peucker as dp
import screen_capture_rev2 as scr2

import matplotlib as mpl
mpl.rc('font',family='Times New Roman')

def iterative_example_main_record():
    s_demos = []
    f_demos = []
    
    fnames = ['../h5 files/my_sine.h5']
    num_demos = 1
    n_pts_resample = 75
    
    for i in range(len(fnames)):
        for n in range(num_demos):
            print(n)
            [[sm_t, sm_x, sm_y], [unsm_t, unsm_x, unsm_y]] = scr2.read_demo_h5_old(fnames[i], n)
            data = np.hstack((np.reshape(sm_x, (len(sm_x), 1)), np.reshape(sm_y, (len(sm_y), 1))))
            data = data - data[-1, :]
            traj = dp.DouglasPeuckerPoints(data, n_pts_resample * 2)
            x = traj[:, 0]
            y = -traj[:, 1]
            t = np.linspace(0, 1, n_pts_resample * 2)
            tt = np.linspace(0, 1, n_pts_resample)
            spx = UnivariateSpline(t, x)
            spy = UnivariateSpline(t, y)
            spx.set_smoothing_factor(999)
            spy.set_smoothing_factor(999)
            xx = spx(tt)
            yy = spy(tt)
            traj = np.hstack((np.reshape(xx, (len(xx), 1)), np.reshape(yy, (len(yy), 1))))
            f_demos.append(np.transpose(traj))
    
    inds = [0, n_pts_resample - 1]
    consts = f_demos[0][:, inds]
    
    K = 30000.0
    
    from matplotlib.patches import Rectangle
    rect = Rectangle((-200, -50), 80, 80, facecolor="black", alpha=0.3)
    rect_bounds = [-200, -50, -200 + 80, -50 + 80]
    
    d = 1
    
    iter = 1
    succ = 'n'
    while succ == 'n':
        print(iter)
        obj = TLFSD(copy.copy(s_demos), copy.copy(f_demos))
        obj.encode_GMMs(8)
        traj = obj.get_successful_reproduction(K, inds, consts)
        plt.figure()
        ax = plt.gca()
        pc = copy.copy(rect)
        ax.add_patch(pc)
        ax.text(-320, -80, 'Iteration: ' + str(iter), fontsize=32)
        for i in range(len(f_demos)):
            f_demo, = plt.plot(f_demos[i][0, :], f_demos[i][1, :], 'r', lw=3, alpha=0.4)
        if (obj.num_f > 0):
            gmr_mf, = plt.plot(obj.mu_f[d, :], obj.mu_f[d + 1, :], 'r', lw=5, alpha=0.8)
        trj, = plt.plot(traj[0, :], traj[1, :], 'k', lw=5)
        init, = plt.plot(consts[0, 0], consts[1, 0], 'ko', ms=10, mew=3)
        endp, = plt.plot(consts[0, 1], consts[1, 1], 'kx', ms=10, mew=3)
        plt.xticks([])
        plt.yticks([])
        plt.show()
        f_demos.append(traj)
        iter += 1
        #succ = input('Was this reproduction successful? (y/n): ')
        succ = 'y'
        for i in range(n_pts_resample):
            if (traj[0, i] > rect_bounds[0] - 3) and (traj[0, i] < rect_bounds[2] + 3) and (traj[1, i] > rect_bounds[1] - 3) and (traj[1, i] < rect_bounds[3] + 3):
                succ = 'n'
    
    plt.figure()
    ax = plt.gca()
    pc = copy.copy(rect)
    ax.add_patch(pc)
    for i in range(len(f_demos)):
        f_demo, = plt.plot(f_demos[i][0, :], f_demos[i][1, :], 'r', lw=3, alpha=0.4)
    if (obj.num_f > 0):
        gmr_mf, = plt.plot(obj.mu_f[d, :], obj.mu_f[d + 1, :], 'r', lw=5, alpha=0.8)
    trj, = plt.plot(traj[0, :], traj[1, :], 'k', lw=5)
    init, = plt.plot(consts[0, 0], consts[1, 0], 'ko', ms=10, mew=3)
    endp, = plt.plot(consts[0, 1], consts[1, 1], 'kx', ms=10, mew=3)
    plt.xticks([])
    plt.yticks([])
    
    plt.legend((f_demo, gmr_mf, trj, init, endp), ('Failed Set', 'Failed GMR Mean', 'FSIL', 'Initial Constraint', 'Endpoint Constraint'), fontsize='xx-large')# bbox_to_anchor=(1, 0.5))
    plt.show()
    
    
if __name__ == '__main__':
    iterative_example_main_record()