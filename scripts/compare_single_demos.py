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

def compare_1D_main():
    s_demos = []
    f_demos = []
    fnames = ['../h5 files/g_bended.h5']
    num_demos = 2
    n_pts_resample = 100
    
    import matplotlib.gridspec as gridspec
    
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 3)
    ax_s = fig.add_subplot(gs[0, 0])
    plt.xticks([])
    plt.yticks([])
    ax_f = fig.add_subplot(gs[1, 0])
    plt.xticks([])
    plt.yticks([])
    ax_b = fig.add_subplot(gs[:, 1:])
    
    
    for i in range(len(fnames)):
        for n in range(num_demos):
            print(n)
            [[sm_t, sm_x, sm_y], [unsm_t, unsm_x, unsm_y]] = scr2.read_demo_h5_old(fnames[i], n)
            data = np.hstack((np.reshape(sm_x, (len(sm_x), 1)), np.reshape(sm_y, (len(sm_y), 1))))
            data = data - data[-1, :]
            data = data / 10
            traj = dp.DouglasPeuckerPoints(data, n_pts_resample * 2)
            x = traj[:, 0]
            y = traj[:, 1]
            t = np.linspace(0, 1, n_pts_resample * 2)
            tt = np.linspace(0, 1, n_pts_resample)
            spx = UnivariateSpline(t, x)
            spy = UnivariateSpline(t, y)
            spx.set_smoothing_factor(999)
            spy.set_smoothing_factor(999)
            xx = spx(tt)
            yy = spy(tt)
            traj = np.hstack((np.reshape(xx, (len(xx), 1)), np.reshape(yy, (len(yy), 1))))
            traj = traj - traj[-1, :]
            if (n == 0):
                s_demos.append(np.transpose(traj))
                s_demo, = ax_s.plot(traj[:, 0], traj[:, 1], 'g', lw=3, alpha=0.8)
                s_demo, = ax_f.plot(traj[:, 0], traj[:, 1], 'g', lw=3, alpha=0.2)
                s_demo, = ax_b.plot(traj[:, 0], traj[:, 1], 'g', lw=3, alpha=0.4)
                s_traj = traj
            else:
                f_demos.append(np.transpose(traj))
                f_demo, = ax_s.plot(traj[:, 0], traj[:, 1], 'r', lw=3, alpha=0.2)
                f_demo, = ax_f.plot(traj[:, 0], traj[:, 1], 'r', lw=3, alpha=0.8)
                f_demo, = ax_b.plot(traj[:, 0], traj[:, 1], 'r', lw=3, alpha=0.4)
                f_traj = traj
    
    
    inds = [0, 99]
    consts_tlfsd = f_demos[0][:, inds]
    consts_other = np.transpose(consts_tlfsd)
    
    obj = TLFSD(copy.copy(s_demos), copy.copy(f_demos))
    obj.encode_GMMs(6)
    K = 3000.0
    traj_tlfsd = obj.get_successful_reproduction(K, inds, consts_tlfsd) #K = 20.0 #comp_tlfsd5.txt
    #obj.plot_results(mode='show')
    traj_tlfsd_comp = np.transpose(traj_tlfsd)
    
    import lte
    lte_traj = lte.LTE_ND_any_constraints(s_traj, consts_other, inds)
    
    
    sys.path.insert(1, './dmp_pastor_2009/')
    import perform_dmp as dmp
    dmp_x = dmp.perform_new_dmp_adapted(s_traj[:, 0], initial=consts_other[0, 0], end=consts_other[-1, 0])
    dmp_y = dmp.perform_new_dmp_adapted(s_traj[:, 1], initial=consts_other[0, 1], end=consts_other[-1, 1])
    dmp_traj = np.transpose(np.vstack((dmp_x, dmp_y)))
    
    sse_tlfsd_succ = sum_of_squared_error(traj_tlfsd_comp, s_traj)
    sse_tlfsd_fail = sum_of_squared_error(traj_tlfsd_comp, f_traj)
    sea_tlfsd_succ = swept_error_area(traj_tlfsd_comp, s_traj)
    sea_tlfsd_fail = swept_error_area(traj_tlfsd_comp, f_traj)
    crv_tlfsd_succ = curvature_comparison(traj_tlfsd_comp, s_traj)
    crv_tlfsd_fail = curvature_comparison(traj_tlfsd_comp, f_traj)
    
    print('sse_tlfsd_succ: %f'%(sse_tlfsd_succ))
    print('sse_tlfsd_fail: %f'%(sse_tlfsd_fail))
    print('sea_tlfsd_succ: %f'%(sea_tlfsd_succ))
    print('sea_tlfsd_fail: %f'%(sea_tlfsd_fail))
    print('crv_tlfsd_succ: %f'%(crv_tlfsd_succ))
    print('crv_tlfsd_fail: %f'%(crv_tlfsd_fail))
    
    sse_dmp_succ = sum_of_squared_error(dmp_traj, s_traj)
    sse_dmp_fail = sum_of_squared_error(dmp_traj, f_traj)
    sea_dmp_succ = swept_error_area(dmp_traj, s_traj)
    sea_dmp_fail = swept_error_area(dmp_traj, f_traj)
    crv_dmp_succ = curvature_comparison(dmp_traj, s_traj)
    crv_dmp_fail = curvature_comparison(dmp_traj, f_traj)
    
    print('sse_dmp_succ: %f'%(sse_dmp_succ))
    print('sse_dmp_fail: %f'%(sse_dmp_fail))
    print('sea_dmp_succ: %f'%(sea_dmp_succ))
    print('sea_dmp_fail: %f'%(sea_dmp_fail))
    print('crv_dmp_succ: %f'%(crv_dmp_succ))
    print('crv_dmp_fail: %f'%(crv_dmp_fail))
    
    sse_lte_succ = sum_of_squared_error(lte_traj, s_traj)
    sse_lte_fail = sum_of_squared_error(lte_traj, f_traj)
    sea_lte_succ = swept_error_area(lte_traj, s_traj)
    sea_lte_fail = swept_error_area(lte_traj, f_traj)
    crv_lte_succ = curvature_comparison(lte_traj, s_traj)
    crv_lte_fail = curvature_comparison(lte_traj, f_traj)
    
    print('sse_lte_succ: %f'%(sse_lte_succ))
    print('sse_lte_fail: %f'%(sse_lte_fail))
    print('sea_lte_succ: %f'%(sea_lte_succ))
    print('sea_lte_fail: %f'%(sea_lte_fail))
    print('crv_lte_succ: %f'%(crv_lte_succ))
    print('crv_lte_fail: %f'%(crv_lte_fail))
	
    tlfsdt, = ax_b.plot(traj_tlfsd[0, :], traj_tlfsd[1, :], 'k-', lw=5)
    dmpt, = ax_b.plot(dmp_x, dmp_y, 'b-', lw=5)
    lte, = ax_b.plot(lte_traj[:, 0], lte_traj[:, 1], 'm-', lw=5)
    
    init, = ax_b.plot(consts_tlfsd[0, 0], consts_tlfsd[1, 0], 'ko', ms=15, mew=5)
    endp, = ax_b.plot(consts_tlfsd[0, 1], consts_tlfsd[1, 1], 'kx', ms=15, mew=5)
    
    plt.legend((tlfsdt, dmpt, lte, init, endp), ('TLFSD', 'DMP', 'LTE', 'Initial Constraint', 'Endpoint Constraint'), fontsize='xx-large')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.xticks([])
    plt.yticks([])
    plt.show()
	
	
if __name__ == '__main__':
    compare_1D_main()