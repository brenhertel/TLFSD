import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, './Guassian-Mixture-Models')
from GMM_GMR import GMM_GMR
import douglas_peucker as dp
from scipy.optimize import minimize
import os
import copy
from TLFSD import *
sys.path.insert(1, './dmp_pastor_2009/')
import perform_dmp as dmp
import screen_capture_rev2 as scr2
from scipy.interpolate import UnivariateSpline
import matplotlib as mpl
mpl.rc('font',family='Times New Roman')

from utils import *

def pushing_main():
    s_demos = []
    f_demos = []
    all_demos = []
    fnames = '../h5 files/pushing_demo.h5'
    num_demos = 6
    n_pts_resample = 100
    w_base = np.ones((n_pts_resample))
    #w_base = np.linspace(1, n_pts_resample, n_pts_resample)
    w = np.array(([]))
    
    for n in range(3):
        print(n)
        gmm_demos = []
        [[sm_t, sm_x, sm_y], [unsm_t, unsm_x, unsm_y], [norm_t, norm_x, norm_y]] = scr2.read_demo_h5(fnames, n)
        data = np.hstack((np.reshape(norm_x, (len(norm_x), 1)), np.reshape(-norm_y, (len(norm_y), 1))))
        data = data - data[-1, :] + [0, 0.1]
        data = data * 5
        traj = dp.DouglasPeuckerPoints(data, n_pts_resample)
        t = np.linspace(0, 1, n_pts_resample).reshape((1, n_pts_resample))
        all_demos.append(np.transpose(traj))
        f_demos.append(np.transpose(traj))
        for ind in range(len(all_demos)):
            gmm_demos.append(np.vstack((t, all_demos[ind])))
        w = np.hstack((w, w_base * (n / 100)))
        
        
    for n in range(3, num_demos):
        plt.figure()
        plt.plot(0.5, 0, 'w.')
        for j in range(n):
            if (j >= 3):
                s_demo, = plt.plot(all_demos[j][0, :], all_demos[j][1, :], 'g', lw=3, alpha=0.4)
            else:
                f_demo, = plt.plot(all_demos[j][0, :], all_demos[j][1, :], 'r', lw=3, alpha=0.4)
            
        gmm_demos = []
        [[sm_t, sm_x, sm_y], [unsm_t, unsm_x, unsm_y], [norm_t, norm_x, norm_y]] = scr2.read_demo_h5(fnames, n)
        data = np.hstack((np.reshape(norm_x, (len(norm_x), 1)), np.reshape(-norm_y, (len(norm_y), 1))))
        data = data - data[-1, :]
        data = data * 5
        traj = dp.DouglasPeuckerPoints(data, n_pts_resample)
        t = np.linspace(0, 1, n_pts_resample).reshape((1, n_pts_resample))
        
        all_demos.append(np.transpose(traj))
        s_demos.append(np.transpose(traj))
        s_demo, = plt.plot(traj[:, 0], traj[:, 1], 'g', lw=3, alpha=0.4)
        
        for ind in range(len(all_demos)):
            gmm_demos.append(np.vstack((t, all_demos[ind])))
            
        w = np.hstack((w[n_pts_resample:], w_base * (n**4 / 10)))
        gmm_demos = gmm_demos[-4:]
        wEM = GMM_GMR(6)
        
        #wEM.weighted_fit(np.hstack(gmm_demos), w)
        wEM.fit(np.hstack(gmm_demos))
        wEM.predict(t)
        wEM_mu = wEM.getPredictedData()
        
        K = 10000.0
        
        obj = TLFSD(copy.copy(s_demos), copy.copy(f_demos))
        obj.encode_GMMs(6)
        traj_fsil = obj.get_successful_reproduction(K)
        
        wem, = plt.plot(wEM_mu[1, :], wEM_mu[2, :], 'b-', lw=5)
        fsil, = plt.plot(traj_fsil[0, :], traj_fsil[1, :], 'k-', lw=5, ms=6)
        
        plt.xticks([])
        plt.yticks([])
        plt.show()
    
	
	
if __name__ == '__main__':
    pushing_main()