import numpy as np
import matplotlib.pyplot as plt

import sys
import os
import copy

from utils import *
from TLFSD import *
import douglas_peucker as dp

import matplotlib as mpl
mpl.rc('font',family='Times New Roman')


def pushing_example():
    
    ## Setup
    s_demos = []
    f_demos = []
    sfnames = ['../h5 files/fsil_demos/pushing_demo1.h5', '../h5 files/fsil_demos/pushing_demo2.h5']
    ffnames = ['../h5 files/fsil_demos/pushing_demo3.h5', '../h5 files/fsil_demos/pushing_demo4.h5']
    all_fnames = sfnames + ffnames
    
    color = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'r']
    n_pts_resample = 50
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scale_x = 2
    scale_y = 1
    scale_z = 1
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([scale_x, scale_y, scale_z, 1]))
    
    ## Load Demos
    for i in range(len(sfnames)):
        [x, y, z] = read_3D_h5(sfnames[i])
        data = np.hstack((np.reshape(x, (len(x), 1)), np.reshape(y, (len(y), 1)), np.reshape(z, (len(z), 1))))
        traj = dp.DouglasPeuckerPoints(data, n_pts_resample)
        s_demos.append(np.transpose(traj))
        s_demo, = ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'g', lw=3, alpha=0.4)
        
    for i in range(len(ffnames)):
        [x, y, z] = read_3D_h5(ffnames[i])
        data = np.hstack((np.reshape(x, (len(x), 1)), np.reshape(y, (len(y), 1)), np.reshape(z, (len(z), 1))))
        traj = dp.DouglasPeuckerPoints(data, n_pts_resample)
        f_demos.append(np.transpose(traj))
        f_demo, = ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'r', lw=3, alpha=0.4)
        
    ## Perform TLFSD
    K = 10000.0
    obj = TLFSD(copy.copy(s_demos), copy.copy(f_demos))
    obj.encode_GMMs(5)
    traj_fsil = obj.get_successful_reproduction(K)
    
    ## Plot
    fsil, = ax.plot(traj_fsil[0, :], traj_fsil[1, :], traj_fsil[2, :], 'k-', lw=5, ms=6)
    
    plot_irregular_cube([0.49, -0.52, 0], 
                        [0.49, -0.52, 0.1], 
                        [0.27, -0.41, 0.1], 
                        [0.27, -0.41, 0], 
                        [0.49+0.11, -0.52+0.03, 0], 
                        [0.49+0.11, -0.52+0.03, 0.1], 
                        [0.27+0.11, -0.41+0.03, 0.1], 
                        [0.27+0.11, -0.41+0.03, 0], ax)
                        
    ## for pretty plots
    
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.view_init(azim=-18, elev=19)
    plt.show()
    
if __name__ == '__main__':
    pushing_example()