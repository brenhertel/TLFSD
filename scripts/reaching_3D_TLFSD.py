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
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d


def reaching_example():
    
    s_demos = []
    f_demos = []
    sfnames = ['../h5 files/fsil_demos/reaching_demo1.h5', '../h5 files/fsil_demos/reaching_demo2.h5']
    ffnames = ['../h5 files/fsil_demos/reaching_demo3.h5', '../h5 files/fsil_demos/reaching_demo4.h5']
    all_fnames = sfnames + ffnames
    
    color = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'r']
    n_pts_resample = 50
    
    ### ROUND 1: ALL SUCCESSFUL
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scale_x = 2
    scale_y = 1
    scale_z = 1
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([scale_x, scale_y, scale_z, 1]))
    #ax.set_ylim3d(0.1, -0.7)
    
    final_pt = None
    
    for i in range(len(all_fnames)):
        [x, y, z] = read_3D_h5(all_fnames[i])
        data = np.hstack((np.reshape(x, (len(x), 1)), np.reshape(y, (len(y), 1)), np.reshape(z, (len(z), 1))))
        traj = dp.DouglasPeuckerPoints(data, n_pts_resample)
        s_demos.append(np.transpose(traj))
        s_demo, = ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'g', lw=3, alpha=0.4)
        final_pt = final_pt + s_demos[i][:, -1] if final_pt is not None else s_demos[i][:, -1]
    
    final_pt = final_pt / len(all_fnames)
    
    K = 10000.0
    obj = TLFSD(copy.copy(s_demos), copy.copy(f_demos))
    obj.encode_GMMs(5)
    
    inds = [0, n_pts_resample - 1]
    
    for i in range(len(all_fnames)):
        consts = s_demos[i][:, inds]
        consts[:, 1] = final_pt
        traj_fsil = obj.get_successful_reproduction(K, inds, consts)
        fsil_traj, = ax.plot(traj_fsil[0, :], traj_fsil[1, :], traj_fsil[2, :], 'k', lw=5)
        ax.plot(consts[0, 0], consts[1, 0], consts[2, 0], 'k.', ms=12, mew=3)
        ax.plot(consts[0, 1], consts[1, 1], consts[2, 1], 'kx', ms=12, mew=3)
        
    #Xc,Yc,Zc = data_for_cylinder_along_z(0.2,0.2,0.05,0.1)
    #ax.plot_surface(Xc, Yc, Zc, 'r', alpha=0.5)
    
    plot_3D_cylinder(ax, radius=0.03, height=0.1, elevation=0, x_center = -0.5, y_center = -0.45, resolution=100, color='r')
    
    # Get rid of colored axes planes
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
    
    ax.view_init(elev=30, azim=-150)
    plt.show()
    
    ### ROUND 2: SOME SUCCESSFUL SOME FAILED
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scale_x = 2
    scale_y = 1
    scale_z = 1
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([scale_x, scale_y, scale_z, 1]))
    #ax.set_ylim3d(0.1, -0.7)
    
    #final_pt = np.zeros((3, 1))
    final_pt = None
    
    for i in range(len(sfnames)):
        [x, y, z] = read_3D_h5(sfnames[i])
        data = np.hstack((np.reshape(x, (len(x), 1)), np.reshape(y, (len(y), 1)), np.reshape(z, (len(z), 1))))
        traj = dp.DouglasPeuckerPoints(data, n_pts_resample)
        s_demos.append(np.transpose(traj))
        s_demo, = ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'g', lw=3, alpha=0.4)
        final_pt = final_pt + s_demos[i][:, -1] if final_pt is not None else s_demos[i][:, -1]
        
    for i in range(len(ffnames)):
        [x, y, z] = read_3D_h5(ffnames[i])
        data = np.hstack((np.reshape(x, (len(x), 1)), np.reshape(y, (len(y), 1)), np.reshape(z, (len(z), 1))))
        traj = dp.DouglasPeuckerPoints(data, n_pts_resample)
        f_demos.append(np.transpose(traj))
        f_demo, = ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'r', lw=3, alpha=0.4)
        final_pt = final_pt + f_demos[i][:, -1] if final_pt is not None else s_demos[i][:, -1]
    
    all_demos = s_demos + f_demos
    
    final_pt = final_pt / len(all_fnames)
    
    K = 10000.0
    obj = TLFSD(copy.copy(s_demos), copy.copy(f_demos))
    obj.encode_GMMs(5)
    
    inds = [0, n_pts_resample - 1]
    
    for i in range(len(all_fnames)):
        consts = all_demos[i][:, inds]
        consts[:, 1] = final_pt
        traj_fsil = obj.get_successful_reproduction(K, inds, consts)
        fsil_traj, = ax.plot(traj_fsil[0, :], traj_fsil[1, :], traj_fsil[2, :], 'k', lw=5)
        ax.plot(consts[0, 0], consts[1, 0], consts[2, 0], 'k.', ms=12, mew=3)
        ax.plot(consts[0, 1], consts[1, 1], consts[2, 1], 'kx', ms=12, mew=3)
        
    #Xc,Yc,Zc = data_for_cylinder_along_z(0.2,0.2,0.05,0.1)
    #ax.plot_surface(Xc, Yc, Zc, 'r', alpha=0.5)
    
    plot_3D_cylinder(ax, radius=0.03, height=0.1, elevation=0, x_center = -0.5, y_center = -0.45, resolution=100, color='r')
    
    xmin = 0.0
    xmax = 0.2
    ymin = -0.4
    ymax = -0.5
    zmin = 0.0
    zmax = 0.1
    
    plot_cube(xmin, xmax, ymin, ymax, zmin, zmax, ax)
    
    # Get rid of colored axes planes
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
    ax.view_init(elev=30, azim=-150)
    plt.show()
    
    
if __name__ == '__main__':
    reaching_example()