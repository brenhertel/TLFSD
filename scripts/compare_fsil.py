import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'C:/Users/BH/Documents/GitHub/pearl_test_env/Guassian-Mixture-Models')
from GMM_GMR import GMM_GMR
import douglas_peucker as dp
from scipy.optimize import minimize
import os
import copy
from lffd_gmm import *
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, './dmp_pastor_2009/')
import perform_dmp as dmp
import screen_capture_rev2 as scr2
from scipy.interpolate import UnivariateSpline
import matplotlib as mpl
mpl.rc('font',family='Times New Roman')
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d

def get_lasa_trajN(shape_name, n=1):
    #ask user for the file which the playback is for
    #filename = raw_input('Enter the filename of the .h5 demo: ')
    #open the file
    filename = '../h5 files/lasa_dataset.h5'
    hf = h5py.File(filename, 'r')
    #navigate to necessary data and store in numpy arrays
    shape = hf.get(shape_name)
    demo = shape.get('demo' + str(n))
    pos_info = demo.get('pos')
    pos_data = np.array(pos_info)
    y_data = np.delete(pos_data, 0, 1)
    x_data = np.delete(pos_data, 1, 1)
    #close out file
    hf.close()
    return [x_data, y_data]
    
def main1():
    s_demos = []
    f_demos = []
    shape_names = ['Trapezoid', 'Angle']
    num_demos = 3
    plt.figure()
    for i in range(len(shape_names)):
        for n in range(num_demos):
            [x, y] = get_lasa_trajN(shape_names[i], n + 1)
            data = np.hstack((np.reshape(x, (len(x), 1)), np.reshape(y, (len(y), 1))))
            
            traj = dp.DouglasPeuckerPoints(data, 50)
            if (i == 0):
                s_demos.append(np.transpose(traj))
                plt.plot(traj[:, 0], traj[:, 1], 'g', lw=3)
            else:
                f_demos.append(np.transpose(traj))
                plt.plot(traj[:, 0], traj[:, 1], 'r', lw=3)
    
    obj_all = LFFD_GMM(copy.copy(s_demos), copy.copy(f_demos))
    obj_all.encode_GMMs(3, True)
    
    inds = [0, 49]
    consts = np.transpose(np.array(traj[inds, :]))
    consts[:, 0] = consts[:, 0] - 5
    
    K = 50.0
    all_traj = obj_all.get_successful_reproduction(K, inds, consts)
    #obj_all.plot_results(mode='show')
    
    plt.plot(all_traj[0, :], all_traj[1, :], 'k-', lw=5)
    
    ### DMP
    x_s = np.zeros((1, 50))
    y_s = np.zeros((1, 50))
    for i in range(len(s_demos)):
        x_s = x_s + s_demos[i][0, :]
        y_s = y_s + s_demos[i][1, :]
    x_s = x_s[0] / len(s_demos)
    y_s = y_s[0] / len(s_demos)
    print(x_s)
    print(y_s)
    dmp_x = dmp.perform_new_dmp_adapted(np.transpose(x_s), initial=consts[0, 0], end=consts[0, 1])
    dmp_y = dmp.perform_new_dmp_adapted(np.transpose(y_s), initial=consts[1, 0], end=consts[1, 1])
    plt.plot(dmp_x, dmp_y, '-', color='orange', lw=5)
    
    ### GMM_GMR with weighted EM
    ss_demos = []
    ws = []
    t = np.linspace(0, 1, 50).reshape((1, 50))
    for i in range(len(s_demos)):
        ss_demos.append(np.vstack((t, s_demos[i])))
        ws.append(np.ones((50,)) * 10.0)
    for i in range(len(f_demos)):
        ss_demos.append(np.vstack((t, f_demos[i])))
        ws.append(np.ones((50,)) * 0.01)
    w_gmm = GMM_GMR(4)
    print(ws)
    w_gmm.weighted_fit(np.hstack(ss_demos), np.hstack(ws))
    w_gmm.predict(t)
    mu_w = w_gmm.getPredictedData()
    cov_w = w_gmm.getPredictedSigma()
    plt.plot(mu_w[1, :], mu_w[2, :], 'b-', lw=5)
    
    
    for i in range(len(inds)):
        plt.plot(consts[0, i], consts[1, i], 'k.', ms=15)
    plt.show()

def w_main():
    s_demos = []
    f_demos = []
    shape_names = ['WShape', 'Spoon']
    num_demos = 3
    plt.figure()
    for i in range(len(shape_names)):
        for n in range(num_demos):
            [x, y] = get_lasa_trajN(shape_names[i], n + 4)
            data = np.hstack((np.reshape(x, (len(x), 1)), np.reshape(y, (len(y), 1))))
            
            traj = dp.DouglasPeuckerPoints(data, 100)
            if (i == 0):
                s_demos.append(np.transpose(traj))
                s_demo, = plt.plot(traj[:, 0], traj[:, 1], 'g', lw=3, alpha=0.6)
            else:
                f_demos.append(np.transpose(traj))
                f_demo, = plt.plot(traj[:, 0], traj[:, 1], 'r', lw=3, alpha=0.6)
    #plt.show()
    
    
    inds = [0, 99]
    consts = np.transpose(np.array(traj[inds, :]))
    consts[:, 0] = consts[:, 0] + 5
    
    #obj = LFFD_GMM(copy.copy(s_demos), copy.copy(f_demos))
    #obj.encode_GMMs(6)
    #K = 10.0
    #obj.set_params(100000.0)
    #traj = obj.get_successful_reproduction(K, inds, consts)
    #obj.plot_results(mode='show')
    
    #alpha_0.txt
    #alpha_10.txt
    #alpha_100000.txt
    
    traj0 = np.loadtxt('alpha_0.txt')
    traj10 = np.loadtxt('alpha_10.txt')
    traj100000 = np.loadtxt('alpha_100000.txt')
    
    #X = traj
    #save_s = input('Would you like to save? (y/n): ')
    #if (save_s == 'y'):
    #    fname = input('Please enter a filename: ')
    #    np.savetxt(fname, X)
    
    succ, = plt.plot(traj0[0, :], traj0[1, :], 'k.', lw=5, ms=6)
    fail, = plt.plot(traj10[0, :], traj10[1, :], 'k--', lw=5)
    both, = plt.plot(traj100000[0, :], traj100000[1, :], 'k-', lw=5)
    
    
    plt.plot(consts[0, 0], consts[1, 0], 'ko', ms=10, mew=3)
    plt.plot(consts[0, 1], consts[1, 1], 'kx', ms=10, mew=3)
    plt.xticks([])
    plt.yticks([])
    
    plt.legend((s_demo, f_demo, succ, fail, both), ('Successful Set', 'Failed Set', r'Zero $\alpha$', r'Low $\alpha$', r'High $\alpha$'), fontsize=12)# bbox_to_anchor=(1, 0.5))
    plt.show()
    
def k_main():
    s_demos = []
    f_demos = []
    #shape_names = ['WShape', 'Spoon']
    fname = '../h5 files/my_spoon.h5'
    num_demos = 3
    n_pts_resample = 50
    plt.figure()
    for n in range(num_demos):
        print(n)
        [[sm_t, sm_x, sm_y], [unsm_t, unsm_x, unsm_y]] = scr2.read_demo_h5_old(fname, n)
        data = np.hstack((np.reshape(sm_x, (len(sm_x), 1)), np.reshape(sm_y, (len(sm_y), 1))))
        data = data - data[-1, :]
        traj = dp.DouglasPeuckerPoints(data, n_pts_resample * 2)
        x = traj[:, 0]
        y = -traj[:, 1]
        t = np.linspace(0, 1, n_pts_resample * 2)
        print(np.shape(x))
        print(np.shape(t))
        tt = np.linspace(0, 1, n_pts_resample)
        spx = UnivariateSpline(t, x)
        spy = UnivariateSpline(t, y)
        spx.set_smoothing_factor(999)
        spy.set_smoothing_factor(999)
        xx = spx(tt)
        yy = spy(tt)
        #plt.plot(x, y)
        #plt.plot(xx, yy)
        #plt.show()
        traj = np.hstack((np.reshape(xx, (len(xx), 1)), np.reshape(yy, (len(yy), 1))))
        s_demos.append(np.transpose(traj))
        s_demo, = plt.plot(traj[:, 0], traj[:, 1], 'g', lw=3, alpha=0.6)
        
    obj = LFFD_GMM(copy.copy(s_demos), copy.copy(f_demos))
    obj.encode_GMMs(5)
    
    inds = [0, n_pts_resample - 1]
    consts = np.transpose(np.array(traj[inds, :]))
    consts[1, 0] = consts[1, 0] - 50
    K = 0.001
    traj_under = obj.get_successful_reproduction(K, inds, consts)
    #obj.plot_results(mode='show')
    K = 1.0
    traj_right = obj.get_successful_reproduction(K, inds, consts)
    #obj.plot_results(mode='show')
    K = 1000.0
    traj_over = obj.get_successful_reproduction(K, inds, consts)
    #obj.plot_results(mode='show')
    
    succ, = plt.plot(traj_under[0, :], traj_under[1, :], 'k--', lw=5)
    fail, = plt.plot(traj_right[0, :], traj_right[1, :], 'k-', lw=5)
    both, = plt.plot(traj_over[0, :], traj_over[1, :], 'k.', lw=5, ms=6)
    
    
    plt.plot(consts[0, 0], consts[1, 0], 'ko', ms=10, mew=3)
    plt.plot(consts[0, 1], consts[1, 1], 'kx', ms=10, mew=3)
    plt.xticks([])
    plt.yticks([])
    
    plt.legend((s_demo, succ, fail, both), ('Successful Set', r'$\lambda = 0.001$', r'$\lambda = 1.0$', r'$\lambda = 1000.0$'), fontsize=12)# bbox_to_anchor=(1, 0.5))
    plt.show()
    
    
def fig1_main():
    s_demos = []
    f_demos = []
    n_pts = 50
    x = np.linspace(-np.pi, np.pi, n_pts)
    yf = -9 * np.abs(np.sin(x))
    #ys = -1 * (x**2 - np.pi**2)
    ys = -yf
    #plt.figure()
    
    s_demo, = plt.plot(x, ys, 'g', lw=1, alpha=0.6)
    f_demo, = plt.plot(x, yf, 'r', lw=1, alpha=0.6)
    s_d = np.vstack((x, ys))
    f_d = np.vstack((x, yf))
    s_demos.append(s_d)
    f_demos.append(f_d)
    
    inds = [0, n_pts - 1]
    consts = s_d[:, inds]
    print(consts)
    consts[1, 0] = consts[1, 0] - 4
    K = 1.0
    
    #obj_s = LFFD_GMM(copy.copy(s_demos))
    #obj_s.encode_GMMs(14)
    #traj_s = obj_s.get_successful_reproduction(K, inds, consts) #K = 100.0 #fig1_suc14.txt
    #obj_s.plot_results(mode='show')
    
    
    #obj_f = LFFD_GMM([], copy.copy(f_demos))
    #obj_f.encode_GMMs(8)
    #traj_f = obj_f.get_successful_reproduction(K, inds, consts) #K = 1.0 #fig1_fail7.txt
    #obj_f.plot_results(mode='show')
    
    
    #obj_b = LFFD_GMM(copy.copy(s_demos), copy.copy(f_demos))
    #obj_b.encode_GMMs(8)
    ##obj_b.set_params(500.0)
    #traj_b = obj_b.get_successful_reproduction(K, inds, consts) #K = 1000.0 #fig1_both8.txt
    #obj_b.plot_results(mode='show')
    
    traj_s = np.loadtxt('fig1_suc14.txt')
    succ, = plt.plot(traj_s[0, :], traj_s[1, :], 'k.', lw=5, ms=6)
    traj_f = np.loadtxt('fig1_fail7.txt')
    fail, = plt.plot(traj_f[0, :], traj_f[1, :], 'k--', lw=5)
    traj_b = np.loadtxt('fig1_both8.txt')
    both, = plt.plot(traj_b[0, :], traj_b[1, :], 'k-', lw=5)
    
    
    plt.plot(consts[0, 0], consts[1, 0], 'ko', ms=10, mew=3)
    plt.plot(consts[0, 1], consts[1, 1], 'kx', ms=10, mew=3)
    plt.xticks([])
    plt.yticks([])
    
    plt.legend((s_demo, f_demo, succ, fail, both), ('Successful Set', 'Failed Set', 'Success Only', 'Failed Only', 'Failed and Successful'), fontsize=12)# bbox_to_anchor=(1, 0.5))
    plt.show()
    
    #X = traj_s
    #save_s = input('Would you like to save? (y/n): ')
    #if (save_s == 'y'):
    #    fname = input('Please enter a filename: ')
    #    np.savetxt(fname, X)
    
def testing_form_main():
    s_demos = []
    f_demos = []
    n_pts = 50
    #x = np.linspace(-0.5, 0.5, n_pts)
    #yf = x**2 - 0.25 
    #x = x * 10
    #yf = yf * 10
    #ys = -yf 
    ##plt.figure()
    #
    #s_demo, = plt.plot(x, ys, 'g', lw=1, alpha=0.6)
    #f_demo, = plt.plot(x, yf, 'r', lw=1, alpha=0.6)
    #s_d = np.vstack((x, ys))
    #f_d = np.vstack((x, yf))
    #s_demos.append(s_d)
    #f_demos.append(f_d)
    shape_names = ['WShape', 'Spoon']
    num_demos = 3
    plt.figure()
    for i in range(len(shape_names)):
        for n in range(num_demos):
            [x, y] = get_lasa_trajN(shape_names[i], n + 1)
            data = np.hstack((np.reshape(x, (len(x), 1)), np.reshape(y, (len(y), 1))))
            
            traj = dp.DouglasPeuckerPoints(data, n_pts)
            if (i == 0):
                s_demos.append(np.transpose(traj))
                s_demo, = plt.plot(traj[:, 0], traj[:, 1], 'g', lw=1, alpha=0.6)
            else:
                f_demos.append(np.transpose(traj))
                f_demo, = plt.plot(traj[:, 0], traj[:, 1], 'r', lw=1, alpha=0.6)
    
    inds = [0, n_pts - 1]
    consts = s_demos[0][:, inds]
    print(consts)
    consts[1, 0] = consts[1, 0] - 1.25
    K = 25.0
    
    
    #obj_b = LFFD_GMM(copy.copy(s_demos), copy.copy(f_demos))
    #obj_b.encode_GMMs(5)
    #obj_b.set_params(0.9)
    #traj_b = obj_b.get_successful_reproduction(K, inds, consts) #K = 8000.0 #both.txt
    #obj_b.plot_results(mode='show')
    
    #ws_reg_from.txt
    #ws_new_from_alpha5.txt
    #ws_new_from_alpha9.txt
    #ws_new_from_alpha1.txt
    
    trj1 = np.loadtxt('ws_reg_from.txt')
    trj2 = np.loadtxt('ws_new_from_alpha5.txt')
    trj3 = np.loadtxt('ws_new_from_alpha9.txt')
    trj4 = np.loadtxt('ws_new_from_alpha1.txt')
    
    #traj_b = np.loadtxt('both.txt')
    t1, = plt.plot(trj1[0, :], trj1[1, :], 'k-', lw=5)
    t2, = plt.plot(trj2[0, :], trj2[1, :], 'c-', lw=5)
    t3, = plt.plot(trj3[0, :], trj3[1, :], 'm-', lw=5)
    t4, = plt.plot(trj4[0, :], trj4[1, :], 'y-', lw=5)
    #
    #
    plt.plot(consts[0, 0], consts[1, 0], 'ko', ms=10, mew=3)
    plt.plot(consts[0, 1], consts[1, 1], 'kx', ms=10, mew=3)
    #plt.xticks([])
    #plt.yticks([])
    #
    plt.legend((s_demo, f_demo, t1, t2, t3, t4), ('Successful Demo', 'Failed Demo', 'Current form', 'New form alpha = 0.5', 'New form alpha = 0.9' , 'New form alpha = 0.1'), fontsize=15)# bbox_to_anchor=(1, 0.5))
    plt.show()
    
    #X = traj_b
    #save_s = input('Would you like to save? (y/n): ')
    #if (save_s == 'y'):
    #    fname = input('Please enter a filename: ')
    #    np.savetxt(fname, X)
    
def exp1_fs_main():
    s_demos = []
    f_demos = []
    
    fnames = ['../h5 files/my_sine.h5', '../h5 files/my_trapezoid1.h5']
    num_demos = 3
    n_pts_resample = 100
    plt.figure()
    for i in range(len(fnames)):
        for n in range(num_demos):
            print(n)
            [[sm_t, sm_x, sm_y], [unsm_t, unsm_x, unsm_y]] = scr2.read_demo_h5(fnames[i], n)
            data = np.hstack((np.reshape(sm_x, (len(sm_x), 1)), np.reshape(sm_y, (len(sm_y), 1))))
            data = data - data[-1, :]
            traj = dp.DouglasPeuckerPoints(data, n_pts_resample * 2)
            x = traj[:, 0]
            y = traj[:, 1]
            t = np.linspace(0, 1, n_pts_resample * 2)
            print(np.shape(x))
            print(np.shape(t))
            tt = np.linspace(0, 1, n_pts_resample)
            spx = UnivariateSpline(t, x)
            spy = UnivariateSpline(t, y)
            spx.set_smoothing_factor(999)
            spy.set_smoothing_factor(999)
            xx = spx(tt)
            yy = spy(tt)
            #plt.plot(x, y)
            #plt.plot(xx, yy)
            #plt.show()
            traj = np.hstack((np.reshape(xx, (len(xx), 1)), np.reshape(yy, (len(yy), 1))))
            if (i == 0):
                s_demos.append(np.transpose(traj))
                s_demo, = plt.plot(traj[:, 0], traj[:, 1], 'g', lw=3, alpha=0.6)
            else:
                f_demos.append(np.transpose(traj))
                f_demo, = plt.plot(traj[:, 0], traj[:, 1], 'r', lw=3, alpha=0.6)
    
    inds = [0, 70, n_pts_resample - 1]
    consts = s_demos[0][:, inds]
    print(consts)
    consts[0, 1] = consts[0, 1] - 25
    K = 0.5
    
    #obj_s = LFFD_GMM(copy.copy(s_demos))
    #obj_s.encode_GMMs(8)
    #traj = obj_s.get_successful_reproduction(K, inds, consts) #K = 0.5 #exp1_succ.txt
    #obj_s.plot_results(mode='show')
    
    
    #obj_f = LFFD_GMM([], copy.copy(f_demos))
    #obj_f.encode_GMMs(8)
    #traj = obj_f.get_successful_reproduction(K, inds, consts) #K = 0.5 #exp1_fail.txt
    #obj_f.plot_results(mode='show')
    
    
    #K = 10.0
    #obj_b = LFFD_GMM(copy.copy(s_demos), copy.copy(f_demos))
    #obj_b.encode_GMMs(8)
    #obj_b.set_params(1000000.0)
    #traj = obj_b.get_successful_reproduction(K, inds, consts) #K = 10.0 obj_b.set_params(0.05) #exp1_both1.txt
    #obj_b.plot_results(mode='show')
    
    
    traj_s = np.loadtxt('exp1_succ.txt')
    traj_f = np.loadtxt('exp1_fail.txt')
    traj_b = np.loadtxt('exp1_both1.txt')
    
    succ, = plt.plot(traj_s[0, :], traj_s[1, :], 'k.', lw=5, ms=6)
    fail, = plt.plot(traj_f[0, :], traj_f[1, :], 'k--', lw=5)
    both, = plt.plot(traj_b[0, :], traj_b[1, :], 'k-', lw=5)
    init, = plt.plot(consts[0, 0], consts[1, 0], 'ko', ms=10, mew=3)
    via, = plt.plot(consts[0, 1], consts[1, 1], 'k*', ms=10, mew=3)
    endp, = plt.plot(consts[0, 2], consts[1, 2], 'kx', ms=10, mew=3)
    plt.xticks([])
    plt.yticks([])
    
    plt.legend((s_demo, f_demo, succ, fail, both, init, via, endp), ('Successful Set', 'Failed Set', 'Success Only', 'Failed Only', 'Failed and Successful', 'Initial Constraint', 'Via-Point Constraint', 'Endpoint Constraint'), fontsize=15)# bbox_to_anchor=(1, 0.5))
    plt.show()
    
    #X = traj
    #save_s = input('Would you like to save? (y/n): ')
    #if (save_s == 'y'):
    #    fname = input('Please enter a filename: ')
    #    np.savetxt(fname, X)

def curvature_comparison(exp_data, num_data):
    (n_points, n_dims) = np.shape(exp_data)
    if not np.shape(exp_data) == np.shape(num_data):
        print('Array dims must match!')
    L = 2.*np.diag(np.ones((n_points,))) - np.diag(np.ones((n_points-1,)),1) - np.diag(np.ones((n_points-1,)),-1)
    L[0,1] = -2.
    L[-1,-2] = -2.
    err_abs = np.absolute(np.subtract(np.matmul(L, exp_data), np.matmul(L, num_data)))
    return np.sum(err_abs)

def herons_formula(a, b, c):
    s = (a + b + c) / 2.
    return (s * (s - a) * (s - b) * (s - c))**0.5

def swept_error_area(exp_data, num_data):
    #naive approach
    (n_points, n_dims) = np.shape(exp_data)
    if not np.shape(exp_data) == np.shape(num_data):
        print('Array dims must match!')
    sum = 0.
    for i in range(n_points - 1):
        p1 = exp_data[i]
        p2 = exp_data[i + 1]
        p3 = num_data[i + 1]
        p4 = num_data[i]
        p1_p2_dist = np.linalg.norm(p1 - p2)
        p1_p3_dist = np.linalg.norm(p1 - p3)
        p1_p4_dist = np.linalg.norm(p1 - p4)
        p2_p3_dist = np.linalg.norm(p2 - p3)
        p3_p4_dist = np.linalg.norm(p3 - p4)
        triangle1_area = herons_formula(p1_p4_dist, p1_p3_dist, p3_p4_dist)
        triangle2_area = herons_formula(p1_p2_dist, p1_p3_dist, p2_p3_dist)
        sum += triangle1_area + triangle2_area
    return sum / n_points

def sum_of_squared_error(exp_data, num_data):
    #naive approach
    (n_points, n_dims) = np.shape(exp_data)
    if not np.shape(exp_data) == np.shape(num_data):
        print('Array dims must match!')
    sum = 0.
    for i in range(n_points):
        sum += (np.linalg.norm(exp_data[i] - num_data[i]))**2
    return sum

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
                #s_demo, = plt.plot(traj[:, 0], traj[:, 1], 'g', lw=3, alpha=0.4)
                s_demo, = ax_s.plot(traj[:, 0], traj[:, 1], 'g', lw=3, alpha=0.8)
                s_demo, = ax_f.plot(traj[:, 0], traj[:, 1], 'g', lw=3, alpha=0.2)
                s_demo, = ax_b.plot(traj[:, 0], traj[:, 1], 'g', lw=3, alpha=0.4)
                s_traj = traj
            else:
                f_demos.append(np.transpose(traj))
                #f_demo, = plt.plot(traj[:, 0], traj[:, 1], 'r', lw=3, alpha=0.4)
                f_demo, = ax_s.plot(traj[:, 0], traj[:, 1], 'r', lw=3, alpha=0.2)
                f_demo, = ax_f.plot(traj[:, 0], traj[:, 1], 'r', lw=3, alpha=0.8)
                f_demo, = ax_b.plot(traj[:, 0], traj[:, 1], 'r', lw=3, alpha=0.4)
                f_traj = traj
    
    
    inds = [0, 99]
    consts_fsil = f_demos[0][:, inds]
    consts_other = np.transpose(consts_fsil)
    
    #obj = LFFD_GMM(copy.copy(s_demos), copy.copy(f_demos))
    #obj.encode_GMMs(6)
    #K = 3000.0
    #traj_fsil = obj.get_successful_reproduction(K, inds, consts_fsil) #K = 20.0 #comp_fsil5.txt
    #obj.plot_results(mode='show')
    #d = 1
    #if (obj.num_s > 0):
    #    gmr_ms, = plt.plot(obj.mu_s[d, :], obj.mu_s[d + 1, :], 'g', lw=5, alpha=0.8)
    #    #print(obj.cov_s[d, d, :])
    #    #plt.fill_between(obj.mu_s[d, :], obj.mu_s[d + 1, :] + obj.cov_s[d, d, :], obj.mu_s[d + 1, :] - obj.cov_s[d, d, :], color='b', alpha=0.3)
    #    #obj.s_gmm.plot(1, 2, "Regression", plt.gca(), dataColor = [0, 0.2, 0.7], clusterColor = [0, 0.2, 0.8], regressionColor = [0,0.2,0.8])
    #if (obj.num_f > 0):
    #    gmr_mf, = plt.plot(obj.mu_f[d, :], obj.mu_f[d + 1, :], 'r', lw=5, alpha=0.8)
    #    #plt.fill_between(obj.mu_f[d, :], obj.mu_f[d + 1, :] + obj.cov_f[d, d, :], obj.mu_f[d + 1, :] - obj.cov_f[d, d, :], color='m', alpha=0.3)
    #    #obj.f_gmm.plot(1, 2, "Regression", plt.gca(), dataColor = [0.8, 0.2, 0], clusterColor = [0.8, 0.2, 0], regressionColor = [0.8,0.2,0])
    
    traj_fsil = np.loadtxt('comp_fsil5.txt')
    
    #X = traj_fsil
    #save_s = input('Would you like to save? (y/n): ')
    #if (save_s == 'y'):
    #    fname = input('Please enter a filename: ')
    #    np.savetxt(fname, X)
    
    traj_fsil_comp = np.transpose(traj_fsil)
    
    import lte
    lte_traj = lte.LTE_ND_any_constraints(s_traj, consts_other, inds)
    
    dmp_x = dmp.perform_new_dmp_adapted(s_traj[:, 0], initial=consts_other[0, 0], end=consts_other[-1, 0])
    dmp_y = dmp.perform_new_dmp_adapted(s_traj[:, 1], initial=consts_other[0, 1], end=consts_other[-1, 1])
    dmp_traj = np.transpose(np.vstack((dmp_x, dmp_y)))
    
    #sse_fsil_succ = sum_of_squared_error(traj_fsil_comp, s_traj)
    #sse_fsil_fail = sum_of_squared_error(traj_fsil_comp, f_traj)
    #sea_fsil_succ = swept_error_area(traj_fsil_comp, s_traj)
    #sea_fsil_fail = swept_error_area(traj_fsil_comp, f_traj)
    #crv_fsil_succ = curvature_comparison(traj_fsil_comp, s_traj)
    #crv_fsil_fail = curvature_comparison(traj_fsil_comp, f_traj)
    #
    #print('sse_fsil_succ: %f'%(sse_fsil_succ))
    #print('sse_fsil_fail: %f'%(sse_fsil_fail))
    #print('sea_fsil_succ: %f'%(sea_fsil_succ))
    #print('sea_fsil_fail: %f'%(sea_fsil_fail))
    #print('crv_fsil_succ: %f'%(crv_fsil_succ))
    #print('crv_fsil_fail: %f'%(crv_fsil_fail))
    #
    #sse_dmp_succ = sum_of_squared_error(dmp_traj, s_traj)
    #sse_dmp_fail = sum_of_squared_error(dmp_traj, f_traj)
    #sea_dmp_succ = swept_error_area(dmp_traj, s_traj)
    #sea_dmp_fail = swept_error_area(dmp_traj, f_traj)
    #crv_dmp_succ = curvature_comparison(dmp_traj, s_traj)
    #crv_dmp_fail = curvature_comparison(dmp_traj, f_traj)
    #
    #print('sse_dmp_succ: %f'%(sse_dmp_succ))
    #print('sse_dmp_fail: %f'%(sse_dmp_fail))
    #print('sea_dmp_succ: %f'%(sea_dmp_succ))
    #print('sea_dmp_fail: %f'%(sea_dmp_fail))
    #print('crv_dmp_succ: %f'%(crv_dmp_succ))
    #print('crv_dmp_fail: %f'%(crv_dmp_fail))
    #
    #sse_lte_succ = sum_of_squared_error(lte_traj, s_traj)
    #sse_lte_fail = sum_of_squared_error(lte_traj, f_traj)
    #sea_lte_succ = swept_error_area(lte_traj, s_traj)
    #sea_lte_fail = swept_error_area(lte_traj, f_traj)
    #crv_lte_succ = curvature_comparison(lte_traj, s_traj)
    #crv_lte_fail = curvature_comparison(lte_traj, f_traj)
    #
    #print('sse_lte_succ: %f'%(sse_lte_succ))
    #print('sse_lte_fail: %f'%(sse_lte_fail))
    #print('sea_lte_succ: %f'%(sea_lte_succ))
    #print('sea_lte_fail: %f'%(sea_lte_fail))
    #print('crv_lte_succ: %f'%(crv_lte_succ))
    #print('crv_lte_fail: %f'%(crv_lte_fail))
    
    #fsil, = plt.plot(traj_fsil[0, :], traj_fsil[1, :], 'k-', lw=5)
    #dmpt, = plt.plot(dmp_x, dmp_y, 'b-', lw=5)
    #lte, = plt.plot(lte_traj[:, 0], lte_traj[:, 1], 'm-', lw=5)
    
    #init, = plt.plot(consts_fsil[0, 0], consts_fsil[1, 0], 'ko', ms=15, mew=5)
    #endp, = plt.plot(consts_fsil[0, 1], consts_fsil[1, 1], 'kx', ms=15, mew=5)
    
    fsil, = ax_b.plot(traj_fsil[0, :], traj_fsil[1, :], 'k-', lw=5)
    dmpt, = ax_b.plot(dmp_x, dmp_y, 'b-', lw=5)
    lte, = ax_b.plot(lte_traj[:, 0], lte_traj[:, 1], 'm-', lw=5)
    
    init, = ax_b.plot(consts_fsil[0, 0], consts_fsil[1, 0], 'ko', ms=15, mew=5)
    endp, = ax_b.plot(consts_fsil[0, 1], consts_fsil[1, 1], 'kx', ms=15, mew=5)
    
    #plt.legend((s_demo, f_demo, gmr_ms, gmr_mf, fsil, dmpt, lte, init, endp), ('Successful Set', 'Failed Set', 'Successful GMR Mean', 'Failed GMR Mean', 'TLFSD', 'DMP', 'LTE', 'Initial Constraint', 'Endpoint Constraint'), fontsize='x-large')# bbox_to_anchor=(1, 0.5))
    plt.legend((fsil, dmpt, lte, init, endp), ('TLFSD', 'DMP', 'LTE', 'Initial Constraint', 'Endpoint Constraint'), fontsize='xx-large')# bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.xticks([])
    plt.yticks([])
    #plt.savefig('../pictures/FSIL/final_pics/dmp_cmp_1.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_mD_main():
    s_demos = []
    f_demos = []
    fnames = ['../h5 files/small_r.h5', '../h5 files/x3.h5']
    num_demos = 5
    n_pts_resample = 100
    plt.figure()
    all_succ = True
    for i in range(len(fnames)):
        for n in range(num_demos):
            print(n)
            [[sm_t, sm_x, sm_y], [unsm_t, unsm_x, unsm_y]] = scr2.read_demo_h5(fnames[i], n)
            data = np.hstack((np.reshape(sm_x, (len(sm_x), 1)), np.reshape(sm_y, (len(sm_y), 1))))
            #data = data - data[-1, :]
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
            if (all_succ):
                s_demos.append(np.transpose(traj))
                s_demo, = plt.plot(traj[:, 0], traj[:, 1], 'g', lw=3, alpha=0.6)
                s_traj = traj
            else:
                if (n == 0):
                    s_demos.append(np.transpose(traj))
                    #s_demo, = plt.plot(traj[:, 0], traj[:, 1], 'g', lw=3, alpha=0.6)
                    s_traj = traj
                else:
                    f_demos.append(np.transpose(traj))
                    #f_demo, = plt.plot(traj[:, 0], traj[:, 1], 'r', lw=3, alpha=0.6)
                    f_traj = traj
    
    
    #inds = [99]
    #consts_fsil = s_demos[0][:, inds]
    #consts_other = np.transpose(consts_fsil)
    #
    #obj = LFFD_GMM(copy.copy(s_demos), copy.copy(f_demos))
    #obj.encode_GMMs(6)
    #K = 1.0
    #traj_fsil = obj.get_successful_reproduction(K, inds, consts_fsil) #K = 20.0 #md_fsil2.txt
    #obj.plot_results(mode='show')
    
    #traj_fsil = np.loadtxt('md_fsil2.txt')
    
    #X = traj_fsil
    #
    #save_s = input('Would you like to save? (y/n): ')
    #if (save_s == 'y'):
    #    fname = input('Please enter a filename: ')
    #    np.savetxt(fname, X)
    
    #traj_fsil_comp = np.transpose(traj_fsil)
    #
    #import lte
    #lte_traj = lte.LTE_ND_any_constraints(s_traj, consts_other, inds)
    #
    #dmp_x = dmp.perform_new_dmp_adapted(s_traj[:, 0], initial=consts_other[0, 0], end=consts_other[-1, 0])
    #dmp_y = dmp.perform_new_dmp_adapted(s_traj[:, 1], initial=consts_other[0, 1], end=consts_other[-1, 1])
    #dmp_traj = np.transpose(np.vstack((dmp_x, dmp_y)))
    #
    #sse_fsil_succ = sum_of_squared_error(traj_fsil_comp, s_traj)
    #sse_fsil_fail = sum_of_squared_error(traj_fsil_comp, f_traj)
    #sea_fsil_succ = swept_error_area(traj_fsil_comp, s_traj)
    #sea_fsil_fail = swept_error_area(traj_fsil_comp, f_traj)
    #crv_fsil_succ = curvature_comparison(traj_fsil_comp, s_traj)
    #crv_fsil_fail = curvature_comparison(traj_fsil_comp, f_traj)
    #
    #print('sse_fsil_succ: %f'%(sse_fsil_succ))
    #print('sse_fsil_fail: %f'%(sse_fsil_fail))
    #print('sea_fsil_succ: %f'%(sea_fsil_succ))
    #print('sea_fsil_fail: %f'%(sea_fsil_fail))
    #print('crv_fsil_succ: %f'%(crv_fsil_succ))
    #print('crv_fsil_fail: %f'%(crv_fsil_fail))
    #
    #sse_dmp_succ = sum_of_squared_error(dmp_traj, s_traj)
    #sse_dmp_fail = sum_of_squared_error(dmp_traj, f_traj)
    #sea_dmp_succ = swept_error_area(dmp_traj, s_traj)
    #sea_dmp_fail = swept_error_area(dmp_traj, f_traj)
    #crv_dmp_succ = curvature_comparison(dmp_traj, s_traj)
    #crv_dmp_fail = curvature_comparison(dmp_traj, f_traj)
    #
    #print('sse_dmp_succ: %f'%(sse_dmp_succ))
    #print('sse_dmp_fail: %f'%(sse_dmp_fail))
    #print('sea_dmp_succ: %f'%(sea_dmp_succ))
    #print('sea_dmp_fail: %f'%(sea_dmp_fail))
    #print('crv_dmp_succ: %f'%(crv_dmp_succ))
    #print('crv_dmp_fail: %f'%(crv_dmp_fail))
    #
    #sse_lte_succ = sum_of_squared_error(lte_traj, s_traj)
    #sse_lte_fail = sum_of_squared_error(lte_traj, f_traj)
    #sea_lte_succ = swept_error_area(lte_traj, s_traj)
    #sea_lte_fail = swept_error_area(lte_traj, f_traj)
    #crv_lte_succ = curvature_comparison(lte_traj, s_traj)
    #crv_lte_fail = curvature_comparison(lte_traj, f_traj)
    #
    #print('sse_lte_succ: %f'%(sse_lte_succ))
    #print('sse_lte_fail: %f'%(sse_lte_fail))
    #print('sea_lte_succ: %f'%(sea_lte_succ))
    #print('sea_lte_fail: %f'%(sea_lte_fail))
    #print('crv_lte_succ: %f'%(crv_lte_succ))
    #print('crv_lte_fail: %f'%(crv_lte_fail))
    #
    #fsil, = plt.plot(traj_fsil[0, :], traj_fsil[1, :], 'k-', lw=5)
    #lte, = plt.plot(lte_traj[:, 0], lte_traj[:, 1], 'm-', lw=5)
    #dmpt, = plt.plot(dmp_x, dmp_y, 'b-', lw=5)
    #
    #init, = plt.plot(consts_fsil[0, 0], consts_fsil[1, 0], 'ko', ms=10, mew=3)
    #endp, = plt.plot(consts_fsil[0, 1], consts_fsil[1, 1], 'kx', ms=10, mew=3)
    #
    #plt.legend((s_demo, f_demo, fsil, lte, dmpt, init, endp), ('Successful Set', 'Failed Set', 'FSIL', 'LTE', 'DMP', 'Initial Constraint', 'Endpoint Constraint'), fontsize=15)# bbox_to_anchor=(1, 0.5))
    #plt.xticks([])
    #plt.yticks([])
    #plt.show()
    print()
    
def iterative_example_main():
    s_demos = []
    f_demos = []
    
    fnames = ['../h5 files/my_sine.h5']
    num_demos = 1
    n_pts_resample = 75
    plt.figure()
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
            print(np.shape(x))
            print(np.shape(t))
            tt = np.linspace(0, 1, n_pts_resample)
            spx = UnivariateSpline(t, x)
            spy = UnivariateSpline(t, y)
            spx.set_smoothing_factor(999)
            spy.set_smoothing_factor(999)
            xx = spx(tt)
            yy = spy(tt)
            traj = np.hstack((np.reshape(xx, (len(xx), 1)), np.reshape(yy, (len(yy), 1))))
            f_demos.append(np.transpose(traj))
            #f_demo, = plt.plot(traj[:, 0], traj[:, 1], 'r', lw=3, alpha=0.6)
    
    inds = [0, n_pts_resample - 1]
    consts = f_demos[0][:, inds]
    print(consts)
    K = 20000.0
    
    
    from matplotlib.patches import Rectangle
    rect = Rectangle((-200, -50), 80, 80)
    rect_bounds = [-200, -50, -200 + 80, -50 + 80]
    
    d = 1
    
    iter = 1
    succ = 'n'
    while succ == 'n':
        print(iter)
        obj = LFFD_GMM(copy.copy(s_demos), copy.copy(f_demos))
        obj.encode_GMMs(8)
        traj = obj.get_successful_reproduction(K, inds, consts)
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
        #if (iter > 3):
        #    init, = plt.plot(consts[0, 0], consts[1, 0], 'ko', ms=10, mew=3)
        #    viap, = plt.plot(consts[0, 1], consts[1, 1], 'k*', ms=10, mew=3)
        #    endp, = plt.plot(consts[0, 2], consts[1, 2], 'kx', ms=10, mew=3)
        #else:
        #    init, = plt.plot(consts[0, 0], consts[1, 0], 'ko', ms=10, mew=3)
        #    endp, = plt.plot(consts[0, 1], consts[1, 1], 'kx', ms=10, mew=3)
        #
        #if (iter == 3):
        #    inds = [0, int(n_pts_resample / 1.65), n_pts_resample - 1]
        #    consts = f_demos[0][:, inds]
        #    print(consts)
        plt.xticks([])
        plt.yticks([])
        plt.savefig('../pictures/FSIL/iterative_demo/iter' + str(iter) + '.png')
        plt.close('all')
        iter += 1
        f_demos.append(traj)
        print(iter)
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
    #if (iter > 3):
    #    init, = plt.plot(consts[0, 0], consts[1, 0], 'ko', ms=10, mew=3)
    #    viap, = plt.plot(consts[0, 1], consts[1, 1], 'k*', ms=10, mew=3)
    #    endp, = plt.plot(consts[0, 2], consts[1, 2], 'kx', ms=10, mew=3)
    #else:
    #    init, = plt.plot(consts[0, 0], consts[1, 0], 'ko', ms=10, mew=3)
    #    endp, = plt.plot(consts[0, 1], consts[1, 1], 'kx', ms=10, mew=3)
    init, = plt.plot(consts[0, 0], consts[1, 0], 'ko', ms=10, mew=3)
    endp, = plt.plot(consts[0, 1], consts[1, 1], 'kx', ms=10, mew=3)
    plt.xticks([])
    plt.yticks([])
    
    plt.legend((f_demo, gmr_mf, trj, init, endp), ('Failed Set', 'Failed GMR Mean', 'FSIL', 'Initial Constraint', 'Endpoint Constraint'), fontsize='xx-large')# bbox_to_anchor=(1, 0.5))
    plt.show()
    
def iterative_example_main_record():
    s_demos = []
    f_demos = []
    
    fnames = ['../h5 files/my_sine.h5']
    num_demos = 1
    n_pts_resample = 75
    plt.figure()
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
            print(np.shape(x))
            print(np.shape(t))
            tt = np.linspace(0, 1, n_pts_resample)
            spx = UnivariateSpline(t, x)
            spy = UnivariateSpline(t, y)
            spx.set_smoothing_factor(999)
            spy.set_smoothing_factor(999)
            xx = spx(tt)
            yy = spy(tt)
            traj = np.hstack((np.reshape(xx, (len(xx), 1)), np.reshape(yy, (len(yy), 1))))
            np.savetxt('../pictures/FSIL/iterative_demo/given_demo' + str(n) + '.txt', np.transpose(traj))
            f_demos.append(np.transpose(traj))
            #f_demo, = plt.plot(traj[:, 0], traj[:, 1], 'r', lw=3, alpha=0.6)
    
    inds = [0, n_pts_resample - 1]
    consts = f_demos[0][:, inds]
    print(consts)
    K = 30000.0
    
    
    from matplotlib.patches import Rectangle
    rect = Rectangle((-200, -50), 80, 80, facecolor="black", alpha=0.3)
    rect_bounds = [-200, -50, -200 + 80, -50 + 80]
    
    d = 1
    
    iter = 1
    succ = 'n'
    while succ == 'n':
        print(iter)
        obj = LFFD_GMM(copy.copy(s_demos), copy.copy(f_demos))
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
            np.savetxt('../pictures/FSIL/iterative_demo/gmr_mean_iter' + str(iter) + '.txt', obj.mu_f[1:, :])
        trj, = plt.plot(traj[0, :], traj[1, :], 'k', lw=5)
        init, = plt.plot(consts[0, 0], consts[1, 0], 'ko', ms=10, mew=3)
        endp, = plt.plot(consts[0, 1], consts[1, 1], 'kx', ms=10, mew=3)
        #if (iter > 3):
        #    init, = plt.plot(consts[0, 0], consts[1, 0], 'ko', ms=10, mew=3)
        #    viap, = plt.plot(consts[0, 1], consts[1, 1], 'k*', ms=10, mew=3)
        #    endp, = plt.plot(consts[0, 2], consts[1, 2], 'kx', ms=10, mew=3)
        #else:
        #    init, = plt.plot(consts[0, 0], consts[1, 0], 'ko', ms=10, mew=3)
        #    endp, = plt.plot(consts[0, 1], consts[1, 1], 'kx', ms=10, mew=3)
        #
        #if (iter == 3):
        #    inds = [0, int(n_pts_resample / 1.65), n_pts_resample - 1]
        #    consts = f_demos[0][:, inds]
        #    print(consts)
        plt.xticks([])
        plt.yticks([])
        plt.savefig('../pictures/FSIL/iterative_demo/iter' + str(iter) + '.png')
        plt.close('all')
        f_demos.append(traj)
        iter += 1
        np.savetxt('../pictures/FSIL/iterative_demo/repro_iter' + str(iter) + '.txt', traj)
        print(iter)
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
    #if (iter > 3):
    #    init, = plt.plot(consts[0, 0], consts[1, 0], 'ko', ms=10, mew=3)
    #    viap, = plt.plot(consts[0, 1], consts[1, 1], 'k*', ms=10, mew=3)
    #    endp, = plt.plot(consts[0, 2], consts[1, 2], 'kx', ms=10, mew=3)
    #else:
    #    init, = plt.plot(consts[0, 0], consts[1, 0], 'ko', ms=10, mew=3)
    #    endp, = plt.plot(consts[0, 1], consts[1, 1], 'kx', ms=10, mew=3)
    init, = plt.plot(consts[0, 0], consts[1, 0], 'ko', ms=10, mew=3)
    endp, = plt.plot(consts[0, 1], consts[1, 1], 'kx', ms=10, mew=3)
    plt.xticks([])
    plt.yticks([])
    
    plt.legend((f_demo, gmr_mf, trj, init, endp), ('Failed Set', 'Failed GMR Mean', 'FSIL', 'Initial Constraint', 'Endpoint Constraint'), fontsize='xx-large')# bbox_to_anchor=(1, 0.5))
    plt.show()
    
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
    print(consts)
    
    from matplotlib.patches import Rectangle
    rect = Rectangle((-45, -7), 14, 14, facecolor="black", alpha=0.3)
    
    K = 3000.0
    
    #obj_s = LFFD_GMM(copy.copy(s_demos))
    #obj_s.encode_GMMs(3)
    #traj_s = obj_s.get_successful_reproduction(K, inds, consts) #K = 100.0 #fig1_suc14.txt
    #mean_s = obj_s.self.mu_s[1:, :]
    #obj_s.plot_results(mode='show')
    
    
    #obj_f = LFFD_GMM([], copy.copy(f_demos))
    #obj_f.encode_GMMs(8)
    #traj_f = obj_f.get_successful_reproduction(K, inds, consts) #K = 1.0 #fig1_fail7.txt
    #obj_f.plot_results(mode='show')
    
    
    obj_b = LFFD_GMM(copy.copy(s_demos), copy.copy(f_demos))
    obj_b.encode_GMMs(3)
    #traj_b = obj_b.get_successful_reproduction(K, inds, consts) #K = 1000.0 #fig1_both8.txt
    #obj_b.plot_results(mode='show')
    #obj_b = LFFD_GMM(copy.copy(f_demos), copy.copy(s_demos))
    #obj_b.encode_GMMs(5)
    #traj_b1 = obj_b.get_successful_reproduction(0, inds, consts) #K = 1000.0 #fig1_both8.txt
    
    
    d = 1
    if (obj_b.num_s > 0):
        gmr_ms, = plt.plot(obj_b.mu_s[d, :], obj_b.mu_s[d + 1, :], 'g', lw=6, alpha=0.8)
        #print(obj_b.cov_s[d, d, :])
        #plt.fill_between(obj_b.mu_s[d, :], obj_b.mu_s[d + 1, :] + obj_b.cov_s[d, d, :], obj_b.mu_s[d + 1, :] - obj_b.cov_s[d, d, :], color='b', alpha=0.3)
        #obj_b.s_gmm.plot(1, 2, "Regression", plt.gca(), dataColor = [0, 0.2, 0.7], clusterColor = [0, 0.2, 0.8], regressionColor = [0,0.2,0.8])
    if (obj_b.num_f > 0):
        gmr_mf, = plt.plot(obj_b.mu_f[d, :], obj_b.mu_f[d + 1, :], 'r', lw=6, alpha=0.8)
        #plt.fill_between(obj_b.mu_f[d, :], obj_b.mu_f[d + 1, :] + obj_b.cov_f[d, d, :], obj_b.mu_f[d + 1, :] - obj_b.cov_f[d, d, :], color='m', alpha=0.3)
        #obj_b.f_gmm.plot(1, 2, "Regression", plt.gca(), dataColor = [0.8, 0.2, 0], clusterColor = [0.8, 0.2, 0], regressionColor = [0.8,0.2,0])
        
    traj_s = np.loadtxt('box_succ2.txt')
    succ, = plt.plot(traj_s[0, :], traj_s[1, :], 'k--', lw=6, ms=6)
    #traj_f = np.loadtxt('fig1_fail7.txt')
    #fail, = plt.plot(traj_f[0, :], traj_f[1, :], 'k--', lw=5)
    traj_b = np.loadtxt('box1.txt')
    both, = plt.plot(traj_b[0, :], traj_b[1, :], 'k-', lw=6)
    #both, = plt.plot(traj_b1[0, :], traj_b1[1, :], 'c-', lw=5)
    #
    #
    plt.plot(consts[0, 0], consts[1, 0], 'ko', ms=20, mew=6)
    plt.plot(consts[0, 1], consts[1, 1], 'kx', ms=20, mew=6)
    ax = plt.gca()
    pc = copy.copy(rect)
    ax.add_patch(pc)
    plt.xticks([])
    plt.yticks([])
    
    #ax.legend((s_demo, f_demo, gmr_ms, gmr_mf, succ, both), ('Successful Set', 'Failed Set', 'Successful GMR Mean', 'Failed GMR Mean', 'Successful Only', 'Failed and Successful'), fontsize='x-large')# bbox_to_anchor=(1, 0.5))
    ax.legend((s_demo, f_demo, succ, both), ('Successful Set', 'Failed Set', 'Successful Only', 'Failed and Successful'), fontsize='x-large', handlelength=2.5, loc='lower left')# bbox_to_anchor=(1, 0.5))
    #plt.show()
    plt.savefig('../pictures/FSIL/final_pics/fig1_5.png', dpi=300)
    
    #X = traj_s
    #save_s = input('Would you like to save? (y/n): ')
    #if (save_s == 'y'):
    #    fname = input('Please enter a filename: ')
    #    np.savetxt(fname, X)
    
def exp1_main():
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
    print(consts)
    
    K = 1000.0
    
    #obj_b = LFFD_GMM(copy.copy(s_demos), [])
    #obj_b.encode_GMMs(4)
    #traj_b = obj_b.get_successful_reproduction(K, inds, consts) #K = 1200.0 #curves_svia2.txt
    #obj_b.plot_results(mode='show')
    
    
    #obj_b = LFFD_GMM([], copy.copy(f_demos))
    #obj_b.encode_GMMs(4)
    #traj_b = obj_b.get_successful_reproduction(K, inds, consts) #K = 1000.0 #curves_fvia2.txt
    #obj_b.plot_results(mode='show')
    
    obj_b = LFFD_GMM(copy.copy(s_demos), copy.copy(f_demos))
    obj_b.encode_GMMs(4)
    #traj_b = obj_b.get_successful_reproduction(K, inds, consts) #K = 1000.0 #curves_bvia4.txt
    #obj_b.plot_results(mode='show')
    d = 1
    if (obj_b.num_s > 0):
        gmr_ms, = plt.plot(obj_b.mu_s[d, :], obj_b.mu_s[d + 1, :], 'g', lw=5, alpha=0.8)
        #print(obj_b.cov_s[d, d, :])
        #plt.fill_between(obj_b.mu_s[d, :], obj_b.mu_s[d + 1, :] + obj_b.cov_s[d, d, :], obj_b.mu_s[d + 1, :] - obj_b.cov_s[d, d, :], color='b', alpha=0.3)
        #obj_b.s_gmm.plot(1, 2, "Regression", plt.gca(), dataColor = [0, 0.2, 0.7], clusterColor = [0, 0.2, 0.8], regressionColor = [0,0.2,0.8])
    if (obj_b.num_f > 0):
        gmr_mf, = plt.plot(obj_b.mu_f[d, :], obj_b.mu_f[d + 1, :], 'r', lw=5, alpha=0.8)
        #plt.fill_between(obj_b.mu_f[d, :], obj_b.mu_f[d + 1, :] + obj_b.cov_f[d, d, :], obj_b.mu_f[d + 1, :] - obj_b.cov_f[d, d, :], color='m', alpha=0.3)
        #obj_b.f_gmm.plot(1, 2, "Regression", plt.gca(), dataColor = [0.8, 0.2, 0], clusterColor = [0.8, 0.2, 0], regressionColor = [0.8,0.2,0])
    
    traj_s = np.loadtxt('curves_svia2.txt')
    traj_f = np.loadtxt('curves_fvia2.txt')
    traj_b = np.loadtxt('curves_bvia4.txt')
    succ, = plt.plot(traj_s[0, :], traj_s[1, :], 'k--', lw=5, ms=6)
    fail, = plt.plot(traj_f[0, :], traj_f[1, :], 'k.', lw=5, ms=6)
    both, = plt.plot(traj_b[0, :], traj_b[1, :], 'k-', lw=5, ms=6)
    
    
    init, = plt.plot(consts[0, 0], consts[1, 0], 'ko', ms=15, mew=5)
    viap, = plt.plot(consts[0, 1], consts[1, 1], 'k*', ms=20, mew=1, markerfacecolor='c', markeredgecolor='k')
    endp, = plt.plot(consts[0, 2], consts[1, 2], 'kx', ms=15, mew=5)
    plt.xticks([])
    plt.yticks([])
    
    plt.legend((s_demo, f_demo, gmr_ms, gmr_mf, succ, fail, both, init, viap, endp), ('Successful Set', 'Failed Set', 'Successful GMR Mean', 'Failed GMR Mean', 'Successful Only', 'Failed Only', 'Failed and Successful', 'Initial Constraint', 'Via-point Constraint', 'Endpoint Constraint'), fontsize='x-large', handlelength=3.0)# bbox_to_anchor=(1, 0.5))
    plt.show()
    
    #X = traj_b
    #save_s = input('Would you like to save? (y/n): ')
    #if (save_s == 'y'):
    #    fname = input('Please enter a filename: ')
    #    np.savetxt(fname, X)
    
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
    print(consts)
    
    #K = 100000.0
    #
    #obj_cart = LFFD_GMM(copy.copy(s_demos), copy.copy(f_demos))
    #obj_cart.encode_GMMs(6)
    #traj_cart = obj_cart.get_successful_reproduction(K, inds, consts) #K = 100000.0 #cart_demo.txt
    #obj_cart.plot_results(mode='show')
    #X = traj_cart
    #save_s = input('Would you like to save? (y/n): ')
    #if (save_s == 'y'):
    #    fname = input('Please enter a filename: ')
    #    np.savetxt(fname, X)
    
    K = 1000.0
    
    #obj_mult = LFFD_GMM(copy.copy(s_demos), copy.copy(f_demos))
    #obj_mult.encode_GMMs(6, True)
    #obj_mult.set_params(0.001, 0.19, 0.8)
    #traj_mult = obj_mult.get_successful_reproduction(K, inds, consts) #K = 0.0 #mult_demo2.txt
    #obj_mult.plot_results(mode='show')
    #X = traj_mult
    #save_s = input('Would you like to save? (y/n): ')
    #if (save_s == 'y'):
    #    fname = input('Please enter a filename: ')
    #    np.savetxt(fname, X)
    #d = 1
    #if (obj_cart.num_s > 0):
    #    gmr_ms, = plt.plot(obj_cart.mu_s[d, :], obj_cart.mu_s[d + 1, :], 'g', lw=5, alpha=0.8)
    #if (obj_cart.num_f > 0):
    #    gmr_mf, = plt.plot(obj_cart.mu_f[d, :], obj_cart.mu_f[d + 1, :], 'r', lw=5, alpha=0.8)
    
    traj_cart = np.loadtxt('cart_demo.txt')
    traj_mult = np.loadtxt('mult_demo2.txt')
    cart, = plt.plot(traj_cart[0, :], traj_cart[1, :], 'k--', lw=5, ms=6)
    mult, = plt.plot(traj_mult[0, :], traj_mult[1, :], 'k-', lw=5, ms=6)
    
    
    init, = plt.plot(consts[0, 0], consts[1, 0], 'ko', ms=15, mew=5)
    viap, = plt.plot(consts[0, 1], consts[1, 1], 'k*', ms=20, mew=1, markerfacecolor='c', markeredgecolor='k')
    endp, = plt.plot(consts[0, 2], consts[1, 2], 'kx', ms=15, mew=5)
    plt.xticks([])
    plt.yticks([])
    
    plt.legend((s_demo, f_demo, cart, mult, init, viap, endp), ('Successful Set', 'Failed Set', 'Single Coordinate', 'Multi-Coordinate', 'Initial Constraint', 'Via-point Constraint', 'Endpoint Constraint'), fontsize='x-large', handlelength=3.0)# bbox_to_anchor=(1, 0.5))
    #plt.savefig('../pictures/FSIL/final_pics/multicoordinate.png', dpi=300, bbox_inches='tight')
    plt.show()
    print()
    
def pushing_main():
    s_demos = []
    f_demos = []
    all_demos = []
    fnames = '../h5 files/pushing_demo.h5'
    num_demos = 6
    n_pts_resample = 100
    #plt.figure()
    #w_base = np.ones((100))
    w_base = np.linspace(1, 100, 100)
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
            
        print(n)
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
        #w = np.hstack((w[n_pts_resample:], w_base * (n**4 / 10)))
        print(w)
        gmm_demos = gmm_demos[-4:]
        wEM = GMM_GMR(6)
        #wEM.weighted_fit(np.hstack(gmm_demos), w)
        wEM.fit(np.hstack(gmm_demos))
        wEM.predict(t)
        wEM_mu = wEM.getPredictedData()
        #wEM_cov = wEM.getPredictedSigma()
        
        
        K = 10000.0
        
        obj = LFFD_GMM(copy.copy(s_demos), copy.copy(f_demos))
        obj.encode_GMMs(6)
        traj_fsil = obj.get_successful_reproduction(K)
        
        np.savetxt('wem_mu_iter' + str(n) + '.txt', wEM_mu)
        np.savetxt('push_fsil_iter' + str(n) + '.txt', traj_fsil)
        
        #d = 1
        #if (obj.num_s > 0):
        #    gmr_ms, = plt.plot(obj.mu_s[d, :], obj.mu_s[d + 1, :], 'g', lw=5, alpha=0.8)
        #if (obj.num_f > 0):
        #    gmr_mf, = plt.plot(obj.mu_f[d, :], obj.mu_f[d + 1, :], 'r', lw=5, alpha=0.8)
        
        wem, = plt.plot(wEM_mu[1, :], wEM_mu[2, :], 'b-', lw=5)
        fsil, = plt.plot(traj_fsil[0, :], traj_fsil[1, :], 'k-', lw=5, ms=6)
        
        plt.xticks([])
        plt.yticks([])
        #plt.show()
        #if (n == 5):
        #    plt.legend((s_demo, f_demo, gmr_ms, gmr_ms, fsil, wem), ('Successful Set', 'Failed Set', 'Successful GMR Mean', 'Failed GMR Mean', 'TLFSD', 'GMM/GMR with wEM'), fontsize='x-large')# bbox_to_anchor=(1, 0.5))
        #    #plt.show()
        plt.savefig('../pictures/FSIL/final_pics/gmmwem_push/traj' + str(n) + '.png', dpi=300)
        plt.close('all')
    
    
    #K = 1.0
    #
    #obj = LFFD_GMM(copy.copy(s_demos), copy.copy(f_demos))
    #obj.encode_GMMs(6)
    #traj_fsil = obj.get_successful_reproduction(K) #K = 100000.0 #cart_demo.txt
    #obj.plot_results(mode='show')
    #X = traj_fsil
    #save_s = input('Would you like to save? (y/n): ')
    #if (save_s == 'y'):
    #    fname = input('Please enter a filename: ')
    #    np.savetxt(fname, X)
    
    
    #traj_cart = np.loadtxt('cart_demo.txt')
    #traj_mult = np.loadtxt('mult_demo2.txt')
    #cart, = plt.plot(traj_cart[0, :], traj_cart[1, :], 'k--', lw=5, ms=6)
    #mult, = plt.plot(traj_mult[0, :], traj_mult[1, :], 'k-', lw=5, ms=6)
    #
    #plt.xticks([])
    #plt.yticks([])
    #
    #plt.legend((s_demo, f_demo, cart, mult, init, viap, endp), ('Successful Set', 'Failed Set', 'Single Coordinate', 'Multi-Coordinate', 'Initial Constraint', 'Via-point Constraint', 'Endpoint Constraint'), fontsize=12)# bbox_to_anchor=(1, 0.5))
    #plt.show()
    print()
    
def read_3D_h5(fname):
    #ask user for the file which the playback is for
    #filename = raw_input('Enter the filename of the .h5 demo: ')
    #open the file
    hf = h5py.File(fname, 'r')
    #navigate to necessary data and store in numpy arrays
    demo = hf.get('demo1')
    tf_info = demo.get('tf_info')
    pos_info = tf_info.get('pos_rot_data')
    pos_data = np.array(pos_info)
    
    x = pos_data[0, :]
    y = pos_data[1, :]
    z = pos_data[2, :]
    #close out file
    hf.close()
    return [x, y, z]
    
def plot_cube(min_x, max_x, min_y, max_y, min_z, max_z, ax):    
    #squares in x-y plane
    xv = [min_x, min_x, max_x, max_x]
    yv = [min_y, max_y, max_y, min_y]
    zv = [min_z, min_z, min_z, min_z]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_facecolor('saddlebrown')
    poly.set_alpha(0.3)
    ax.add_collection3d(poly)
    xv = [min_x, min_x, max_x, max_x]
    yv = [min_y, max_y, max_y, min_y]
    zv = [max_z, max_z, max_z, max_z]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_facecolor('saddlebrown')
    poly.set_alpha(0.3)
    ax.add_collection3d(poly)
    
    #squares in x-z plane
    xv = [min_x, min_x, max_x, max_x]
    yv = [min_y, min_y, min_y, min_y]
    zv = [min_z, max_z, max_z, min_z]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_facecolor('saddlebrown')
    poly.set_alpha(0.3)
    ax.add_collection3d(poly)
    xv = [min_x, min_x, max_x, max_x]
    yv = [max_y, max_y, max_y, max_y]
    zv = [min_z, max_z, max_z, min_z]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_facecolor('saddlebrown')
    poly.set_alpha(0.3)
    ax.add_collection3d(poly)
    
    #squares in y-z plane
    xv = [min_x, min_x, min_x, min_x]
    yv = [min_y, min_y, max_y, max_y]
    zv = [min_z, max_z, max_z, min_z]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_facecolor('saddlebrown')
    poly.set_alpha(0.3)
    ax.add_collection3d(poly)
    xv = [max_x, max_x, max_x, max_x]
    yv = [min_y, min_y, max_y, max_y]
    zv = [min_z, max_z, max_z, min_z]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_facecolor('saddlebrown')
    poly.set_alpha(0.3)
    ax.add_collection3d(poly)
    
def plot_irregular_cube(v1, v2, v3, v4, v5, v6, v7, v8, ax):    
    #front and back
    xv = [v1[0], v2[0], v3[0], v4[0]]
    yv = [v1[1], v2[1], v3[1], v4[1]]
    zv = [v1[2], v2[2], v3[2], v4[2]]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_facecolor('saddlebrown')
    poly.set_alpha(0.3)
    ax.add_collection3d(poly)
    xv = [v5[0], v6[0], v7[0], v8[0]]
    yv = [v5[1], v6[1], v7[1], v8[1]]
    zv = [v5[2], v6[2], v7[2], v8[2]]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_facecolor('saddlebrown')
    poly.set_alpha(0.3)
    ax.add_collection3d(poly)
    
    #sides
    xv = [v1[0], v2[0], v6[0], v5[0]]
    yv = [v1[1], v2[1], v6[1], v5[1]]
    zv = [v1[2], v2[2], v6[2], v5[2]]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_facecolor('saddlebrown')
    poly.set_alpha(0.3)
    ax.add_collection3d(poly)
    xv = [v3[0], v4[0], v8[0], v7[0]]
    yv = [v3[1], v4[1], v8[1], v7[1]]
    zv = [v3[2], v4[2], v8[2], v7[2]]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_facecolor('saddlebrown')
    poly.set_alpha(0.3)
    ax.add_collection3d(poly)
    
    #sides
    xv = [v1[0], v5[0], v8[0], v4[0]]
    yv = [v1[1], v5[1], v8[1], v4[1]]
    zv = [v1[2], v5[2], v8[2], v4[2]]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_facecolor('saddlebrown')
    poly.set_alpha(0.3)
    ax.add_collection3d(poly)
    xv = [v2[0], v6[0], v7[0], v3[0]]
    yv = [v2[1], v6[1], v7[1], v3[1]]
    zv = [v2[2], v6[2], v7[2], v3[2]]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_facecolor('saddlebrown')
    poly.set_alpha(0.3)
    ax.add_collection3d(poly)
    
#this function from https://stackoverflow.com/questions/26989131/add-cylinder-to-plot
def data_for_cylinder_along_z(center_x,center_y,radius,height_z):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid
    
def plot_3D_cylinder(ax, radius, height, elevation=0, x_center = 0, y_center = 0, resolution=100, color='r'):
    #fig=plt.figure()
    #ax = Axes3D(fig, azim=30, elev=30)

    x = np.linspace(x_center-radius, x_center+radius, resolution)
    z = np.linspace(elevation, elevation+height, resolution)
    X, Z = np.meshgrid(x, z)

    Y = np.sqrt(radius**2 - (X - x_center)**2) + y_center # Pythagorean theorem


    ax.plot_surface(X, Y, Z, linewidth=0, color=color)
    ax.plot_surface(X, (2*y_center-Y), Z, linewidth=0, color=color)

    floor = Circle((x_center, y_center), radius, color=color)
    ax.add_patch(floor)
    art3d.pathpatch_2d_to_3d(floor, z=elevation, zdir="z")

    ceiling = Circle((x_center, y_center), radius, color=color)
    ax.add_patch(ceiling)
    art3d.pathpatch_2d_to_3d(ceiling, z=elevation+height, zdir="z")
    
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
    
    #final_pt = np.zeros((3, 1))
    final_pt = None
    
    for i in range(len(all_fnames)):
        [x, y, z] = read_3D_h5(all_fnames[i])
        data = np.hstack((np.reshape(x, (len(x), 1)), np.reshape(y, (len(y), 1)), np.reshape(z, (len(z), 1))))
        traj = dp.DouglasPeuckerPoints(data, n_pts_resample)
        s_demos.append(np.transpose(traj))
        s_demo, = ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'g', lw=3, alpha=0.4)
        final_pt = final_pt + s_demos[i][:, -1] if final_pt is not None else s_demos[i][:, -1]
        print(final_pt)
    
    final_pt = final_pt / len(all_fnames)
    
    #K = 10000.0
    #obj = LFFD_GMM(copy.copy(s_demos), copy.copy(f_demos))
    #obj.encode_GMMs(5)
    
    inds = [0, n_pts_resample - 1]
    
    for i in range(len(all_fnames)):
        consts = s_demos[i][:, inds]
        consts[:, 1] = final_pt
        print(consts)
        #input()
        #traj_fsil = obj.get_successful_reproduction(K, inds, consts)
        #np.savetxt('reaching_all_s_repro' + str(i) + '.txt', traj_fsil)
        traj_fsil = np.loadtxt('reaching_all_s_repro' + str(i) + '.txt')
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
    
    ax.view_init(-145, 30)
    plt.show()
    
    #### ROUND 2: SOME SUCCESSFUL SOME FAILED
    #
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #scale_x = 2
    #scale_y = 1
    #scale_z = 1
    #ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([scale_x, scale_y, scale_z, 1]))
    ##ax.set_ylim3d(0.1, -0.7)
    #
    ##final_pt = np.zeros((3, 1))
    #final_pt = None
    #
    #for i in range(len(sfnames)):
    #    [x, y, z] = read_3D_h5(sfnames[i])
    #    data = np.hstack((np.reshape(x, (len(x), 1)), np.reshape(y, (len(y), 1)), np.reshape(z, (len(z), 1))))
    #    traj = dp.DouglasPeuckerPoints(data, n_pts_resample)
    #    s_demos.append(np.transpose(traj))
    #    s_demo, = ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'g', lw=3, alpha=0.4)
    #    final_pt = final_pt + s_demos[i][:, -1] if final_pt is not None else s_demos[i][:, -1]
    #    print(final_pt)
    #for i in range(len(ffnames)):
    #    [x, y, z] = read_3D_h5(ffnames[i])
    #    data = np.hstack((np.reshape(x, (len(x), 1)), np.reshape(y, (len(y), 1)), np.reshape(z, (len(z), 1))))
    #    traj = dp.DouglasPeuckerPoints(data, n_pts_resample)
    #    f_demos.append(np.transpose(traj))
    #    f_demo, = ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'r', lw=3, alpha=0.4)
    #    final_pt = final_pt + f_demos[i][:, -1] if final_pt is not None else s_demos[i][:, -1]
    #    print(final_pt)
    #
    #all_demos = s_demos + f_demos
    #
    #final_pt = final_pt / len(all_fnames)
    #
    ##K = 10000.0
    ##obj = LFFD_GMM(copy.copy(s_demos), copy.copy(f_demos))
    ##obj.encode_GMMs(5)
    #
    #inds = [0, n_pts_resample - 1]
    #
    #for i in range(len(all_fnames)):
    #    consts = all_demos[i][:, inds]
    #    consts[:, 1] = final_pt
    #    print(consts)
    #    #input()
    #    #traj_fsil = obj.get_successful_reproduction(K, inds, consts)
    #    #np.savetxt('reaching_all_f_repro' + str(i) + '.txt', traj_fsil)
    #    traj_fsil = np.loadtxt('reaching_all_f_repro' + str(i) + '.txt')
    #    fsil_traj, = ax.plot(traj_fsil[0, :], traj_fsil[1, :], traj_fsil[2, :], 'k', lw=5)
    #    ax.plot(consts[0, 0], consts[1, 0], consts[2, 0], 'k.', ms=12, mew=3)
    #    ax.plot(consts[0, 1], consts[1, 1], consts[2, 1], 'kx', ms=12, mew=3)
    #    
    ##Xc,Yc,Zc = data_for_cylinder_along_z(0.2,0.2,0.05,0.1)
    ##ax.plot_surface(Xc, Yc, Zc, 'r', alpha=0.5)
    #
    #plot_3D_cylinder(ax, radius=0.03, height=0.1, elevation=0, x_center = -0.5, y_center = -0.45, resolution=100, color='r')
    #
    #xmin = 0.0
    #xmax = 0.2
    #ymin = -0.4
    #ymax = -0.5
    #zmin = 0.0
    #zmax = 0.1
    #
    #plot_cube(xmin, xmax, ymin, ymax, zmin, zmax, ax)
    #
    ## Get rid of colored axes planes
    ## First remove fill
    #ax.xaxis.pane.fill = False
    #ax.yaxis.pane.fill = False
    #ax.zaxis.pane.fill = False
    #
    ## Now set color to white (or whatever is "invisible")
    #ax.xaxis.pane.set_edgecolor('w')
    #ax.yaxis.pane.set_edgecolor('w')
    #ax.zaxis.pane.set_edgecolor('w')
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])
    #ax.set_zticklabels([])
    #
    #plt.show()
    
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #
    #for i in range(len(sfnames)):
    #    [x, y, z] = read_3D_h5(sfnames[i])
    #    data = np.hstack((np.reshape(x, (len(x), 1)), np.reshape(y, (len(y), 1)), np.reshape(z, (len(z), 1))))
    #    traj = dp.DouglasPeuckerPoints(data, n_pts_resample)
    #    #ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], c=color[i])
    #    #plt.show()
    #    s_demos.append(np.transpose(traj))
    #    s_demo, = ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'g', lw=3, alpha=0.4)
    #    #ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], c='g')
    #for i in range(len(ffnames)):
    #    [x, y, z] = read_3D_h5(ffnames[i])
    #    data = np.hstack((np.reshape(x, (len(x), 1)), np.reshape(y, (len(y), 1)), np.reshape(z, (len(z), 1))))
    #    traj = dp.DouglasPeuckerPoints(data, n_pts_resample)
    #    #ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], c=color[i])
    #    #plt.show()
    #    f_demos.append(np.transpose(traj))
    #    f_demo, = ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'r', lw=3, alpha=0.4)
    #    #ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], c='r')
    #    
    #    
    #K = 10000.0
    #obj = LFFD_GMM(copy.copy(s_demos), copy.copy(f_demos))
    #obj.encode_GMMs(5)
    ##traj_fsil = obj.get_successful_reproduction(K)
    #d = 1
    #if (obj.num_s > 0):
    #    gmr_ms, = ax.plot(obj.mu_s[d, :], obj.mu_s[d + 1, :], obj.mu_s[d + 2, :], 'g', lw=5, alpha=0.8)
    #if (obj.num_f > 0):
    #    gmr_mf, = ax.plot(obj.mu_f[d, :], obj.mu_f[d + 1, :], obj.mu_f[d + 2, :], 'r', lw=5, alpha=0.8)
    #
    ##traj_fsil = np.loadtxt('tlfsd_reaching_repro.txt')
    #traj_fsil = np.loadtxt('reaching_traj50.txt')
    #
    #print(np.shape(traj_fsil))
    #print(traj_fsil)
    #fsil, = ax.plot(traj_fsil[0, :], traj_fsil[1, :], traj_fsil[2, :], 'k-', lw=5, ms=6)
    #
    #xmin = 0.0
    #xmax = 0.2
    #ymin = -0.4
    #ymax = -0.5
    #zmin = 0.0
    #zmax = 0.1
    #
    #plot_cube(xmin, xmax, ymin, ymax, zmin, zmax, ax)
    ##ax.set_facecolor('0.9')
    ##ax.set_xticks(
    ##plt.axis('off')
    ## Get rid of colored axes planes
    ## First remove fill
    #ax.xaxis.pane.fill = False
    #ax.yaxis.pane.fill = False
    #ax.zaxis.pane.fill = False
    #
    ## Now set color to white (or whatever is "invisible")
    #ax.xaxis.pane.set_edgecolor('w')
    #ax.yaxis.pane.set_edgecolor('w')
    #ax.zaxis.pane.set_edgecolor('w')
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])
    #ax.set_zticklabels([])
    ##plt.legend((s_demo, f_demo, gmr_ms, gmr_mf, fsil), ('Successful Set', 'Failed Set', 'Successful GMR Mean', 'Failed GMR Mean', 'TLFSD'), fontsize='x-large')# bbox_to_anchor=(1, 0.5))
    #plt.show()
    #
    #X = traj_fsil
    #save_s = input('Would you like to save? (y/n): ')
    #if (save_s == 'y'):
    #    fname = input('Please enter a filename: ')
    #    np.savetxt(fname, X) 
    print()

    
def pushing_example():
    
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
    
    for i in range(len(sfnames)):
        [x, y, z] = read_3D_h5(sfnames[i])
        data = np.hstack((np.reshape(x, (len(x), 1)), np.reshape(y, (len(y), 1)), np.reshape(z, (len(z), 1))))
        traj = dp.DouglasPeuckerPoints(data, n_pts_resample)
        #ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], c=color[i])
        #plt.show()
        s_demos.append(np.transpose(traj))
        #s_demo, = plt.plot(traj[:, 0], traj[:, 1], 'g', lw=3, alpha=0.4)
        s_demo, = ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'g', lw=3, alpha=0.4)
    for i in range(len(ffnames)):
        [x, y, z] = read_3D_h5(ffnames[i])
        data = np.hstack((np.reshape(x, (len(x), 1)), np.reshape(y, (len(y), 1)), np.reshape(z, (len(z), 1))))
        traj = dp.DouglasPeuckerPoints(data, n_pts_resample)
        #ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], c=color[i])
        #plt.show()
        f_demos.append(np.transpose(traj))
        #f_demo, = plt.plot(traj[:, 0], traj[:, 1], 'r', lw=3, alpha=0.4)
        f_demo, = ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'r', lw=3, alpha=0.4)
        
        
        
    #K = 10000.0
    #obj = LFFD_GMM(copy.copy(s_demos), copy.copy(f_demos))
    #obj.encode_GMMs(5)
    #traj_fsil = obj.get_successful_reproduction(K)
    
    #traj_fsil = np.loadtxt('tlfsd_pushing_repro.txt')
    traj_fsil = np.loadtxt('fsil_pushing50.txt')
    
    print(np.shape(traj_fsil))
    print(traj_fsil)
    fsil, = ax.plot(traj_fsil[0, :], traj_fsil[1, :], traj_fsil[2, :], 'k-', lw=5, ms=6)
    
    #xmin = 0.0
    #xmax = 0.2
    #ymin = -0.4
    #ymax = -0.5
    #zmin = 0.0
    #zmax = 0.1
    #
    plot_irregular_cube([0.49, -0.52, 0], 
                        [0.49, -0.52, 0.1], 
                        [0.27, -0.41, 0.1], 
                        [0.27, -0.41, 0], 
                        [0.49+0.11, -0.52+0.03, 0], 
                        [0.49+0.11, -0.52+0.03, 0.1], 
                        [0.27+0.11, -0.41+0.03, 0.1], 
                        [0.27+0.11, -0.41+0.03, 0], ax)
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
    plt.show()
    
    #X = traj_fsil
    #save_s = input('Would you like to save? (y/n): ')
    #if (save_s == 'y'):
    #    fname = input('Please enter a filename: ')
    #    np.savetxt(fname, X)
    
    
if __name__ == '__main__':
    pushing_example()