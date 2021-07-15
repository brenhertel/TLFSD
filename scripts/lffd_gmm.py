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

class LFFD_GMM(object):

    def __init__(self, sucesses=[], failures=[]):
        self.other_coords = False
        self.n_states = 4
        self.DEBUG = True    
        self.obstacles = []
        self.num_obs = 0
        
        self.s = sucesses
        self.f = failures
        self.num_s = len(self.s)
        self.num_f = len(self.f)
        
        if (self.num_s > 0):
            (self.n_dims, self.n_pts) = np.shape(self.s[0])
        elif (self.num_f > 0):
            (self.n_dims, self.n_pts) = np.shape(self.f[0])
        else:
            print('No demonstrations given! Cannot continue.')
            exit()
            
        self.t = np.linspace(0, 1, self.n_pts).reshape((1, self.n_pts))
        for s_ind in range(self.num_s):
            #if (self.DEBUG):
            #    print('s demo')
            #    print(self.s[s_ind])
            #    print('time')
            #    print(self.t)
            #    print('stacked')
            #    print(np.vstack((self.t, self.s[s_ind])))
            self.s[s_ind] = np.vstack((self.t, self.s[s_ind]))
        for f_ind in range(self.num_f):
            self.f[f_ind] = np.vstack((self.t, self.f[f_ind]))
        self.set_params()
            
    def encode_GMMs(self, num_states=4, include_other_systems=False):
        #encode successful and failed demonstrations into GMMs, then use GMR to get mean and covariance
        self.n_states = num_states
        if (self.num_s > 0):
            self.s_gmm = GMM_GMR(num_states)
            if (self.DEBUG):
                print('stacked s')
                print(np.hstack(self.s))
            self.s_gmm.fit(np.hstack(self.s))
            self.s_gmm.predict(self.t)
            self.mu_s = self.s_gmm.getPredictedData()
            self.cov_s = self.s_gmm.getPredictedSigma()
            self.inv_cov_s = np.zeros((np.shape(self.cov_s)))
            for i in range(self.n_pts):
                self.inv_cov_s[:, :, i] = np.linalg.inv(self.cov_s[:, :, i])
            if (self.DEBUG):
                print('mu')
                print(np.shape(self.mu_s))
                print(self.mu_s)
                print('cov')
                print(np.shape(self.cov_s))
                print(self.cov_s)
        if (self.num_f > 0):
            self.f_gmm = GMM_GMR(num_states)
            self.f_gmm.fit(np.hstack(self.f))
            self.f_gmm.predict(self.t)
            self.mu_f = self.f_gmm.getPredictedData()
            self.cov_f = self.f_gmm.getPredictedSigma()
            self.inv_cov_f = np.zeros((np.shape(self.cov_f)))
            for i in range(self.n_pts):
                self.inv_cov_f[:, :, i] = self.cov_f[:, :, i]#np.linalg.inv(self.cov_f[:, :, i])
            
        if (include_other_systems):
            self.other_coords = True
            
            self.T = np.diag(np.ones((self.n_pts-1,)),1) - np.diag(np.ones((self.n_pts,)))
            
            self.L = 2.*np.diag(np.ones((self.n_pts,))) - np.diag(np.ones((self.n_pts-1,)),1) - np.diag(np.ones((self.n_pts-1,)),-1)
            self.L[0,1] = -2.
            self.L[-1,-2] = -2.
            self.L = self.L / 2.
            
            self.ts = []
            self.ls = []
            self.tf = []
            self.lf = []
            
            for s_ind in range(self.num_s):
                self.ts.append(np.vstack((self.t, np.matmul(self.s[s_ind][1:, :], self.T))))
                self.ls.append(np.vstack((self.t, np.matmul(self.s[s_ind][1:, :], self.L))))
            for f_ind in range(self.num_f):
                self.tf.append(np.vstack((self.t, np.matmul(self.f[f_ind][1:, :], self.T))))
                self.lf.append(np.vstack((self.t, np.matmul(self.f[f_ind][1:, :], self.L))))
            
            if (self.num_s > 0):
                ts_gmm = GMM_GMR(num_states)
                ts_gmm.fit(np.hstack(self.ts))
                ts_gmm.predict(self.t)
                self.mu_ts = ts_gmm.getPredictedData()
                self.cov_ts = ts_gmm.getPredictedSigma()
                self.inv_cov_ts = np.zeros((np.shape(self.cov_ts)))
                for i in range(self.n_pts):
                    self.inv_cov_ts[:, :, i] = np.linalg.inv(self.cov_ts[:, :, i])
            if (self.num_f > 0):
                tf_gmm = GMM_GMR(num_states)
                tf_gmm.fit(np.hstack(self.tf))
                tf_gmm.predict(self.t)
                self.mu_tf = tf_gmm.getPredictedData()
                self.cov_tf = tf_gmm.getPredictedSigma()
                self.inv_cov_tf = np.zeros((np.shape(self.cov_tf)))
                for i in range(self.n_pts):
                    self.inv_cov_tf[:, :, i] = np.linalg.inv(self.cov_tf[:, :, i])
            if (self.num_s > 0):
                ls_gmm = GMM_GMR(num_states)
                ls_gmm.fit(np.hstack(self.ls))
                ls_gmm.predict(self.t)
                self.mu_ls = ls_gmm.getPredictedData()
                self.cov_ls = ls_gmm.getPredictedSigma()
                self.inv_cov_ls = np.zeros((np.shape(self.cov_ls)))
                for i in range(self.n_pts):
                    self.inv_cov_ls[:, :, i] = np.linalg.inv(self.cov_ls[:, :, i])
            if (self.num_f > 0):
                lf_gmm = GMM_GMR(num_states)
                lf_gmm.fit(np.hstack(self.lf))
                lf_gmm.predict(self.t)
                self.mu_lf = lf_gmm.getPredictedData()
                self.cov_lf = lf_gmm.getPredictedSigma()
                self.inv_cov_lf = np.zeros((np.shape(self.cov_lf)))
                for i in range(self.n_pts):
                    self.inv_cov_lf[:, :, i] = np.linalg.inv(self.cov_lf[:, :, i])
        return
        
    def get_successful_reproduction(self, k=None, indices=[], constraints=[]):
        #get suggest_traj
        self.inds = indices
        self.consts = constraints
        if (self.num_f > 0):
            if (self.num_s > 0):
                self.w = np.linalg.norm(self.mu_s - self.mu_f, axis=0)
                if (self.other_coords):
                    self.wt = np.linalg.norm(self.mu_ts - self.mu_tf, axis=0)
                    self.wl = np.linalg.norm(self.mu_ls - self.mu_lf, axis=0)
            else:
                if (self.consts != []):
                    temp_succ = np.vstack((np.linspace(self.consts[0, 0], self.consts[0, -1], self.n_pts), np.linspace(self.consts[-1, 0], self.consts[-1, -1], self.n_pts) ))
                else:
                    temp_succ = np.vstack((np.linspace(self.mu_f[1, 0], self.mu_f[1, -1], self.n_pts), np.linspace(self.mu_f[2, 0], self.mu_f[2, -1], self.n_pts)))
                self.w = np.linalg.norm(temp_succ - self.mu_f[1:, :], axis=0)
                if (self.other_coords):
                    self.wt = np.linalg.norm(np.matmul(temp_succ, self.T) - self.mu_tf, axis=0)
                    self.wl = np.linalg.norm(np.matmul(temp_succ, self.L) - self.mu_lf, axis=0)
            self.w = self.w / np.max(self.w)
            if (self.other_coords):
                self.wt = self.wt / np.max(self.wt)
                self.wl = self.wl / np.max(self.wl)
                
        if (self.num_s > 0):
            suggest_traj = self.mu_s[1:, :]
        else:
            suggest_traj = self.mu_f[1:, :] + np.random.normal(0, 5, self.n_pts * self.n_dims).reshape((self.n_dims, self.n_pts))
            
        if (k is None):
            self.K = self.est_k(suggest_traj)
        else:
            self.K = k
            
        #suggest_traj = suggest_traj + + np.random.normal(0, 10, np.size(suggest_traj)).reshape(np.shape(suggest_traj))
            
        import lte
        suggest_traj = np.transpose(lte.LTE_ND_any_constraints(np.transpose(suggest_traj), np.transpose(self.consts), self.inds))
            
        #if (constraints != []):
        #   suggest_traj = np.hstack((np.linspace(self.consts[0, 0], self.consts[0, -1], self.n_pts), np.linspace(self.consts[-1, 0], self.consts[-1, -1], self.n_pts) ))
        #   suggest_traj = np.hstack((np.linspace(self.consts[0, 0], self.consts[0, -1], self.n_pts) + np.random.normal(0, 5, self.n_pts), np.linspace(self.consts[-1, 0], self.consts[-1, -1], self.n_pts) + np.random.normal(0, 5, self.n_pts)))
            
        self.costs = []
        res = minimize(self.get_traj_cost, suggest_traj.flatten(), tol=1e-6, options={'disp': self.DEBUG})
        self.best_traj = np.reshape(res.x, (self.n_dims, self.n_pts))
        if (self.num_f > 0):
            print(self.w)
        return self.best_traj
            
    def iterate_success(self, k=None, indices=[], constraints=[]):
        is_successful = False
        while not is_successful:
            self.encode_GMMs(self.n_states, self.other_coords)
            repro = self.get_successful_reproduction(k=k, indices=indices, constraints=constraints)
            print([self.num_s, self.num_f])
            self.plot_results(mode='show')
            print('Was this reproduction successful? (y/n)')
            ans = input()
            if (ans == 'y'):
                self.s.append(np.vstack((self.t, repro)))
                self.num_s += 1
                is_successful = True
            else:
                self.f.append(np.vstack((self.t, repro)))
                self.num_f += 1
        return repro
            
    def get_traj_cost(self, X):
        self.traj = X.reshape((self.n_dims, self.n_pts))
        if (self.other_coords):
            self.t_traj = np.matmul(self.traj, self.T)
            self.l_traj = np.matmul(self.traj, self.L)
        J1 = self.get_failure_cost() 
        J2 = self.get_elastic_cost()
        J3 = self.get_constraint_cost()
        total = J1 + J2 + J3
        if (self.DEBUG):
            print("J1 {}, J2 {}, J3 {}, Total {}".format(J1,J2,J3,total))
        self.costs.append(total)
        return total
            
    def set_params(self, beta_c=1.0, beta_g=1.0, beta_l=1.0):
        self.alpha_c = beta_c
        self.alpha_g = beta_g
        self.alpha_l = beta_l
            
    def get_failure_cost(self):
        sum = 0.
        #print('mu')
        #print(self.mu_s)
        for n in range(self.n_pts):
            if (self.num_s > 0):
                diff = self.traj[:, n] - self.mu_s[1:, n]
                sum += self.alpha_c * np.matmul(np.matmul(np.transpose(diff), self.inv_cov_s[:, :, n]), diff)
            if (self.num_f > 0):
                diff = self.traj[:, n] - self.mu_f[1:, n]
                sum -= self.w[n] * self.alpha_c * np.matmul(np.matmul(np.transpose(diff), self.inv_cov_f[:, :, n]), diff)
                
            if (self.other_coords):
            
                if (self.num_s > 0):
                    diff = self.t_traj[:, n] - self.mu_ts[1:, n]
                    sum += self.alpha_g * np.matmul(np.matmul(np.transpose(diff), self.inv_cov_ts[:, :, n]), diff)
                    diff = self.l_traj[:, n] - self.mu_ls[1:, n]
                    sum += self.alpha_l * np.matmul(np.matmul(np.transpose(diff), self.inv_cov_ls[:, :, n]), diff)
                    
                if (self.num_f > 0):
                    diff = self.t_traj[:, n] - self.mu_tf[1:, n]
                    sum -= self.wt[n] * self.alpha_g * np.matmul(np.matmul(np.transpose(diff), self.inv_cov_tf[:, :, n]), diff)
                    diff = self.l_traj[:, n] - self.mu_lf[1:, n]
                    sum -= self.wl[n] * self.alpha_l * np.matmul(np.matmul(np.transpose(diff), self.inv_cov_lf[:, :, n]), diff)
        return sum
        
    def get_elastic_cost(self):
        sum = 0.
        for n in range(self.n_pts - 1):
            sum += np.linalg.norm(self.traj[:, n+1] - self.traj[:, n])**2
        return self.K * sum / 2
        
    def get_constraint_cost(self):
        sum = 0.
        for i in range(len(self.inds)):
            sum += np.linalg.norm(self.traj[:, self.inds[i]] - self.consts[:, i])**2
        return 1e12 * sum
        
    def est_k(self, traj):
        sum = 0.
        gamma = 1e3
        for i in range(len(self.inds)):
            sum += np.linalg.norm(traj[:, self.inds[i]] - self.consts[:, i])
        return gamma * sum
       
    def add_obstacle(self, obs):
       self.num_obs += 1
       self.obstacles.append(obs)
       return
       
    def plot_results(self, mode='save', filepath=''):
        tt = self.t.flatten()
        print(self.consts)
        for d in range(self.n_dims):
            #print('dimension: ' + str(d))
            #print(self.cov_s)
            fig = plt.figure()
            title = 'dimension ' + str(d+1) + ' vs. t'
            plt.title(title)
            #plt.xticks([])
            #plt.yticks([])
            for s_ind in range(self.num_s):
                plt.plot(tt, self.s[s_ind][d + 1, :], 'g', lw=3)
            for f_ind in range(self.num_f):
                plt.plot(tt, self.f[f_ind][d + 1, :], 'r', lw=3)
            if (self.num_s > 0):
                plt.plot(tt, self.mu_s[d + 1, :], 'b', lw=5)
                print(self.cov_s[d, d, :])
                plt.fill_between(tt, self.mu_s[d + 1, :] + self.cov_s[d, d, :], self.mu_s[d + 1, :] - self.cov_s[d, d, :], color='b', alpha=0.3)
            if (self.num_f > 0):
                plt.plot(tt, self.mu_f[d + 1, :], 'm', lw=5)
                plt.fill_between(tt, self.mu_f[d + 1, :] + self.cov_f[d, d, :], self.mu_f[d + 1, :] - self.cov_f[d, d, :], color='m', alpha=0.3)
            plt.plot(tt, self.best_traj[d, :], 'k', lw=5)
            for i in range(len(self.inds)):
                plt.plot(tt[self.inds[i]], self.consts[d, i], 'k.', ms=15)
            if (mode == 'save'):
                plt.savefig(filepath + title + '.png')
        if self.other_coords:
            for d in range(self.n_dims):
                #print('dimension: ' + str(d))
                #print(self.cov_s)
                fig = plt.figure()
                title = 'Tangent dimension ' + str(d+1) + ' vs. t'
                plt.title(title)
                #plt.xticks([])
                #plt.yticks([])
                for s_ind in range(self.num_s):
                    plt.plot(tt, self.ts[s_ind][d + 1, :], 'g', lw=3)
                for f_ind in range(self.num_f):
                    plt.plot(tt, self.tf[f_ind][d + 1, :], 'r', lw=3)
                if (self.num_s > 0):
                    plt.plot(tt, self.mu_ts[d + 1, :], 'b', lw=5)
                    print(self.cov_ts[d, d, :])
                    plt.fill_between(tt, self.mu_ts[d + 1, :] + self.cov_ts[d, d, :], self.mu_ts[d + 1, :] - self.cov_ts[d, d, :], color='b', alpha=0.3)
                if (self.num_f > 0):
                    plt.plot(tt, self.mu_tf[d + 1, :], 'm', lw=5)
                    plt.fill_between(tt, self.mu_tf[d + 1, :] + self.cov_tf[d, d, :], self.mu_tf[d + 1, :] - self.cov_tf[d, d, :], color='m', alpha=0.3)
                plt.plot(tt, np.matmul(self.best_traj[d, :], self.T), 'k', lw=5)
                if (mode == 'save'):
                    plt.savefig(filepath + title + '.png')
                    
                fig = plt.figure()
                title = 'Laplacian dimension ' + str(d+1) + ' vs. t'
                plt.title(title)
                #plt.xticks([])
                #plt.yticks([])
                for s_ind in range(self.num_s):
                    plt.plot(tt, self.ls[s_ind][d + 1, :], 'g', lw=3)
                for f_ind in range(self.num_f):
                    plt.plot(tt, self.lf[f_ind][d + 1, :], 'r', lw=3)
                if (self.num_s > 0):
                    plt.plot(tt, self.mu_ls[d + 1, :], 'b', lw=5)
                    print(self.cov_ts[d, d, :])
                    plt.fill_between(tt, self.mu_ls[d + 1, :] + self.cov_ls[d, d, :], self.mu_ls[d + 1, :] - self.cov_ls[d, d, :], color='b', alpha=0.3)
                if (self.num_f > 0):
                    plt.plot(tt, self.mu_lf[d + 1, :], 'm', lw=5)
                    plt.fill_between(tt, self.mu_lf[d + 1, :] + self.cov_lf[d, d, :], self.mu_lf[d + 1, :] - self.cov_lf[d, d, :], color='m', alpha=0.3)
                plt.plot(tt, np.matmul(self.best_traj[d, :], self.L), 'k', lw=5)
                if (mode == 'save'):
                    plt.savefig(filepath + title + '.png')
            
        if (self.n_dims == 2):
            fig = plt.figure()
            title = '2 dimensional result'
            plt.title(title)
            #plt.xticks([])
            #plt.yticks([])
            for s_ind in range(self.num_s):
                plt.plot(self.s[s_ind][1, :], self.s[s_ind][2, :], 'g', lw=3)
            for f_ind in range(self.num_f):
                plt.plot(self.f[f_ind][1, :], self.f[f_ind][2, :], 'r', lw=3)
            if (self.num_s > 0):
                plt.plot(self.mu_s[1, :], self.mu_s[2, :], 'b', lw=5)
                #plt.fill_between(self.t, self.mu_s[d + 1, :] + self.cov_s[d + 1, 0, :], self.mu_s[d + 1, :] - self.cov_s[d + 1, 0, :], color='b', alpha=0.3)
            if (self.num_f > 0):
                plt.plot(self.mu_f[1, :], self.mu_f[2, :], 'm', lw=5)
                #plt.fill_between(self.t, self.mu_f[d + 1, :] + self.cov_f[d + 1, 0, :], self.mu_f[d + 1, :] - self.cov_f[d + 1, 0, :], color='m', alpha=0.3)
            plt.plot(self.best_traj[0, :], self.best_traj[1, :], 'k', lw=5)
            for i in range(len(self.inds)):
                plt.plot(self.consts[0, i], self.consts[1, i], 'k.', ms=15)
            ax = plt.gca()
            for j in range(self.num_obs):
                pc = copy.copy(self.obstacles[j])
                ax.add_patch(pc)
            if (mode == 'save'):
                plt.savefig(filepath + title + '.png')
        if (mode != 'save'):
            plt.show()
        plt.close('all')
        return

def read_my_python_demos(shape_name, n, smoothed=True):
    filename = '../h5 files/' + shape_name + '_demos.h5'
    hf = h5py.File(filename, 'r')
    #print(shape_name)
    #navigate to necessary data and store in numpy arrays
    shape = hf.get(shape_name)
    if smoothed:
        trajs = shape.get('smoothed')
    else:
        trajs = shape.get('unsmoothed')
    demo = trajs.get(str(n))
    pos_data = np.array(demo)
    #close out file
    hf.close()
    return pos_data
    
def read_my_demos(shape_name, n):
    filename = '../h5 files/my_demos.h5'
    hf = h5py.File(filename, 'r')
    #print(shape_name)
    #navigate to necessary data and store in numpy arrays
    shape = hf.get(shape_name)
    demo = shape.get(str(n))
    pos_data = np.array(demo)
    #close out file
    hf.close()
    return pos_data

def main1():    
    #shape_names = ['s', 'c', 'bent', 'sine', 'straight', 'u']
    shape_names = ['hill']#['s', 'bent']
    colors = ['r', 'g', 'b', 'c', 'm', 'k']
    num_demos = 5
    #fig = plt.figure()
    s_demos = []
    f_demos = []
    for i in range(len(shape_names)):
        for n in range(num_demos):
            #data = read_my_demos(shape_names[i], n + 1)
            data = np.transpose(read_my_python_demos(shape_names[i], n + 1))
            print(data)
            #plt.plot(data[0, :], data[1, :], colors[i])
            
            traj = dp.DouglasPeuckerPoints(np.transpose(data), 50)
            if (i == 0):
                s_demos.append(np.transpose(traj))
            else:
                f_demos.append(np.transpose(traj))
    
    obj = LFFD_GMM(s_demos, f_demos)
    obj.encode_GMMs()
    
    inds = [0, 49]
    consts = np.transpose(np.array(traj[inds, :]) + 10) 
    
    parent_fpath = 'C:/Users/BH/Documents/GitHub/pearl_test_env/pictures/failure_lfd/new_tuning_example/'
    K = 0
    
    while (K < 1e6):
        plt_fpath = parent_fpath + str(K) + '/'
        try:
            os.makedirs(plt_fpath)
        except OSError:
            print ("Creation of the directory %s failed" % plt_fpath)
        else:
            print ("Successfully created the directory %s" % plt_fpath)
        
        obj.get_successful_reproduction(K, inds, consts)
        obj.plot_results(mode='save', filepath=plt_fpath)
        K = 2 * K + 1
    
    #plt.show()
    
def main2():    
    #shape_names = ['s', 'c', 'bent', 'sine', 'straight', 'u']
    shape_names = ['sw']#['s', 'bent']
    colors = ['r', 'g', 'b', 'c', 'm', 'k']
    num_demos = 1
    #fig = plt.figure()
    s_demos = []
    f_demos = []
    for i in range(len(shape_names)):
        for n in range(num_demos):
            #data = read_my_demos(shape_names[i], n + 1)
            data = np.transpose(read_my_python_demos(shape_names[i], n + 1))
            print(data)
            #plt.plot(data[0, :], data[1, :], colors[i])
            
            traj = dp.DouglasPeuckerPoints(np.transpose(data), 50)
            f_demos.append(np.transpose(traj))
    
    obj = LFFD_GMM(s_demos, f_demos)
    obj.encode_GMMs(3, False)
    
    from matplotlib.patches import Rectangle
    rect = Rectangle((-200, -50), 80, 80)
    obj.add_obstacle(rect)
    inds = [0, 49]
    consts = np.transpose(np.array(traj[inds, :])) 
    
    K = 39.0
    obj.iterate_success(K, inds, consts)
    
    #plt.show()
    
def main3():    
    #shape_names = ['s', 'c', 'bent', 'sine', 'straight', 'u']
    shape_names = ['light_z', 'hard_z']#['s', 'bent']
    colors = ['r', 'g', 'b', 'c', 'm', 'k']
    num_demos = 3
    #fig = plt.figure()
    s_demos = []
    f_demos = []
    for i in range(len(shape_names)):
        for n in range(num_demos):
            #data = read_my_demos(shape_names[i], n + 1)
            data = np.transpose(read_my_python_demos(shape_names[i], n + 1))
            print(data)
            #plt.plot(data[0, :], data[1, :], colors[i])
            
            traj = dp.DouglasPeuckerPoints(np.transpose(data), 50)
            if (i == 0):
                s_demos.append(np.transpose(traj))
            else:
                f_demos.append(np.transpose(traj))
    
    obj_cart = LFFD_GMM(copy.copy(s_demos), copy.copy(f_demos))
    obj_all = LFFD_GMM(copy.copy(s_demos), copy.copy(f_demos))
    obj_cart.encode_GMMs(3, False)
    obj_all.encode_GMMs(5, True)
    
    inds = [0, 25, 49]
    consts = np.transpose(np.array(traj[inds, :])) 
    
    K = 5.0
    cart_traj = obj_cart.get_successful_reproduction(0.1, inds, consts)
    obj_cart.plot_results(mode='show')
    all_traj = obj_all.get_successful_reproduction(K, inds, consts)
    obj_all.plot_results(mode='show')
    plt.figure()
    plt.plot(cart_traj[0, :], cart_traj[1, :], 'k-', lw=5)
    plt.plot(all_traj[0, :], all_traj[1, :], color='0.5', linestyle='-', lw=5)
    for i in range(len(inds)):
        plt.plot(consts[0, i], consts[1, i], 'k.', ms=15)
    plt.show()
    
if __name__ == '__main__':
    main3()