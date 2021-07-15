import numpy as np
import matplotlib.pyplot as plt
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'C:/Users/BH/Documents/GitHub/pearl_test_env/Guassian-Mixture-Models')
from GMM_GMR import GMM_GMR
from scipy.optimize import minimize
import os
import copy

class FSIL(object):

    def __init__(self, sucesses=[], failures=[]):
        #default constants
        self.other_coords = False
        self.n_states = 4
        self.DEBUG = True    
        self.obstacles = []
        self.num_obs = 0
        self.set_params()
        
        #get successful and failed sets
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
            
        #append time dimension (assume not given)
        self.t = np.linspace(0, 1, self.n_pts).reshape((1, self.n_pts))
        for s_ind in range(self.num_s):
            self.s[s_ind] = np.vstack((self.t, self.s[s_ind]))
        for f_ind in range(self.num_f):
            self.f[f_ind] = np.vstack((self.t, self.f[f_ind]))
            
    def encode_GMMs(self, num_states=4, include_other_systems=False):
        #encode successful and failed demonstrations into GMMs, then use GMR to get mean and covariance
        self.n_states = num_states
        #successful cartesian GMM
        if (self.num_s > 0):
            s_gmm = GMM_GMR(num_states)
            s_gmm.fit(np.hstack(self.s))
            s_gmm.predict(self.t)
            self.mu_s = s_gmm.getPredictedData()
            self.cov_s = s_gmm.getPredictedSigma()
            self.inv_cov_s = np.zeros((np.shape(self.cov_s)))
            for i in range(self.n_pts):
                self.inv_cov_s[:, :, i] = np.linalg.inv(self.cov_s[:, :, i])
        #failed cartesian GMM
        if (self.num_f > 0):
            f_gmm = GMM_GMR(num_states)
            f_gmm.fit(np.hstack(self.f))
            f_gmm.predict(self.t)
            self.mu_f = f_gmm.getPredictedData()
            self.cov_f = f_gmm.getPredictedSigma()
            self.inv_cov_f = np.zeros((np.shape(self.cov_f)))
            for i in range(self.n_pts):
                self.inv_cov_f[:, :, i] = self.cov_f[:, :, i]#np.linalg.inv(self.cov_f[:, :, i])
            
        if (include_other_systems):
            self.other_coords = True
            #graph incidence and graph laplacian matrices
            self.T = np.diag(np.ones((self.n_pts-1,)),1) - np.diag(np.ones((self.n_pts,)))
            
            self.L = 2.*np.diag(np.ones((self.n_pts,))) - np.diag(np.ones((self.n_pts-1,)),1) - np.diag(np.ones((self.n_pts-1,)),-1)
            self.L[0,1] = -2.
            self.L[-1,-2] = -2.
            self.L = self.L / 2.
            
            self.ts = []
            self.ls = []
            self.tf = []
            self.lf = []
            
            #append time
            for s_ind in range(self.num_s):
                self.ts.append(np.vstack((self.t, np.matmul(self.s[s_ind][1:, :], self.T))))
                self.ls.append(np.vstack((self.t, np.matmul(self.s[s_ind][1:, :], self.L))))
            for f_ind in range(self.num_f):
                self.tf.append(np.vstack((self.t, np.matmul(self.f[f_ind][1:, :], self.T))))
                self.lf.append(np.vstack((self.t, np.matmul(self.f[f_ind][1:, :], self.L))))
            
            #encode GMMs
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
        self.inds = indices
        self.consts = constraints
        
        #get suggest_traj
        if (self.num_s > 0):
            suggest_traj = self.mu_s[1:, :]
        else:
            suggest_traj = self.mu_f[1:, :] + 0.1
            
        self.K = k
        
        res = minimize(self.get_traj_cost, suggest_traj.flatten(), tol=1e-6) #minimize
        self.best_traj = np.reshape(res.x, (self.n_dims, self.n_pts))
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
        return total
            
    def set_params(self, alpha=1.0, beta_c=1.0, beta_g=1.0, beta_l=1.0):
        self.ws = 1
        self.wf = alpha
        self.alpha_c = beta_c
        self.alpha_g = beta_g
        self.alpha_l = beta_l
            
    def get_failure_cost(self):
        sum = 0.
        for n in range(self.n_pts):
            #cartesian success
            if (self.num_s > 0):
                diff = self.traj[:, n] - self.mu_s[1:, n]
                sum += self.ws * self.alpha_c * np.matmul(np.matmul(np.transpose(diff), self.inv_cov_s[:, :, n]), diff)
            #cartesian failed
            if (self.num_f > 0):
                diff = self.traj[:, n] - self.mu_f[1:, n]
                sum += self.wf * self.alpha_c / np.matmul(np.matmul(np.transpose(diff), self.inv_cov_f[:, :, n]), diff)
               
            #other coordinate systems
            if (self.other_coords):
            
                if (self.num_s > 0):
                    diff = self.t_traj[:, n] - self.mu_ts[1:, n]
                    sum += self.ws * self.alpha_g * np.matmul(np.matmul(np.transpose(diff), self.inv_cov_ts[:, :, n]), diff)
                    diff = self.l_traj[:, n] - self.mu_ls[1:, n]
                    sum += self.ws * self.alpha_l * np.matmul(np.matmul(np.transpose(diff), self.inv_cov_ls[:, :, n]), diff)
                    
                if (self.num_f > 0):
                    diff = self.t_traj[:, n] - self.mu_tf[1:, n]
                    sum += self.wf * self.alpha_g / np.matmul(np.matmul(np.transpose(diff), self.inv_cov_tf[:, :, n]), diff)
                    diff = self.l_traj[:, n] - self.mu_lf[1:, n]
                    sum += self.wf * self.alpha_l / np.matmul(np.matmul(np.transpose(diff), self.inv_cov_lf[:, :, n]), diff)
        return sum
        
    def get_elastic_cost(self):
        sum = 0.
        for n in range(self.n_pts - 1):
            sum += np.linalg.norm(self.traj[:, n+1] - self.traj[:, n])**2
        return self.K * sum / 2 #spring energy
        
    def get_constraint_cost(self):
        sum = 0.
        for i in range(len(self.inds)):
            sum += np.linalg.norm(self.traj[:, self.inds[i]] - self.consts[:, i])
        return 1e12 * sum**6 #strong attractor
       
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

def main1():    
    s_demos = []
    f_demos = []
    
    n_pts = 50
    
    x = np.linspace(-np.pi, np.pi, n_pts)
    yf = -9 * np.abs(np.sin(x))
    ys = -yf
    
    s_d = np.vstack((x, ys))
    f_d = np.vstack((x, yf))
    s_demos.append(s_d)
    f_demos.append(f_d)
    
    inds = [0, n_pts - 1]
    consts = s_d[:, inds]
    print(consts)
    consts[1, 0] = consts[1, 0] - 4
    
    K = 1.0
    obj_s = FSIL(copy.copy(s_demos))
    obj_s.encode_GMMs(8)
    traj_s = obj_s.get_successful_reproduction(K, inds, consts) 
    obj_s.plot_results(mode='show')
    
    obj_f = FSIL([], copy.copy(f_demos))
    obj_f.encode_GMMs(8)
    traj_f = obj_f.get_successful_reproduction(K, inds, consts)
    obj_f.plot_results(mode='show')
    
    K = 100.0
    obj_b = FSIL(copy.copy(s_demos), copy.copy(f_demos))
    obj_b.encode_GMMs(8)
    traj_b = obj_b.get_successful_reproduction(K, inds, consts)
    obj_b.plot_results(mode='show')
    
    plt.figure()
    
    s_demo, = plt.plot(x, ys, 'g', lw=1, alpha=0.6)
    f_demo, = plt.plot(x, yf, 'r', lw=1, alpha=0.6)
    
    succ, = plt.plot(traj_s[0, :], traj_s[1, :], 'k.', lw=3, ms=6)
    fail, = plt.plot(traj_f[0, :], traj_f[1, :], 'k--', lw=3)
    both, = plt.plot(traj_b[0, :], traj_b[1, :], 'k-', lw=5)
    
    
    plt.plot(consts[0, 0], consts[1, 0], 'ko', ms=10, mew=3)
    plt.plot(consts[0, 1], consts[1, 1], 'kx', ms=10, mew=3)
    plt.xticks([])
    plt.yticks([])
    
    plt.legend((s_demo, f_demo, succ, fail, both), ('Successful Set', 'Failed Set', 'Success Only', 'Failed Only', 'Failed and Successful'), fontsize=12)# bbox_to_anchor=(1, 0.5))
    plt.show()
    
if __name__ == '__main__':
    main1()