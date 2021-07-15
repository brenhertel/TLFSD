import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns#; sns.set_theme()
from scipy.optimize import minimize
import funx_approx
import h5py
import douglas_peucker as dp
from gmr import MVN, GMM, plot_error_ellipses
import similaritymeasures
import os

def test1():
    t = np.linspace(0, 1)
    s1 = (t - 0.5)**2 - 0.25
    s2 = -s1
    f1 = 0.125 * np.sin(2 * np.pi * t)
    
    start = [0, 0]
    end = [1, 0]
    obstacle = [0.5, 0]
        
    x_q = np.linspace(0.5, -0.5)
    failure_prbs = np.zeros((len(x_q), len(t)))
    
    for i in range(len(t)):
        rvs1 = norm(loc = s1[i], scale = 0.2)
        rvs2 = norm(loc = s2[i], scale = 0.2)
        rvf1 = norm(loc = f1[i], scale = 0.3)
        for j in range(len(x_q)):
            failure_prbs[j, i] = rvs1.pdf(x_q[j]) + rvs2.pdf(x_q[j]) - rvf1.pdf(x_q[j])
        
    
    plt.figure()
    #ax = sns.heatmap(failure_prbs)
    plt.plot(start[0], start[1], 'ko')
    plt.plot(end[0], end[1], 'kx')
    plt.plot(t, s1)
    plt.plot(t, s2)
    plt.plot(t, f1)
    plt.show()
    
    
#function to get the total distance of a n x d trajectory
#arguments
#traj: nxd vector, where n is number of points and d is number of dims
#returns the total distance of traj, calculated using euclidean distance  
def get_traj_dist(traj):
    dist = 0.
    for n in range(len(traj) - 1):
        dist = dist + np.linalg.norm(traj[n + 1] - traj[n])
    #if (DEBUG):
    #    print('Traj total dist: %f' % (dist))
    return dist
    
class LFFD(object):
    
    def __init__(self, successes=[], failures=[], k=6):
        self.s = successes
        self.f = failures
        traj_lens = []
        for s_traj in self.s:
            traj_lens.append(get_traj_dist(s_traj))
        for f_traj in self.f:
            traj_lens.append(get_traj_dist(f_traj))
        mean_traj_len = np.mean(np.array(traj_lens))
        print(mean_traj_len)
        self.success_spread = 0.2 * mean_traj_len
        self.failure_spread = 0.3 * mean_traj_len
        self.k2 = k / 2 # k / 2 for spring attractor
        self.prb_succ = []
        self.gauss_calced = False
        self.n_components = 4
        self.is_gmm = False
        
    def calc_consts(self, start=None, end=None):
        #get shape of trajs
        (self.n_pts, self.n_dims) = np.shape(self.s[0]) if self.s else np.shape(self.f[0])
        
        #if (start == None):
        #    self.init = self.s[0][0, :] if not self.s else self.f[0][0, :]
        #if (end == None):
        #    self.end = self.s[0][-1, :] if not self.s else self.f[0][-1, :]
        
        self.st = start
        self.en = end
        
    def calculate_prb_success(self, plot=False, save=False, fpath=''):
        self.calc_consts()
        for d in range(self.n_dims): #currently unidimensional process
            dim_prb_succ = np.zeros((self.n_pts, self.n_pts))
            #get max and min query values
            max_val = -float('inf')
            min_val = float('inf')
            for i in range(self.n_pts):
                for s_traj in self.s:
                    if s_traj[i, d] > max_val:
                        max_val = s_traj[i, d]
                    if s_traj[i, d] < min_val:
                        min_val = s_traj[i, d]
                for f_traj in self.f:
                    if f_traj[i, d] > max_val:
                        max_val = f_traj[i, d]
                    if f_traj[i, d] < min_val:
                        min_val = f_traj[i, d]
            #deal with possible positive/negative
            q_max = max_val * 1.2 if max_val > 0 else max_val * 0.8
            q_min = min_val * 0.8 if min_val > 0 else min_val * 1.2
            query_vals = np.linspace(q_max, q_min, self.n_pts)
            #query probabilities
            #for i in range(self.n_pts):
            #    #create a gaussian which will add to success or failure probabilities 
            #    s_gauss = [norm(loc=s_traj[i, d], scale=self.success_spread) for s_traj in self.s]
            #    f_gauss = [norm(loc=f_traj[i, d], scale=self.failure_spread) for f_traj in self.f]
            #    #add guassians for all points along query
            #    for j in range(self.n_pts):
            #        dim_prb_succ[j, i] = sum([s_g.pdf(query_vals[j]) for s_g in s_gauss]) - sum([f_g.pdf(query_vals[j]) for f_g in f_gauss])
            self.calc_gaussians()
            for i in range(self.n_pts):
                for j in range(self.n_pts):
                    dim_prb_succ[j, i] = self.query_gauss(d, i, query_vals[j])
            
            #return
            self.prb_succ.append(dim_prb_succ)
            if (plot):
                plt.figure()
                ax = sns.heatmap(dim_prb_succ)
                title = 'prbs dim ' + str(d)
                plt.title(title)
                if (save):
                    plt.savefig(fpath + title + '.png')
                else:
                    plt.show()
                plt.close('all')
                
        return self.prb_succ
    
    #def gradient_solver(self, init=None, final=None):
    #    if init is None:
    #        init = self.s[0][0, :] if not self.s else self.f[0][0, d]
    #    if final is None:
    #        final = self.s[0][-1, :] if not self.s else self.f[0][-1, d]
    #    self.calc_consts(init, final)
    #    for d in range(n_dims): #currently unidimensional process
    #        sol_d = np.linspace(self.st[d], self.en[d], n_pts) #get possible solution #in reality I should just initialize it to one of the successful demos and go from there, but that may only see local optimums
    #        is_moved = True #is not optimal
    #        while is_moved:
    #            is_moved = False
    #            for i in range(n_pts
    
    def calc_gaussians(self):
        if not self.gauss_calced:
            self.s_gauss = []
            self.f_gauss = []
            for d in range(self.n_dims):
                for i in range(self.n_pts):
                    #create a gaussian which will add to success or failure probabilities 
                    self.s_gauss.append([norm(loc=s_traj[i, d], scale=self.success_spread) for s_traj in self.s])
                    self.f_gauss.append([norm(loc=f_traj[i, d], scale=self.failure_spread) for f_traj in self.f])
                
            self.gauss_calced = True
            
    def query_gauss(self, dim, index, query):
        sum = 0.
        for s_g in self.s_gauss[(dim * self.n_pts) + index]:
            sum += s_g.pdf(query)
        for f_g in self.f_gauss[(dim * self.n_pts) + index]:
            sum -= f_g.pdf(query)
        
        #return sum([s_g.pdf(query) for s_g in self.s_gauss[(dim * self.n_pts) + index]]) - sum([f_g.pdf(query) for f_g in self.f_gauss][(dim * self.n_pts) + index])
        return sum
       
    def define_gmms(self):
        self.is_gmm = True
        self.t = np.linspace(0, 1, self.n_pts).reshape((self.n_pts, 1))
        if len(self.s) > 0:
            s_X = np.vstack(([np.hstack((self.t, self.s[ind])) for ind in range(len(self.s))]))
            self.s_gmm = GMM(n_components=self.n_components)
            self.s_gmm.from_samples(s_X)
            print(self.s_gmm.means)
            print(self.s_gmm.covariances)
        if len(self.f) > 0:
            f_X = np.vstack(([np.hstack((self.t, self.f[ind])) for ind in range(len(self.f))]))
            self.f_gmm = GMM(n_components=self.n_components)
            self.f_gmm.from_samples(f_X)
            print(self.f_gmm.means)
            print(self.f_gmm.covariances)
        
       
    #def gmr_success(self, traj):
    #    sum = 0.
    #    for i in range(self.n_pts):
    #        for n in range(self.n_components):
    #            if len(self.s) > 0:
    #                diff = np.abs(traj[i] - self.s_gmm.means[n])
    #                #sum -= np.linalg.norm(np.matmul(np.matmul(diff, self.s_gmm.covariances[n]), np.transpose(diff)))
    #                sum -= np.linalg.norm(diff)
    #            if len(self.f) > 0:
    #                diff = np.abs(traj[i] - self.f_gmm.means[n])
    #                #sum += np.linalg.norm(np.matmul(np.matmul(diff, self.f_gmm.covariances[n]), np.transpose(diff)))
    #                sum +=  0.5 * np.linalg.norm(diff)
    #    return sum
    
    def gmr_success(self, traj):
        sum = 0.
        for i in range(self.n_pts):
            for n in range(self.n_components):
                if len(self.s) > 0:
                    #diff = np.abs(traj[i] - self.s_gmm.means[n])
                    sum += np.linalg.norm(np.matmul(np.matmul(diff, self.s_gmm.covariances[n]), np.transpose(diff)))
                    #sum -= np.linalg.norm(diff)
                if len(self.f) > 0:
                    diff = np.abs(traj[i] - self.f_gmm.means[n])
                    sum -= np.sqrt(np.linalg.norm(np.matmul(np.matmul(diff, self.f_gmm.covariances[n]), np.transpose(diff))))
                    #sum +=  0.5 * np.linalg.norm(diff)
        return sum
        
    def gmr_success1(self, traj):
        sum = 0.
        if len(self.s) > 0:
            diff = np.abs(traj - self.s_gmr_traj)
            #print('cov')
            #print(self.s_gmm.covariances[0, 1:, 1:])
            sum += np.linalg.norm(np.matmul(diff, np.transpose(diff)))
        if len(self.f) > 0:
            diff = np.abs(traj - self.f_gmr_traj)
            sum -= np.sqrt(np.linalg.norm(np.matmul(diff, np.transpose(diff))))
        return sum
        
    def gmr_success_point(self, query):
        sum = 0.
        for i in range(self.n_pts):
            for n in range(self.n_components):
                if len(self.s) > 0:
                    diff = np.abs(traj[i] - self.s_gmm.means[n])
                    sum += np.linalg.norm(np.matmul(np.matmul(diff, self.s_gmm.covariances), np.transpose(diff)))
                    #sum -= np.linalg.norm(diff)
                if len(self.f) > 0:
                    diff = np.abs(traj[i] - self.f_gmm.means[n])
                    sum -= np.sqrt(np.linalg.norm(np.matmul(np.matmul(diff, self.f_gmm.covariances[n]), np.transpose(diff))))
                    #sum +=  0.5 * np.linalg.norm(diff)
        return sum
    
    def gmr_success_field(self, plot=False, save=False, fpath=''):
        self.calc_consts(init, final)
        self.define_gmms()
        return
    
    #def gmr_success(self, traj):
    #    sum = 0.
    #    theta = np.linspace(0, 2 * np.pi, self.n_pts).reshape((self.n_pts, 1))
    #    ## NOTE: THIS IS ONLY FOR 2D RIGHT NOW, 3D WILL COME LATER
    #    x = np.cos(theta)
    #    y = np.sin(theta)
    #    xy = np.hstack((x, y))
    #    for n in range(self.n_components):
    #        if len(self.s) > 0:
    #            for i in range(1, 4):
    #                ellipse = i * np.matmul(xy, self.s_gmm.covariances[n])
    #                dist_from = similaritymeasures.frechet_dist(ellipse, traj)
    #                sum += 1 / (i * (np.sqrt(dist_from) + 1e-3))
    #        if len(self.f) > 0:
    #            for i in range(1, 6):
    #                ellipse = i * np.matmul(xy, self.s_gmm.covariances[n])
    #                dist_from = similaritymeasures.frechet_dist(ellipse, traj)
    #                sum -= 1 / ((i**2) * (np.sqrt(dist_from) + 1e-3))
    #    return sum
    
    def opt_solver(self, X):
        #C = X.reshape(( len(X) // self.n_dims, self.n_dims ))
        #print(C)
        #traj = funx_approx.get_y_from_coeffs(self.t, C)
        traj = X.reshape(( self.n_pts, self.n_dims ))
        sum = 0
        #success probabilities
        for d in range(self.n_dims):
            for i in range(self.n_pts): 
                sum -= self.query_gauss(d, i, traj[i, d])
        
        #sum += self.gmr_success1(traj)
        
        #keep points close to each other using spring attractor
        for i in range(self.n_pts - 1): 
            sum += self.k2 * np.linalg.norm(traj[i] - traj[i + 1])**2
        #start/endpoint convergence using a strong attractor
        if self.st is not None:
            #print(traj[0] - self.st)
            #print(np.exp(np.linalg.norm(traj[0] - self.st)))
            sum += 100 * min([np.exp(np.linalg.norm(traj[0] - self.st)), 100])
        if self.en is not None:
            sum += 100 * min([np.exp(np.linalg.norm(traj[-1] - self.en)), 100])
        print(sum)
        return sum
        
    def get_opt_traj(self, init=None, final=None, plot=False, save=False, fpath=''):
        self.calc_consts(init, final)
        self.calc_gaussians()
        self.define_gmms()
        self.s_gmr_traj = self.s_gmm.predict(np.array([0]), self.t)
        self.f_gmr_traj = self.f_gmm.predict(np.array([0]), self.t)
        #self.t = np.linspace(0, self.n_pts - 1, self.n_pts)
        init_guess = self.s[np.random.randint(len(self.s))].flatten() if self.s else self.f[np.random.randint(len(self.f))].flatten()
        res = minimize(self.opt_solver, init_guess, tol=1e-6, options={'disp': True})
        #best_coeffs = np.reshape(res.x, (( len(res.x) // self.n_dims, self.n_dims )) )
        #print(best_coeffs)
        #best_traj = funx_approx.get_y_from_coeffs(self.t, best_coeffs)
        best_traj = np.reshape(res.x, (( self.n_pts, self.n_dims )))
        print(best_traj)
        if (plot):
            plt.figure()
            for s_traj in self.s:
                plt.plot(s_traj[:, 0], s_traj[:, 1], 'g', lw=3)
            for f_traj in self.f:
                plt.plot(f_traj[:, 0], f_traj[:, 1], 'r', lw=3)
            plt.plot(best_traj[:, 0], best_traj[:, 1], 'b', lw=5)
            plt.plot(self.st[0], self.st[1], 'ko', ms=10)
            plt.plot(self.en[0], self.en[1], 'kx', ms=10, mew=5)
            #if (self.is_gmm):
            #    plot_error_ellipses(plt.gca(), self.s_gmm, colors=["g"])
            #    plot_error_ellipses(plt.gca(), self.f_gmm, colors=["r"])
            
            title = 'highest success traj k2=' + str(self.k2)
            plt.title(title)
            if (save):
                plt.savefig(fpath + title + '.png')
            else:
                plt.show()
            plt.close('all')
        return best_traj
    
    
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
    
def main_lasa():
    lasa_trajs = []
    for i in range(2):
        x, y = get_lasa_trajN('RShape', i + 1)
        traj1 = np.hstack((np.reshape(x, (len(x), 1)), np.reshape(y, (len(y), 1))))
        traj = dp.DouglasPeuckerPoints(traj1, 50) + np.random.normal(0, 1, 100).reshape((50, 2))
        lasa_trajs.append(traj)
        
    #k_vals = np.linspace(5, 7, 100)
    plt_fpath = '../pictures/failure_lfd/noise_test/'
    mn = 0.
    mx = 10.
    while True:
        my_k = (mx + mn) / 2
        lffd = LFFD(successes=lasa_trajs, failures=[], k=my_k)
        succ_prbs = lffd.calculate_prb_success(plot=True, save=True, fpath=plt_fpath)
        #plt.figure()
        #ax = sns.heatmap(succ_prbs[0])
        #plt.title('prbs[0]')
        #plt.figure()
        #ax = sns.heatmap(succ_prbs[1])
        #plt.title('prbs[1]')
        #plt.show()
        
        best_traj = lffd.get_opt_traj(init=traj[0, :], final=traj[-1, :], plot=True, save=True, fpath=plt_fpath)
        #plt.figure()
        #plt.plot(best_traj[:, 0], best_traj[:, 1], 'b', lw=3)
        #for i in range(1):
        #    plt.plot(lasa_trajs[i][:, 0], lasa_trajs[i][:, 1], 'g')
        #plt.title('LASA Successful')
        #plt.show()
        ch = input('h/l?')
        if (ch == 'h'):
            mn = my_k
        else:
            mx = my_k
    
def main_all_lasa():
    lasa_names = ['Angle','BendedLine','CShape','DoubleBendedLine','GShape', \
                  'heee','JShape','JShape_2','Khamesh','Leaf_1', \
                  'Leaf_2','Line','LShape','NShape','PShape', \
                  'RShape','Saeghe','Sharpc','Sine','Snake', \
                  'Spoon','Sshape','Trapezoid','Worm','WShape', \
                  'Zshape','Multi_Models_1','Multi_Models_2','Multi_Models_3','Multi_Models_4']
    for shape in lasa_names:
        plt_fpath = '../pictures/lffd_lasa/' + shape + '/'
        try:
            os.makedirs(plt_fpath)
        except OSError:
            print ("Creation of the directory %s failed" % plt_fpath)
        else:
            print ("Successfully created the directory %s" % plt_fpath)
        lasa_trajs = []
        for i in range(7):
            x, y = get_lasa_trajN(shape, i + 1)
            traj = np.hstack((np.reshape(x, (len(x), 1)), np.reshape(y, (len(y), 1))))
            traj = dp.DouglasPeuckerPoints(traj, 50)
            lasa_trajs.append(traj)
        lffd = LFFD(successes=lasa_trajs, failures=[])
        #lffd.calculate_prb_success(plot=True, save=True, fpath=plt_fpath)
        lffd.get_opt_traj(init=traj[0, :], final=traj[-1, :], plot=True, save=True, fpath=plt_fpath)
        
    
def main2():
    x = np.linspace(0, 1).reshape((50, 1))
    ys1 = x - 0.5
    ys2 = np.abs(ys1)
    s = []
    f = []
    for _ in range(3):
        noise = np.random.normal(0, 0.1, 50).reshape((50, 1))
        s_traj = ys1 + noise
        s_traj = np.hstack((x, s_traj))
        s.append(s_traj)
        f_traj = ys2 - noise
        f_traj = np.hstack((x, f_traj))
        f.append(f_traj)
    
    plt_fpath = '../pictures/failure_lfd/gmr_testing2/'
    
    my_k = 0.
    while True:
        my_k = float(input('Enter a new k, last k = ' + str(my_k)))
        lffd = LFFD(successes=s, failures=f, k=my_k)
        #succ_prbs = lffd.calculate_prb_success(plot=True, save=True, fpath=plt_fpath)
    
        best_traj = lffd.get_opt_traj(init=np.array([0, 1.25]), final=np.array([1, 0.5]), plot=True, save=True, fpath=plt_fpath)
        
        
def main():
    t = np.linspace(0, 1).reshape((50, 1))
    s1 = (t - 0.5)**2 - 0.25
    s2 = -s1
    s3 = -4 * (t - 0.5)**4 + 0.25
    f1 = 0.125 * np.sin(2 * np.pi * t)
    f2 = 0. * t
    
    start = [0, 0]
    end = [1, 0]
    obstacle = [0.5, 0]
    
    s1 = np.hstack((t, s1))
    s2 = np.hstack((t, s2))
    s3 = np.hstack((t, s3))
    f1 = np.hstack((t, f1))
    f2 = np.hstack((t, f2))
    
    plt_fpath = '../pictures/failure_lfd/gmr_testing/'
    
    mn = 0.
    mx = 100.
    my_k = 0.
    while True:
        #my_k = (mx + mn) / 2
        my_k = float(input('Enter a new k, last k = ' + str(my_k)))
        lffd = LFFD(successes=[s1, s2, s3], failures=[f1, f2], k=my_k)
        #succ_prbs = lffd.calculate_prb_success(plot=True, save=True, fpath=plt_fpath)
    
        best_traj = lffd.get_opt_traj(init=np.array([0, 0.25]), final=np.array([1, 0]), plot=True, save=True, fpath=plt_fpath)
        #ch = input('h/l?')
        #if (ch == 'h'):
        #    mn = my_k
        #else:
        #    mx = my_k
    
    #plt.figure()
    ###plt.plot(t, best_traj, 'b')
    #plt.plot(t, s1, 'g', lw=3)
    #plt.plot(t, s2, 'g', lw=3)
    #plt.plot(t, s3, 'g', lw=3)
    #plt.plot(t, f1, 'r', lw=3)
    #plt.plot(t, f2, 'r', lw=3)
    #new_traj = np.zeros(np.shape(s1))
    #for i in range(len(t)):
    #    new_traj[i] = (s1[i] + s2[i] + s3[i]) / 3.
    #plt.plot(t, new_traj, 'c', lw=5)
    #plt.plot(start[0], start[1], 'ko', ms=10)
    #plt.plot(end[0], end[1], 'kx', ms=10, mew=5)
    ##
    ###plt.figure()
    ###ax = sns.heatmap(succ_prbs[0])
    ###plt.plot(start[0], start[1], 'ko')
    ###plt.plot(end[0], end[1], 'kx')
    ###plt.plot(t, s1)
    ###plt.plot(t, s2)
    ###plt.plot(t, f1)
    #plt.show()

if __name__ == "__main__":
    main2()
