# quadratic form test

import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'C:/Users/BH/Documents/GitHub/pearl_test_env/Guassian-Mixture-Models')
from GMM_GMR import GMM_GMR
import douglas_peucker as dp

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
    
s = []
num_demos = 3
num_states = 3
n_pts = 50
n_dims = 2
for n in range(num_demos):
    [x, y] = get_lasa_trajN('Angle', n + 1)
    data = np.hstack((np.reshape(x, (len(x), 1)), np.reshape(y, (len(y), 1))))
    
    traj = dp.DouglasPeuckerPoints(data, n_pts)
    s.append(np.transpose(traj))
        
t = np.linspace(0, 1, n_pts).reshape((1, n_pts))
for s_ind in range(num_demos):
    s[s_ind] = np.vstack((t, s[s_ind]))

s_gmm = GMM_GMR(num_states)
s_gmm.fit(np.hstack(s))
s_gmm.predict(t)
mu_s = s_gmm.getPredictedData()
cov_s = s_gmm.getPredictedSigma()
inv_cov_s_pt = np.zeros((np.shape(cov_s)))
inv_cov_s_all = np.zeros((n_pts * n_dims, n_pts * n_dims))
for i in range(n_pts):
    inv_cov_s_pt[:, :, i] = np.linalg.inv(cov_s[:, :, i])
    inv_cov_s_all[(i * n_dims) : (((i + 1) * n_dims)), (i * n_dims) : (((i + 1) * n_dims))] = cov_s[:, :, i]
inv_cov_s_all = np.linalg.inv(inv_cov_s_all)

[x, y] = get_lasa_trajN('Angle', 5)
data = np.hstack((np.reshape(x, (len(x), 1)), np.reshape(y, (len(y), 1))))

traj = dp.DouglasPeuckerPoints(data, n_pts)
cmp = np.transpose(traj)

mu = mu_s[1:, :]

cmp_flat = cmp.flatten()
mu_flat = mu.flatten()

diff = np.subtract(mu_flat, cmp_flat)
sum_all = np.matmul(diff, np.matmul(inv_cov_s_all, np.transpose(diff)))
print(['Sum all', sum_all])

sum_ind = 0.
for i in range(n_pts):
    diff_i = mu[:, i] - cmp[:, i]
    sum_i = np.matmul(np.transpose(diff_i), np.matmul(inv_cov_s_pt[:, :, i], diff_i))
    sum_ind += sum_i
print(['Sum ind', sum_ind])
