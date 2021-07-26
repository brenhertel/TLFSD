from GMM_GMR import GMM_GMR
from matplotlib import pyplot as plt
import numpy as np

#if __name__ == "__main__":
#    data = np.loadtxt("data.txt", delimiter=',')
#    data = data[:, 0:2].T
#    gmr = GMM_GMR(4)
#    gmr.fit(data)
#    timeInput = np.linspace(1, np.max(data[0, :]), 100)
#    gmr.predict(timeInput)
#    fig = plt.figure()
#
#    ax1 = fig.add_subplot(221)
#    print(type(ax1))
#    plt.title("Data")
#    gmr.plot(ax=ax1, plotType="Data")
#
#    ax2 = fig.add_subplot(222)
#    plt.title("Gaussian States")
#    gmr.plot(ax=ax2, plotType="Clusters")
#
#    ax3 = fig.add_subplot(223)
#    plt.title("Regression")
#    gmr.plot(ax=ax3, plotType="Regression")
#
#    ax4 = fig.add_subplot(224)
#    plt.title("Clusters + Regression")
#    gmr.plot(ax=ax4, plotType="Clusters")
#    gmr.plot(ax=ax4, plotType="Regression")
#    predictedMatrix = gmr.getPredictedMatrix()
#    print(predictedMatrix)
#    plt.show()
def get_lasa_trajN(shape_name, n=1):
    import h5py
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
    
def main2():
    import sys
    # insert at 1, 0 is the script path (or '' in REPL)
    sys.path.insert(1, 'C:/Users/BH/Documents/GitHub/pearl_test_env/scripts')
    import douglas_peucker as dp
    
    n = 20
    
    t = np.arange(n).reshape((n, 1))
    fig = plt.figure()
    lasa_trajs = []
    for i in range(7):
        x, y = get_lasa_trajN('RShape', i + 1)
        plt.plot(x, y, 'b')
        traj1 = np.hstack((np.reshape(x, (len(x), 1)), np.reshape(y, (len(y), 1))))
        traj = dp.DouglasPeuckerPoints(traj1, n)
        traj2 = np.hstack((t, traj))
        lasa_trajs.append(traj2)
    data = np.transpose(np.vstack((lasa_trajs)))
    #print(data)
    my_gmm = GMM_GMR(4)
    my_gmm.fit(data)
    #means, labels = my_gmm.k_means(Data)
    #means, covariances = my_gmm.weighted_k_means(data)
    timeInput = np.linspace(min(t), max(t), 100).reshape((100))
    my_gmm.predict(timeInput)
    prediction = my_gmm.getPredictedMatrix()
    plt.plot(prediction[1, :], prediction[2, :], 'k')
    print(prediction)
    plt.show()
    
if __name__ == '__main__':
    main2()