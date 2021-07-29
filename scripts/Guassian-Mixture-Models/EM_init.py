import numpy as np

def kMeans(X, K, maxIters = 30):
    centroids = X[np.random.choice(np.arange(len(X)), K), :]
    for i in range(maxIters):
        C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in X])
        centroids = [X[C == k].mean(axis = 0) for k in range(K)]
    return np.array(centroids) , C
    
    
def my_kMeans(X, K, w, maxIters = 30):
    centroids = X[np.random.choice(np.arange(len(X)), K), :]
    for i in range(maxIters):
        C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in X])
        for k in range(K):
            sum = None
            count = None
            for ind in range(len(X)):
                if (C[ind] == k):
                    sum = sum + w[ind] * X[ind] if sum is not None else w[ind] * X[ind]
                    count = count + w[ind] if count is not None else w[ind]
            centroids[k] = sum / count if sum is not None else X[np.random.choice(np.arange(len(X)), 1), :]
    return np.array(centroids) , C

def EM_init(Data, nbStates):
    nbVar, nbData = np.shape(Data)
    Priors = np.ndarray(shape = (1, nbStates))
    Sigma = np.ndarray(shape = (nbVar, nbVar, nbStates))
    Centers, Data_id = kMeans(np.transpose(Data), nbStates)
    Mu = np.transpose(Centers)
    for i in range (0,nbStates):
        idtmp = np.nonzero(Data_id==i)
        idtmp = list(idtmp)
        idtmp = np.reshape(idtmp,(np.size(idtmp)))
        Priors[0,i] = np.size(idtmp)
        a = np.concatenate((Data[:, idtmp],Data[:, idtmp]), axis = 1)
        Sigma[:,:,i] = np.cov(a)
        Sigma[:,:,i] = Sigma[:,:,i] + 0.00001 * np.diag(np.diag(np.ones((nbVar,nbVar))))
    Priors = Priors / nbData
    return (Priors, Mu, Sigma)
    
    
def weighted_k_means(Data, nbStates, w):
    nbVar, nbData = np.shape(Data)
    Priors = np.ndarray(shape = (1, nbStates))
    Sigma = np.ndarray(shape = (nbVar, nbVar, nbStates))
    Centers, Data_id = my_kMeans(np.transpose(Data), nbStates, w)
    Mu = np.transpose(Centers)
    for i in range (0,nbStates):
        idtmp = np.nonzero(Data_id==i)
        idtmp = list(idtmp)
        idtmp = np.reshape(idtmp,(np.size(idtmp)))
        Priors[0,i] = np.size(idtmp)
        a = np.concatenate((Data[:, idtmp],Data[:, idtmp]), axis = 1)
        Sigma[:,:,i] = np.cov(a)
        Sigma[:,:,i] = Sigma[:,:,i] + 0.00001 * np.diag(np.diag(np.ones((nbVar,nbVar))))
    Priors = Priors / nbData
    return (Priors, Mu, Sigma)