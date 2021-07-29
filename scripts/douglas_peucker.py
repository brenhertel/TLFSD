import numpy as np
import matplotlib.pyplot as plt
'''
based on the following psuedocode from wikipedia: https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
function DouglasPeucker(PointList[], epsilon)
    // Find the point with the maximum distance
    dmax = 0
    index = 0
    end = length(PointList)
    for i = 2 to (end - 1) {
        d = perpendicularDistance(PointList[i], Line(PointList[1], PointList[end])) 
        if (d > dmax) {
            index = i
            dmax = d
        }
    }
    
    ResultList[] = empty;
    
    // If max distance is greater than epsilon, recursively simplify
    if (dmax > epsilon) {
        // Recursive call
        recResults1[] = DouglasPeucker(PointList[1...index], epsilon)
        recResults2[] = DouglasPeucker(PointList[index...end], epsilon)

        // Build the result list
        ResultList[] = {recResults1[1...length(recResults1) - 1], recResults2[1...length(recResults2)]}
    } else {
        ResultList[] = {PointList[1], PointList[end]}
    }
    // Return the result
    return ResultList[]
end
'''

def perpendicularDistance(pp, p1, p2):
    #find distance from pp to line p1p2
    # vector formulation from: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    n = p2 - p1
    n = n / np.linalg.norm(n)
    return np.linalg.norm( (p1 - pp) - (np.dot((p1 - pp), n) * n) )

def DouglasPeucker(PointList, epsilon):
    # Find the point with the maximum distance
    dmax = 0
    index = 0
    (n_pts, n_dims) = np.shape(PointList)
    for i in range(1, n_pts):
        d = perpendicularDistance(PointList[i], PointList[0], PointList[n_pts - 1]) 
        if (d > dmax):
            index = i
            dmax = d
            
    # If max distance is greater than epsilon, recursively simplify
    if (dmax > epsilon):
        # Recursive call
        recResults1 = DouglasPeucker(PointList[0:index], epsilon)
        recResults2 = DouglasPeucker(PointList[index - 1:], epsilon)

        # Build the result list
        ResultList = np.vstack((recResults1, recResults2))
    else:
        ResultList = np.vstack((PointList[0], PointList[n_pts - 1]))
    # Return the result
    return ResultList
    
def DouglasPeuckerIterative(PointList, epsilon):
    (n_pts, n_dims) = np.shape(PointList)
    above_eps = False
    ResultList = np.vstack((PointList[0], PointList[n_pts-1]))
    inds = [0, n_pts-1]
    while not above_eps:
        above_eps = True
        # Find the point with the maximum distance for each segment
        for seg in range(len(inds) - 1):
            dmax = 0
            index = 0
            for i in range(inds[seg], inds[seg+1]):
                #print([i, index])
                d = perpendicularDistance(PointList[i], ResultList[seg], ResultList[seg + 1]) 
                if (d > dmax):
                    index = i - 1 #this is to fix some indexing error
                    dmax = d
            if (dmax > epsilon):
                above_eps = False
                #ResultList.insert(PointList[index, :].copy(), seg + 1)
                ResultList = np.insert(ResultList, seg + 1, PointList[index, :], axis=0)
                inds.insert(seg + 1, index)
    # Return the result
    return ResultList

def DouglasPeuckerPoints(PointList, num_points):
    (n_pts, n_dims) = np.shape(PointList)
    ResultList = np.vstack((PointList[0], PointList[n_pts-1]))
    inds = [0, n_pts-1]
    while len(inds) < num_points:
        dmax = 0
        index = 0
        segnum = 0
        for seg in range(len(inds) - 1):
            for i in range(inds[seg], inds[seg+1]):
                d = perpendicularDistance(PointList[i], ResultList[seg], ResultList[seg + 1]) 
                if (d > dmax):
                    index = i
                    dmax = d
                    segnum = seg
        ResultList = np.insert(ResultList, segnum + 1, PointList[index, :], axis=0)
        inds.insert(segnum + 1, index)
    # Return the result
    return ResultList
    
if __name__ == '__main__':
    #in-file testing
    x = np.linspace(0, 10)
    x = np.reshape(x, (len(x), 1))
    y = 6 * np.sin(x)
    traj = np.hstack((x, y))
    
    #for eps in [10, 5, 2.5, 1.5, 1, 0.5, 0.25]:
    #    fig = plt.figure()
    #    plt.title('epsilon = ' + str(eps))
    #    plt.plot(traj[:, 0], traj[:, 1], 'k')
    #    dp_traj = DouglasPeucker(traj, eps)
    #    dpi_traj = DouglasPeuckerIterative(traj, eps)
    #    plt.plot(dp_traj[:, 0], dp_traj[:, 1], 'r')
    #    plt.plot(dpi_traj[:, 0], dpi_traj[:, 1], 'g')
    #    plt.show()
    for pts in [10, 20, 30, 40]:
        fig = plt.figure()
        plt.title('num_points = ' + str(pts))
        plt.plot(traj[:, 0], traj[:, 1], 'k')
        dp_traj = DouglasPeuckerPoints(traj, pts)
        print(dp_traj)
        plt.plot(dp_traj[:, 0], dp_traj[:, 1], 'r')
        plt.show()
        