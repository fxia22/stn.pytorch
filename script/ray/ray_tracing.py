import numpy as np
def trace(theta, phi, r_np):
    occupancy_min_r = np.ones(theta.shape) * 10000
    occupancy_input = np.zeros(theta.shape)
    occupancy = np.zeros(theta.shape)
    threshold = 0.2
    b, h, w  = theta.shape
    for i in range(b):
        for j in range(h):
            for k in range(w):
                idx1 = int((theta[i,j,k] + 1)/(2/float(h)))
                idx2 = int((phi[i,j,k] + 1)/(2/float(w)))
                if r_np[i,j,k] < occupancy_min_r[i,idx1,idx2]:
                    occupancy_min_r[i,idx1,idx2] = r_np[i,j,k]
                    occupancy[i,idx1-1:idx1+1,idx2-1:idx2+1] = 1


    for i in range(b):
        for j in range(h):
            for k in range(w):
                idx1 = int((theta[i,j,k] + 1)/(2/float(h)))
                idx2 = int((phi[i,j,k] + 1)/(2/float(w)))
                if r_np[i,j,k] > occupancy_min_r[i,idx1,idx2]  + threshold:
                    occupancy_input[i,j,k] = 1

    return occupancy, occupancy_input
