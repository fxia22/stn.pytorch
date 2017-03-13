import numpy as np

cimport numpy as np

DTYPE = np.float32

ctypedef np.float32_t DTYPE_t


def trace(np.ndarray[DTYPE_t, ndim=3] theta, np.ndarray[DTYPE_t, ndim=3] phi, np.ndarray[DTYPE_t, ndim=3] r_np):
    #assert theta.dtype == DTYPE and phi.dtype == DTYPE and r_np.dtype == DTYPE
    cdef int b = theta.shape[0]
    cdef int h = theta.shape[1]
    cdef int w = theta.shape[2]
    
    
    cdef np.ndarray[DTYPE_t, ndim=3] occupancy_min_r = (np.ones([b,h,w], dtype = DTYPE) * 1e4)
    cdef np.ndarray[DTYPE_t, ndim=3] occupancy_input = np.ones([b,h,w], dtype = DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] occupancy = np.zeros([b,h,w], dtype = DTYPE)
    cdef DTYPE_t threshold = 0.1
    
    
    cdef int i,j,k,idx1,idx2
    
    cdef DTYPE_t h_step = 2.0/float(h)
    cdef DTYPE_t w_step = 2.0/float(w)
    
    cdef int padding = 1
    cdef int pad_x, pad_y
    
    cdef DTYPE_t local_min;

    for i in range(b):
        for j in range(h):
            for k in range(w):
                idx1 = int((theta[i,j,k] + 1)/(h_step))
                idx2 = int((phi[i,j,k] + 1)/(w_step))
                #print r_np[i,j,k], occupancy_min_r[i,idx1,idx2]
                
                if padding < idx1 < h-padding and padding < idx2 < w-padding:
   
                   
                    for pad_x in range(-padding, padding):
                        for pad_y in range(-padding, padding):
                            occupancy[i,idx1+pad_x,idx2+pad_y] = 1
                        
                            if r_np[i,j,k] < occupancy_min_r[i,idx1+pad_x,idx2+pad_y]:
                                occupancy_min_r[i,idx1+pad_x,idx2+pad_y] = r_np[i,j,k]
                            

    #print occupancy_min_r
    
    for i in range(b):
        for j in range(h):
            for k in range(w):
                idx1 = int((theta[i,j,k] + 1)/(h_step))
                idx2 = int((phi[i,j,k] + 1)/(w_step))
                if padding < idx1 < h-padding and padding < idx2 < w-padding:
                    if r_np[i,j,k] > occupancy_min_r[i,idx1,idx2]  + threshold:
                        occupancy_input[i,j,k] = 0

    return occupancy, occupancy_input
