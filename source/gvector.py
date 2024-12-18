import numpy as np

# Reciprocal grid point

def mm_to_nn(i,n):
    if i > n // 2:
        i=i-n
    return(i)


class G_vector(object):
    def __init__(self, rec_latt, wf_cutoff, grid_point):
        #grid_point = np.array([5, 5, 5])
        self.n_gw          = 0
        self.n_point       = np.prod(grid_point)
        self.g_vector_mask = np.zeros(grid_point)
        self.g2_vector     = np.zeros(grid_point) # i, j, k list
        self.g2_vector[:,:,:]     = -1
        self.g1_vector     = np.zeros((grid_point[0],grid_point[1],grid_point[2],3))
 
        # Get valid g points within the cutoff
        for i in range(grid_point[0]):
            for j in range(grid_point[1]):
                for k in range(grid_point[2]):
                    ii,jj,kk=mm_to_nn(i, grid_point[0]), mm_to_nn(j,grid_point[1]), mm_to_nn(k, grid_point[2])
                    temp=ii*rec_latt[0] + jj*rec_latt[1] + kk*rec_latt[2]
                    temp1=np.sum(temp**2)                    
                    if temp1 <= 8. * wf_cutoff:
                        self.g_vector_mask[i,j,k] = 1
                        self.n_gw += 1
                        self.g2_vector[i,j,k] = temp1
                        self.g1_vector[i,j,k] = temp
                        
        self.g_vector_mask =  self.g_vector_mask == 1
        self.gx_vector_mask = np.logical_and( self.g2_vector <= 2.* wf_cutoff,  self.g2_vector >= 0.)
        self.n_gxw          = np.sum(self.gx_vector_mask)
        self.g2_vector[self.g2_vector == -1 ] = 0
        #print(self.g_vector_mask); import sys; sys.exit()   
