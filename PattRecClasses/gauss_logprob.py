'''
logP = gauss_logprob(pDs,x)
method to give the probability of a data sequence,
assumed to be drawn from given Gaussian Distribution(s).

Input:
pD=    GaussD object or array of GaussD objects
x=     row vector with data assumed to be drawn from a Gaussian Distribution

Result:
logP=  array with log-probability values for each element in x,
       for each given GaussD object
       size(p)== [length(pDs),size(x,2)], if pDs is one-dimensional vector
       size(p)== [size(pDs),size(x,2)], if pDs is multidim array
'''
import numpy as np
def gauss_logprob(pDs, x):
    nObj = len(pDs) # Number of GaussD Objects
    nx = x.shape[1] # Number of observed vectors
    logP = np.zeros((nObj, nx))

    for i, pD in enumerate(pDs):
        dSize = pD.dataSize
        assert dSize == x.shape[0]

        z = np.dot(pD.covEigen, (x-np.matlib.repmat(pD.means, 1, nx)))

        z /= np.matlib.repmat(np.expand_dims(pD.stdevs, 1), 1, nx)

        logP[i, :] = -np.sum(z*z, axis=0)/2 
        logP[i, :] = logP[i, :] - sum(np.log(pD.stdevs)) - dSize*np.log(2*np.pi)/2

    return logP


## TESTCASES

# One-dimensional Gaussian
#g1 = GaussD( means=np.array( [0] ) , stdevs=np.array( [1.0] ) )
#obs = np.array([-0.2, 2.6, 1.3])
#gauss_prob([g1], obs)

# Output: array([[-1.17657582]])

# Multi-dimensional Gaussian
#g1 = GaussD( means=np.array( [[0.0], [1.0]] ) , cov=np.array( [[1.0, 0.0], [0.0, 1.0]] ) )
#gauss_prob([g1], g1.rand(3))

# Output: array([[-1.86376038, -2.74915905, -2.73003471]])

# Mismatched Parameters
#g2 = GaussD( means=np.array( [[3.0], [0.0]] ) , cov=np.array( [[1.0, 0.5], [0.5, 1.0]] ) )
#gauss_prob([g2], g1.rand(3))

# Output: array([[-11.66457869,  -3.23578104,  -4.46681203]])