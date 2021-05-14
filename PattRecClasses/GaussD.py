from typing import List

import numpy as np
import numpy.matlib
from collections import Counter
import scipy.stats as stats
import math


class GaussD:
    """
    GaussD - Probability distribution class, representing
    Gaussian random vector
    EITHER with statistically independent components,
               i.e. diagonal covariance matrix, with zero correlations,
    OR with a full covariance matrix, including correlations
    -----------------------------------------------------------------------

    Several GaussD objects may be collected in a multidimensional array,
               even if they do not have the same DataSize.
    """

    def __init__(self, means, stdevs=None, cov=None):

        self.means = np.array(means)
        self.stdevs = np.array(stdevs)
        self.dataSize = len(self.means)  # number of gaussian = number of means for each gaussian
        try:
            self.nDims = len(self.means[0])
        except TypeError:
            self.nDims = self.dataSize

        if cov is None:
            self.variance = self.stdevs ** 2
            self.cov = np.eye(self.dataSize) * self.variance  # matrix with variance values in the diagonal
            self.covEigen = 1
        else:
            self.cov = cov
            v, self.covEigen = np.linalg.eig(0.5 * (cov + cov.T))  # eigenvalues and eigenvectors of the matrix
            self.stdevs = np.sqrt(np.abs(v))
            self.variance = self.stdevs ** 2

    def rand(self, nData):
        """
        R=rand(pD,nData) returns random vectors drawn from a single GaussD object.

        Input:
        pD=    the GaussD object
        nData= scalar defining number of wanted random data vectors

        Result:
        R= matrix with data vectors drawn from object pD
           size(R)== [length(pD.Mean), nData]
        """
        R = np.random.randn(self.dataSize,
                            nData)  # random matrix with size num of gaussian distr * number of random vector
        R = np.diag(self.stdevs) @ R  # diagonal from an array?? * random matrix R

        if not isinstance(self.covEigen, int):  # IF THERE IS NOT CORRELATION???
            R = self.covEigen @ R  # multipliquem 1 per R??

        R = R + np.matlib.repmat(self.means.reshape(-1, 1), 1,
                                 nData)  # repeteix l'array mean en forma de columnes ndata times

        return R

    def init(self):
        pass

    def prob(self, obs: np.ndarray):
        obs_len = len(obs)
        pb = np.zeros(obs_len)

        for j in range(obs_len):
            pb[j] = stats.norm.pdf(obs[j], self.means[0], self.stdevs[0])

        return pb

    def logprob(self):
        pass

    def plotCross(self):
        pass

    def adaptStart(self):
        pass

    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass