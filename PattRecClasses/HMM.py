from typing import List

import numpy as np
import scipy.stats

"""
Full Hidden Markov Model Class Implementation
"""


class HMM:

    """
    Constructor
    """

    def __init__(self, q, A, B, finite=False):
        self.A = A
        self.B = B
        self.q = q
        self.finite = finite

    """
    Generate Observations from a HMM
    """

    def rand(self, num):
        res = (np.zeros((num, self.B[0].random(1).ndim)), np.zeros((num), dtype=int))
        temp = np.random.choice(range(self.q.shape[0]), 1, p=self.q)[0]
        res[1][0] = temp
        res[0][0] = self.B[temp].rand()
        num -= 1
        if num == 0:
            return res
        for i in range(num):
            temp = np.random.choice(range(self.A.shape[0]), 1, p=self.A[res[1][i]])[0]
            res[1][i + 1] = temp
            dist = self.B[temp]
            res[0][i + 1] = self.B[temp].rand()
        return res

    """
    Calculate alpha_hat values for the forward algorithm
    """

    def alphahat(self, obs, scale=True):
        if self.finite:
            A = self.A[:, :-1]
        else:
            A = self.A
        p, scaled = prob(obs, self.B)
        if not scale:
            scaled = p
        c = np.zeros(obs.shape[0])
        alpha = np.zeros((obs.shape[0], A.shape[0]))
        temp = np.zeros(A.shape[0])
        c[0] = np.sum(self.q * scaled[0])
        alpha[0, :] = (self.q * scaled[0]) / c[0]

        for t in range(1, obs.shape[0]):
            for j in range(A.shape[0]):
                temp[j] = alpha[t - 1].dot(A[:, j]) * scaled[t, j]
            c[t] = np.sum(temp)
            alpha[t, :] = temp / c[t]

        if self.finite:
            variable = alpha[-1].dot(self.A[:, -1])
            c = np.append(c, np.array([variable]))
        return alpha, c

    """
    Calculate beta_hat values for the backward algorithm
    """

    def betahat(self, obs, scale=True):
        if self.finite:
            A = self.A[:, :-1]
        else:
            A = self.A
        p, scaled = prob(obs, self.B)
        if not scale:
            scaled = p
        alphas, cs = self.alphahat(obs)
        beta = np.zeros((obs.shape[0], self.A.shape[0]))
        temp = np.zeros(self.A.shape[0])
        if self.finite:
            temp = self.A[:, -1]
            temp = temp / (cs[-1] * cs[-2])
        else:
            temp = np.ones((self.A.shape[0]))
            temp = temp / cs[-1]
        beta[-1] = temp

        for t in range(obs.shape[0] - 2, -1, -1):
            temp = np.zeros(A.shape[0])
            for i in range(A.shape[0]):
                for j in range(A.shape[0]):
                    temp[i] += A[i, j] * scaled[t + 1, j] * beta[t + 1, j]
            beta[t] = temp / cs[t]
        return beta

    """
    Viterbi algorithm for maximum likelihood sequence detection
    """

    def viterbi(self, obs, scale=True):
        if self.finite:
            A = self.A[:, :-1]
        else:
            A = self.A
        chi = np.zeros((obs.shape[0], A.shape[0]))
        prev = np.zeros((obs.shape[0] - 1, A.shape[0]), dtype=int)
        p, scaled = logprob(obs, self.B)
        if not scale:
            scaled = p
        chi[0, :] = np.log(self.q) + scaled[0]

        for t in range(1, obs.shape[0]):
            for j in range(A.shape[0]):
                proba = chi[t - 1] + np.log(A[:, j])
                prev[t - 1, j] = np.argmax(proba)
                chi[t, j] = np.amax(proba + scaled[t, j])

        if self.finite:
            chi[-1] += np.log(self.A[:, -1])
        ihat = np.zeros(obs.shape[0], dtype=int)
        last = np.argmax(chi[-1, :])
        ihat[0] = last

        index = 1
        for i in range(obs.shape[0] - 2, -1, -1):
            temp = prev[i, int(last)]
            ihat[index] = temp
            last = temp
            index += 1
        ihat = np.flip(ihat, axis=0)
        return ihat

    # page 113
    def calcgammas(self, alphahats, betahats, cs, obs, uselog=False):
        gammas = []
        for i in range(obs.shape[0]):
            temp = []
            for t in range(obs[i].shape[0]):
                if uselog:
                    temp += [
                        np.log(alphahats[i][t])
                        + np.log(betahats[i][t])
                        + np.log(cs[i][t])
                    ]
                else:
                    temp += [alphahats[i][t] * betahats[i][t] * cs[i][t]]
            gammas += [np.array(temp)]
        gammas = np.array(gammas)
        return gammas

    # page 130
    def calcinit(self, gammas, uselog=False):
        if uselog:
            return np.sum(np.exp(gammas[:, 0]), axis=0) / np.sum(np.exp(gammas[:, 0]))
        else:
            return np.sum(gammas[:, 0], axis=0) / np.sum(gammas[:, 0])

    def calcabc(self, obs):
        alphahats = []
        betahats = []
        cs = []
        for i in range(obs.shape[0]):
            alph, c = self.alphahat(obs[i])
            beth = self.betahat(obs[i])
            alphahats += [alph]
            betahats += [beth]
            cs += [c]
        return alphahats, betahats, cs

    # page 132
    def calcxi(self, alphahats, betahats, cs, obs, uselog=False):
        xirbars = []
        xirs = []
        for i in range(obs.shape[0]):
            if self.finite:
                xi = np.zeros((obs[i].shape[0], self.A.shape[0], self.A.shape[1]))
            else:
                xi = np.zeros((obs[i].shape[0] - 1, self.A.shape[0], self.A.shape[1]))
            p, scaled = prob(obs[i], self.B)
            if uselog:
                xi = np.log(xi)
                p, scaled = logprob(obs[i], self.B)
            for t in range(obs[i].shape[0] - 1):
                for j in range(self.A.shape[0]):
                    for k in range(self.A.shape[0]):
                        if uselog:
                            xi[t, j, k] = (
                                np.log(alphahats[i][t][j])
                                + np.log(self.A[j, k])
                                + scaled[t + 1][k]
                                + np.log(betahats[i][t + 1][k])
                            )
                        else:
                            xi[t, j, k] = (
                                alphahats[i][t][j]
                                * self.A[j, k]
                                * scaled[t + 1][k]
                                * betahats[i][t + 1][k]
                            )
            if self.finite:
                for j in range(self.A.shape[0]):
                    if uselog:
                        xi[-1][j][-1] = (
                            np.log(alphahats[i][-1][j])
                            + np.log(betahats[i][-1][j])
                            + np.log(cs[i][-1])
                        )
                    else:
                        xi[-1][j][-1] = (
                            alphahats[i][-1][j] * betahats[i][-1][j] * cs[i][-1]
                        )

            if uselog:
                xi = np.exp(xi)
            xirs += [xi]
            xirbars += [np.sum(xi, axis=0)]

        xibar = np.sum(xirbars, axis=0)
        return xibar

    def printoutput(self, newmean, newcov):
        print("Estimated a:")
        print(self.pi)
        print()
        print("Estimated A:")
        print(self.A)
        print()
        print("Estimated means:")
        print(newmean)
        print()
        print("Estimated covariances:")
        print(newcov)

    """
    Baum-Welch Algorithm for learning HMM parameters from observations
    """

    def baum_welch(self, obs, niter, uselog=False, prin=0, scaled=True):
        A = self.A
        B = self.B

        old_log = float("inf")
        for it in range(niter):
            alphahats, betahats, cs = self.calcabc(obs)  # from Assignment 3 and 4

            new_log = np.nanmean(
                np.sum(
                    np.log([a for a in cs if not any([np.isnan(ax) for ax in a])])
                )
            )
            if abs(old_log - new_log) < 0.000001:
                print(f"\t\tTraining took {it} iterations")
                break
            old_log = new_log

            gammas = self.calcgammas(
                alphahats, betahats, cs, obs, uselog
            )  # alpha*beta*c
            newpi = self.calcinit(gammas, uselog)  # average of gammas[:,0]
            xibar = self.calcxi(alphahats, betahats, cs, obs, uselog)  # page 132
            if uselog:
                xibar = np.exp(xibar)

            newA = np.array(
                [i / np.sum(i) for i in xibar]
            )  # xibar/sum_k(xibar); page 130

            if uselog:
                gammas = np.exp(gammas)

            summ = np.zeros((self.B.shape[0], obs[0].shape[1]))
            sumc = np.zeros((self.B.shape[0], obs[0].shape[1], obs[0].shape[1]))
            sumg = np.zeros((self.B.shape[0]))

            for i in range(obs.shape[0]):
                for t in range(obs[i].shape[0]):
                    for j in range(self.B.shape[0]):
                        summ[j] += obs[i][t] * gammas[i][t][j]
                        sumg[j] += gammas[i][t][j]
                        temp = obs[i][t] - np.atleast_2d(self.B[j].getmean())
                        sumc[j] += gammas[i][t][j] * (temp.T.dot(temp))

            newmean = np.zeros(summ.shape)
            newcov = np.zeros(sumc.shape)
            for i in range(newmean.shape[0]):
                if sumg[i] > 0:
                    newmean[i] = summ[i] / sumg[i]
                    newcov[i] = sumc[i] / sumg[i]
                else:
                    newmean[i] = 0
                    newcov[i] = 0

            # update all variables
            self.pi = newpi
            self.A = newA
            newB = np.array(
                [multigaussD(newmean[i], newcov[i]) for i in range(B.shape[0])]
            )
            self.B = newB
        if prin:
            self.printoutput(newmean, newcov)


"""
Multivariate Gaussian Distribution Class
"""


class multigaussD:
    mean = np.array([0])
    cov = np.array([[0]])

    def __init__(self, mu, C):
        if C.shape[0] is not C.shape[1]:
            print("error, non-square covariance matrix supplied")
            return
        if mu.shape[0] is not C.shape[0]:
            print("error, mismatched mean vector and covariance matrix dimensions")
            return
        self.mean = mu
        if np.where(np.diag(C) == 0)[0].shape[0] != 0:
            C += np.diagflat(np.ones(C.shape[0]) / 10000)
        C[np.isnan(C)] = 1
        self.cov = C
        return

    def random(self, num):
        return np.random.multivariate_normal(self.mean, self.cov, num)

    def rand(self):
        return np.random.multivariate_normal(self.mean, self.cov, 1)[0]

    def likelihood(self, X):
        p = scipy.stats.multivariate_normal(self.mean, self.cov, 1)
        pd = p.pdf(X)
        return pd

    def loghood(self, X):
        return np.log(self.likelihood(X))

    def getmean(self):
        return self.mean

    def getcov(self):
        return self.cov


def prob(x, B):
    T = x.shape[0]
    N = B.shape[0]
    res = np.zeros((T, N))
    for i in range(T):
        for j in range(N):
            res[i, j] = B[j].likelihood(x[i])
    scaled = np.zeros(res.shape)
    for i in range(scaled.shape[0]):
        for j in range(scaled.shape[1]):
            scaled[i, j] = res[i, j] / np.amax(res[i])
    return res, scaled


def logprob(x, B):
    res, scaled = prob(x, B)
    return np.log(res), np.log(scaled)


def logprob_2(c_ts: List[float]) -> float:
    return sum(np.log(c_ts))
