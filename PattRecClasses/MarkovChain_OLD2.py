import random
from typing import Tuple, List

import numpy as np
from .DiscreteD import DiscreteD


MAX_ITERS = 5
LOGPROB_THRESHOLD = 0.01


class MarkovChain:
    """
    MarkovChain - class for first-order discrete Markov chain,
    representing discrete random sequence of integer "state" numbers.
    
    A Markov state sequence S(t), t=1..T
    is determined by fixed initial probabilities P[S(1)=j], and
    fixed transition probabilities P[S(t) | S(t-1)]
    
    A Markov chain with FINITE duration has a special END state,
    coded as nStates+1.
    The sequence generation stops at S(T), if S(T+1)=(nStates+1)
    """
    def __init__(self, n: int, m: int):
        self.n = n
        self.m = m

        self.q = np.array([(1 / self.n) + random.uniform(-0.05, 0.05) for _ in range(self.n)])
        self.A = np.array([np.array([(1 / self.n) + random.uniform(-0.05, 0.05) for _ in range(self.n)]) for _ in range(self.n)])
        self.B = np.array([np.array([(1 / self.n) + random.uniform(-0.05, 0.05) for _ in range(self.m)]) for _ in range(self.n)])

        self.c_ts = None
        self.a_ts = None
        self.b_ts = None
        self.gamma = None

        self.is_finite = False

    def probDuration(self, tmax):
        """
        Probability mass of durations t=1...tMax, for a Markov Chain.
        Meaningful result only for finite-duration Markov Chain,
        as pD(:)== 0 for infinite-duration Markov Chain.
        
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.8.
        """
        pD = np.zeros(tmax)

        if self.is_finite:
            pSt = (np.eye(self.n)-self.A.T)@self.q

            for t in range(tmax):
                pD[t] = np.sum(pSt)
                pSt = self.A.T@pSt

        return pD

    def probStateDuration(self, tmax):
        """
        Probability mass of state durations P[D=t], for t=1...tMax
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.7.
        """
        t = np.arange(tmax).reshape(1, -1)
        aii = np.diag(self.A).reshape(-1, 1)
        
        logpD = np.log(aii)*t+ np.log(1-aii)
        pD = np.exp(logpD)

        return pD

    def meanStateDuration(self):
        """
        Expected value of number of time samples spent in each state
        """
        return 1/(1-np.diag(self.A))
    
    def rand(self, tmax: int) -> np.array:
        """
        S=rand(self, tmax) returns a random state sequence from given MarkovChain object.
        
        Input:
        tmax= scalar defining maximum length of desired state sequence.
           An infinite-duration MarkovChain always generates sequence of length=tmax
           A finite-duration MarkovChain may return shorter sequence,
           if END state was reached before tmax samples.
        
        Result:
        S= integer row vector with random state sequence,
           NOT INCLUDING the END state,
           even if encountered within tmax samples
        If mc has INFINITE duration,
           length(S) == tmax
        If mc has FINITE duration,
           length(S) <= tmaxs
        """

        s_seq = [random.choices(range(len(self.q)), weights=self.q)[0]]
        
        for _ in range(tmax):
            s = random.choices(range(len(self.A[0])), weights=self.A[s_seq[-1]])[0]
            if s == len(self.q):
                break
            s_seq.append(s)

        return np.array(s_seq)

    def viterbi(self):
        pass

    def stationaryProb(self):
        pass

    def stateEntropyRate(self):
        pass

    def setStationary(self):
        pass

    def join(self):
        pass

    def initLeftRight(self):
        pass

    def initErgodic(self):
        pass

    def forward(self, obs: List[int]) -> Tuple[List[List[float]], List[float]]:
        self.c_ts = [0.0]
        self.a_ts = [[]]
        for i in range(self.n):
            self.a_ts[0].append(self.q[i] * self.B[i][obs[0]])
            self.c_ts[0] += self.a_ts[0][-1]

        self.c_ts[0] = 1 / self.c_ts[0]
        for i in range(self.n):
            self.a_ts[0][i] *= self.c_ts[0]

        for t in range(1, len(obs)):
            self.c_ts.append(0.0)
            self.a_ts.append([])
            for i in range(self.n):
                self.a_ts[t].append(0.0)
                for j in range(self.n):
                    self.a_ts[t][i] += self.a_ts[t - 1][j] * self.A[j][i]
                self.a_ts[t][i] *= self.B[i][obs[t]]
                self.c_ts[t] += self.a_ts[t][i]
            self.c_ts[t] = 1 / self.c_ts[t]
            for i in range(self.n):
                self.a_ts[t][i] *= self.c_ts[t]

        self.c_ts = list(1/np.array(self.c_ts))
        self.c_ts.append(sum([self.a_ts[-1][i] * self.A[i][-1] for i in range(self.n)]))

        return self.a_ts, self.c_ts

    def compute_gamma(self, obs: List[int]) -> List[List[float]]:
        self.gamma = []
        for t in range(len(obs) - 1):
            self.gamma.append([])
            for i in range(self.n):
                self.gamma[t][i] += self.a_ts[t][i] * self.b_ts[t][i] * self.c_ts[t][i]
        self.gamma.append(self.a_ts[-1])

        return self.gamma

    def finiteDuration(self):
        pass
    
    def backward(self, obs: List[int]) -> List[List[float]]:
        # Infinite
        if len(self.B) == len(self.B[0]):
            self.b_ts = [self.n * [1 / self.c_ts[-1]]]
        # Finite
        else:
            self.b_ts = [[self.A[i][-1] / (self.c_ts[-2] * self.c_ts[-1]) for i in range(self.n)]]

        for t in reversed(range(len(obs) - 1)):
            self.b_ts.insert(0, [])
            for i in range(self.n):
                self.b_ts[0].append(0.0)
                for j in range(self.n):
                    self.b_ts[0][i] += self.A[i][j] * self.B[j][obs[t + 1]] * self.b_ts[1][j]
                self.b_ts[0][i] *= 1 / self.c_ts[t]

        return self.b_ts

    def baum_welch(self, obs: List[int]):
        i = 0
        log_prob = float("inf")

        while i < MAX_ITERS:
            self.forward(obs=obs)

            log_prob_new = self.logprob()
            if abs(log_prob_new - log_prob)< LOGPROB_THRESHOLD:
                break
            log_prob = log_prob_new

            self.backward(obs=obs)
            self.compute_gamma(obs=obs)
            self.reestimate(obs=obs)

            i += 1

    def logprob(self):
        return -sum(np.log(self.c_ts))

    def reestimate(self, obs: List[int]):
        # Re-estimate Q
        self.q = self.gamma[0]

        # Re-estimate A
        for i in range(self.n):
            denom = 0.000001
            for t in range(len(obs) - 1):
                denom += self.gamma[t][i]
            for j in range(self.n):
                numer = 0
                for t in range(len(obs) - 1):
                    numer += self.digamma[t][i][j]
                self.A[i][j] = numer / denom

        # Re-estimate B
        for i in range(self.n):
            denom = 0.000001
            for t in range(len(obs)):
                denom += self.gamma[t][i]
            for j in range(self.m):
                numer = 0
                for t in range(len(obs)):
                    if(obs[t] == j):
                        numer += self.gamma[t][i]
                self.B[i][j] = numer / denom

    def adaptStart(self):
        pass

    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass
