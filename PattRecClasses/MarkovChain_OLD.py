import random
from typing import List, Optional

import numpy as np


MAX_ITERS = 10000


class MarkovChain:
    def __init__(self, n: Optional[int] = None, m: Optional[int] = None, q: Optional[np.array] = None, a: Optional[np.array] = None, obs: List[int], is_finite: bool = False):
        self.obs = obs
        if q is None or a is None:
            self.n = n
            self.m = m
            q_unnorm = np.array([1 / n + random.uniform(-0.05, 0.05) for _ in range(n)])
            self.q = np.array(q_unnorm / np.linalg.norm(q_unnorm))
            a_unnorm = np.array(
                [
                    np.array(
                        [
                            1 / n + random.uniform(-0.05, 0.05)
                            for _ in range(n + (1 if is_finite else 0))
                        ]
                    )
                    for _ in range(n)
                ]
            )
            self.a = np.array(a_unnorm / np.linalg.norm(a_unnorm))
        else:
            self.n = len(self.a)
            self.m = len(self.a[0])
            self.q = q
            self.a = a
        b_unnorm = np.array(
            [
                np.array([1 / n + random.uniform(-0.05, 0.05) for _ in range(m)])
                for _ in range(n)
            ]
        )
        self.b = np.array(b_unnorm / np.linalg.norm(b_unnorm))

    def forward(self):
        i = 0
        old_log_prob = float("-inf")

        c_0 = 0.0
        a_ts = [[]]
        for i in range(self.n):
            a_ts[0].append(self.q[i] * self.b[i][self.obs[0]])
            c_0 += a_ts[0][-1]

        c_0 = 1 / c_0
        for i in range(self.n):
            a_ts[0][i] *= c_0

        for t in range(1, len(self.obs)):
            c_t = 0.0
            a_ts.append([])
            for i in range(self.n):
                a_ts[t].append(0.0)
                for j in range(self.n):
                    a_ts[t][i] += a_ts[t-1][j] * self.a[j][i]
                a_ts[t][i] *= self.b[i][self.obs[t]]
                c_t += a_ts[t][i]
            c_t = 1 / c_t
            for i in range(self.n):
                a_ts[t][i] *= c_t

    def finiteDuration(self):
        pass

    def backward(self):


    def adaptStart(self):
        pass

    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass
