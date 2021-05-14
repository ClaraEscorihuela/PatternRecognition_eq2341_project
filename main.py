import math
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import scipy.stats as stats

from PattRecClasses import MarkovChain, GaussD, HMM, gauss_logprob
import matplotlib.font_manager as font_manager

# # One-dimensional Gaussian
# g1 = GaussD( means=np.array([0]) , stdevs=np.array([1.0]))
# g2 = GaussD( means=np.array([3]) , stdevs=np.array([2.0]))
# obs = np.array([-0.2, 2.6, 1.3])
# prueba1 = gauss_logprob.gauss_logprob([g1], np.array([obs]))
# prueba2 = gauss_logprob.gauss_logprob([g2], np.array([obs]))
def main():
    classes = {0: 0, 1: 0}
    iters = 500
    # rng = 100000
    rng = 1
    lengths = {i: 0 for i in range(iters + 2)}
    xs = []

    font_dirs = ['../fonts']
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)

    plt.rcParams['font.family'] = "Helvetica"

    q = np.array([1.0, 0.0])
    a = np.array([np.array([0.9, 0.1, 0.0]), np.array([0.0, 0.9, 0.1])])
    obs = np.array([-0.2, 2.6, 1.3])

    g1 = GaussD(means=[0.0], stdevs=[1.0])
    g2 = GaussD(means=[3.0], stdevs=[2.0])
    prob_obs = [g.prob(obs) for g in [g1, g2]]
    mc = MarkovChain(initial_prob=q, transition_prob=a, dist_obs=prob_obs, obs=obs, scaling=True)
    h = HMM(mc, [g1, g2])
    a_ts, c_ts = mc.forward()
    b_ts = mc.backward()
    log_prob = h.logprob(c_ts)
    print("A_hat =", a_ts, "C_ts =", c_ts, "B_hat =", b_ts)
    print("LogProbability = ", log_prob)

    q = np.array([1.0, 0.0, 0.0])
    a = np.array([np.array([0.3, 0.7, 0.0]), np.array([0.0, 0.5, 0.5]), np.array([0.0, 0.0, 1.0])])
    obs = np.array([1, 2, 4, 4, 1])
    g3 = GaussD(means=[6.0], stdevs=[3.0])
    prob_obs = [g.prob(obs) for g in [g1, g2, g3]]

    mc = MarkovChain(initial_prob=q, transition_prob=a, dist_obs=prob_obs, obs=obs, scaling=False)
    h = HMM(mc, [g1, g2, g3])
    a_ts, c_ts = mc.forward()
    b_ts = mc.backward()
    log_prob = h.logprob(c_ts)
    print("A_hat =", a_ts, "C_ts =", c_ts, "B_hat =", b_ts)
    print("LogProbability = ", log_prob)


if __name__ == "__main__":
    main()
