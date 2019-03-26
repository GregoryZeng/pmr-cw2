import numpy as np
import scipy as sp
from scipy.stats import norm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pystan


if __name__ == "__main__":
    data = np.loadtxt('./q4_pca.txt')
    sm_pca = pystan.StanModel(file='pca.stan')
    result = sm_pca.sampling(data={'N': 15,
                                   'D': 3,
                                   'K': 2,
                                   'X': data.T,
                                   'mu_W': 0,
                                   'alpha0': 1,
                                   'beta0': 1},
                             iter=10000,
                             chains=2,
                             n_jobs=2)