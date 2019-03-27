import numpy as np
import scipy as sp
from scipy.stats import norm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pystan
import pickle
import os

if __name__ == "__main__":
    data = np.loadtxt('./q4_pca.txt')
    fig, ax = plt.subplots()

    # **** PPCA ****
    if os.path.isfile('q4c.pkl'):
        sm_pca = pickle.load(open('q4c.pkl', 'rb'))
    else:
        sm_pca = pystan.StanModel(file='pca.stan')
        with open('q4c.pkl', 'wb') as f:
            pickle.dump(sm_pca, f)



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

    Ws = result.extract()['W']

    ax.scatter(x=Ws[:,0,0],y=Ws[:,0,1],s=5)


    # wrap up plotting
    ax.set_xlabel('$W_{11}$')
    ax.set_ylabel('$W_{12}$')
    ax.legend(loc='upper left')
    fig.savefig('ppca_W_m0.pdf')