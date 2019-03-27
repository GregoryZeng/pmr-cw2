import numpy as np
import scipy as sp
from scipy.stats import norm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pystan
import pickle
import os
from sklearn.decomposition import PCA

if __name__ == "__main__":
    data = np.loadtxt('./q4_pca.txt')
    fig, ax = plt.subplots()
    ax.scatter(x=data[0],y=data[1],label='data')


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

    idx = np.random.choice(a=10000,size=25)
    Ws, Taus = result.extract()['W'], result.extract()['tau']
    u_pairs = []
    for i in range(25):
        e_val, e_vec = np.linalg.eig(Ws[i] @ Ws[i].T + 1/Taus[i])
        # print(e_val.shape,e_vec.shape)
        sorted_idx = np.argsort(e_val)
        # rescaled - column 0 is the second eigenvector, column 1 is the first eigenvector
        rescaled = e_vec[:,sorted_idx[-2:]] * np.sqrt(e_val[-2:])[None,:]
        u_pairs += [rescaled[:2,1], rescaled[:2,0]]
    u_pairs = np.array(u_pairs)

    ax.scatter(x=u_pairs[::2, 0], y=u_pairs[::2,1],label='PPCA $u_1$')
    ax.scatter(x=u_pairs[1::2, 0], y=u_pairs[1::2, 1], label='PPCA $u_2$')

    # **** standard PCA ****
    u_pairs = []
    e_val, e_vec = np.linalg.eig(data @ data.T)
    sorted_idx = np.argsort(e_val)
    rescaled_vec = e_vec * np.sqrt(e_val)[None,:]
    ax.scatter(x=rescaled_vec[0,sorted_idx[-1]],
               y=rescaled_vec[1,sorted_idx[-1]],
               label='PCA $u_1$')
    ax.scatter(x=rescaled_vec[0, sorted_idx[-2]],
               y=rescaled_vec[1, sorted_idx[-2]],
               label='PCA $u_2$')

    # wrap up plotting
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.legend(loc='upper left')
    fig.savefig('ppca.pdf')
