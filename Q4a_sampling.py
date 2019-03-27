import numpy as np
import scipy as sp
from scipy.stats import norm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pystan
import os
import pickle

if __name__ == "__main__":
    xs = [-5.051905265552104618e-01,
          -1.718571932218771470e-01,
          1.614761401114561679e-01,
          4.948094734447895382e-01,
          8.150985069051909226e-01]
    ys = [1, 0, 2, 1, 2]

    if os.path.isfile('q4a.pkl'):
        sm = pickle.load(open('q4a.pkl', 'rb'))
    else:
        sm = pystan.StanModel(file='q4a_sampling.stan')
        with open('q4a.pkl', 'wb') as f:
            pickle.dump(sm, f)

    results = sm.sampling(data={"N": 5, "xs": xs, "ys": ys},
                          iter=10000,
                          chains=1)
    alphas = results.extract()["alpha"]
    betas = results.extract()["beta"]

    fig, ax = plt.subplots()
    samples_df = pd.DataFrame({'alpha': alphas, 'beta': betas})
    # ax = sns.scatterplot(data=samples_df, x='alpha', y='beta', s=10, )
    ax.scatter(x=alphas, y=betas, s=5)
    ax.set_xlabel('$\\alpha$')
    ax.set_ylabel('$\\beta$')
    fig.savefig('stan_poisson_it10000.pdf')

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Sample')
    ax.plot(np.arange(5000), alphas, 'b-', linewidth=.1)
    print(alphas)
    fig.savefig('stan_poisson_alpha_trace.pdf')

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Sample')
    ax.plot(np.arange(5000), betas, 'b-', linewidth=.1)
    fig.savefig('stan_poisson_beta_trace.pdf')

    print('mean:', samples_df.mean())
    print('corrcoef:', np.corrcoef(x=np.array(samples_df.values), rowvar=False))
