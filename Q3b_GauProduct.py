import numpy as np
import scipy as sp
from scipy.stats import norm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mh import mh

if __name__ == "__main__":
    ## Q3b
    # 5000 samples, init = (0,0), ep = 1
    samples1 = mh(p_star=lambda x: norm.pdf(x[0], 0, 1) * norm.pdf(x[1], 0, 1),
                  param_init=np.array([0, 0]),
                  num_samples=5000,
                  stepsize=1)
    samples1_df = pd.DataFrame(samples1, columns=['x', 'y'])
    # ax = sns.scatterplot(data=samples1_df.iloc[20:], x='x', y='y', s=10, )
    # ax = sns.scatterplot(data=samples1_df.iloc[:20], x='x', y='y', s=30, ax=ax)
    # ax.get_figure().savefig('gauprod_x0_y0_e1.pdf')
    fig, ax = plt.subplots()
    samples1=np.array(samples1)
    ax.scatter(x=samples1[20:, 0], y=samples1[20:, 1], s=10, label='others')
    ax.scatter(x=samples1[:20, 0], y=samples1[:20, 1], s=10, marker='x', label='the first 20 samples')
    ax.legend(loc='upper left')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    fig.savefig('gauprod_x0_y0_e1.pdf')

    # 5000 samples, init = (7,7), ep = 1
    samples1 = mh(p_star=lambda x: norm.pdf(x[0], 0, 1) * norm.pdf(x[1], 0, 1),
                  param_init=np.array([7, 7]),
                  num_samples=5000,
                  stepsize=1,
                  )
    samples1 = np.array(samples1)
    samples1_df = pd.DataFrame(samples1, columns=['x', 'y'])
    # plt.clf()
    # ax = sns.scatterplot(data=samples1_df.iloc[20:], x='x', y='y', s=10, )
    # ax = sns.scatterplot(data=samples1_df.iloc[:20], x='x', y='y', s=30, ax=ax)
    # ax.get_figure().savefig('gauprod_x7_y7_e1.pdf')
    fig, ax = plt.subplots()
    ax.scatter(x=samples1[20:, 0], y=samples1[20:, 1], s=10, label='others')
    ax.scatter(x=samples1[:20, 0], y=samples1[:20, 1], s=10, marker='x', label='the first 20 samples')
    ax.legend(loc='upper left')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    fig.savefig('gauprod_x7_y7_e1.pdf')