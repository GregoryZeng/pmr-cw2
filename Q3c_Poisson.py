import numpy as np
import scipy as sp
from scipy.stats import norm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mh import mh

if __name__ == "__main__":
    xs = [-5.051905265552104618e-01,
          -1.718571932218771470e-01,
          1.614761401114561679e-01,
          4.948094734447895382e-01,
          8.150985069051909226e-01]
    ys = [1, 0, 2, 1, 2]


    def unnormalized_posterior(v):
        a, b = v[0], v[1]
        result = -(a ** 2 + b ** 2) / 200
        for i in range(5):
            result += ys[i] * (a * xs[i] + b) - np.exp(a * xs[i] + b)
        return np.exp(result)


    samples1 = mh(p_star=unnormalized_posterior,
                  param_init=np.array([0, 0]),
                  num_samples=5000,
                  stepsize=1,
                  W=1000)
    print('sample size:', len(samples1))
    samples1_df = pd.DataFrame(samples1, columns=['x', 'y'])
    samples1=np.array(samples1)
    # ax = sns.scatterplot(data=samples1_df, x='x', y='y', s=10, )
    # ax.get_figure().savefig('poisson_x0_y0_e1_w1000.pdf')
    fig, ax = plt.subplots()
    ax.scatter(x=samples1[:,0],y=samples1[:,1],s=10)
    ax.set_xlabel('$\\alpha$')
    ax.set_ylabel('$\\beta$')
    # ax.legend('upper left')
    fig.savefig('poisson_x0_y0_e1_w1000.pdf')


    print('mean:', samples1_df.mean())
    print('corrcoef:', np.corrcoef(x=np.array(samples1), rowvar=False))
