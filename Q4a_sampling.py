import numpy as np
import scipy as sp
from scipy.stats import norm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pystan


if __name__ == "__main__":
    stan_code = """
    data{
        int N;
        vector[N] xs;
        int ys[N];
    }

    parameters{
        real alpha;
        real beta;
    }

    transformed parameters{
        vector[N] lambda;
        lambda = exp( alpha * xs + beta ); 
    }

    model {
        alpha ~ normal(0,10);
        beta ~ normal(0,10);
        ys ~ poisson(lambda);
    }
    """
    xs = [-5.051905265552104618e-01,
          -1.718571932218771470e-01,
          1.614761401114561679e-01,
          4.948094734447895382e-01,
          8.150985069051909226e-01]
    ys = [1, 0, 2, 1, 2]

    sm = pystan.StanModel(model_code=stan_code)
    results = sm.sampling(data={"N": 5, "xs": xs, "ys": ys},
                          iter=10000,
                          chains=1)
    alphas = results.extract()["alpha"]
    betas = results.extract()["beta"]

    samples_df = pd.DataFrame({'alpha': alphas, 'beta': betas})
    ax = sns.scatterplot(data=samples_df, x='alpha', y='beta', s=10, )
    ax.get_figure().savefig('stan_poisson_it10000.pdf')

    print('mean:', samples_df.mean())
    print('corrcoef:', np.corrcoef(x=np.array(samples_df.values), rowvar=False))

