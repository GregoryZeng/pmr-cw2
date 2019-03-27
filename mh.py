import numpy as np
def mh(p_star, param_init, num_samples=5000, stepsize=1.0, W=0):
    param_init = np.array(param_init)
    if num_samples == 0:
        return []
    x_curr = param_init
    # burn-in stage
    for i in range(W):
        x_cand = np.random.normal(x_curr,stepsize,x_curr.shape)
        a = p_star(x_cand)/p_star(x_curr)
        if a >= 1:
            x_curr = x_cand
        else:
            u = np.random.uniform()
            if u < a:
                x_curr = x_cand
    # sampling stage
    samples = [x_curr]
    for i in range(num_samples-1):
        x_cand = np.random.normal(x_curr,stepsize,x_curr.shape)
        a = p_star(x_cand)/p_star(x_curr)
        if a >= 1:
            x_curr = x_cand
        else:
            u = np.random.uniform()
            if u < a:
                x_curr = x_cand
        samples.append(x_curr)
    return samples
