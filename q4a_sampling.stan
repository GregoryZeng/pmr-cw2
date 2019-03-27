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