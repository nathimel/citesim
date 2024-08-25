data {
    int<lower=0> N;
    vector[N] y;
    int<lower=0> N_test;
    vector[N_test] y_test;
}
parameters {
    real alpha;
    real<lower=0> sigma;
}
model {
    y ~ normal(alpha, sigma);
}
generated quantities {
   // log_p is used to calculate the log posterior predictive density
   real log_p = normal_lpdf(y_test | alpha, sigma);
}