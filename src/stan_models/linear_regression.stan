data {
    int<lower=0> N;
    vector[N] x;
    vector[N] y;
    int<lower=0> N_test;
    vector[N_test] x_test;
    vector[N_test] y_test;
}
parameters {
    real alpha;
    real beta;
    real<lower=0> sigma;
}
model {
    y ~ normal(alpha + beta * x, sigma);
}
generated quantities {
   array[N_test] real y_tilde = normal_rng(alpha + beta * x_test, sigma);
   real log_p = normal_lpdf(y_test | alpha + beta * x_test, sigma);
}