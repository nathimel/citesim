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

    // We predict a full probability distribution for each x_test
    // This gets crazy because it produces an array of size (N, N_test)
    array[N_test] real y_tilde = normal_rng(alpha + beta * x_test, sigma);

    // log_p is used to calculate the log posterior predictive density
    real log_p = normal_lpdf(y_test | alpha + beta * x_test, sigma);
}