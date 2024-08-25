data {
    int<lower=0> N;
    vector[N] x;
    vector[N] y;
}
parameters {
    real alpha;
    real beta;
    real<lower=0> sigma;
}
model {
    y ~ normal(alpha + beta * x, sigma);
}
// generated quantities {
    // array[N] real y_rep = normal_rng(alpha + beta * x, sigma);
// }