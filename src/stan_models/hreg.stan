data {
    int<lower=1> D; // Number of dimensions per obs
    int<lower=0> N; // Number of obs
    int<lower=1> L; // Number of categories (levels)
    array[N] real y; // Outcome per obs
    array[N] int<lower=1, upper=L> ll; // Category per obs
    array[N] row_vector[D] x; // Observations
}
parameters {
    real alpha; // Global intercept
    real<lower=0> sigma; // Global error
    array[D] real mu; // Mean slope across levels
    array[D] real<lower=0> sigma_beta; // Std between slopes of diff levels
    array[L] vector[D] beta; // Slope per level
}
model {
    // Loop through levels and draw the slopes
    for (l in 1:L) {
        beta[l] ~ normal(mu, sigma_beta); // One beta per level and dimension
    }
    // Loop through observations and create the y input for each
    vector[N] y_mu;
    for (n in 1:N) {
        y_mu[n] = alpha + x[n] * beta[ll[n]];
    }
    // Vectorized normal fn
    y ~ normal(y_mu, sigma);
}
// data {
//    int<lower=0> N;
//    vector[N] x;
//    vector[N] y;
// }
// parameters {
//     real alpha;
//     real beta;
//     real<lower=0> sigma;
// }
// model {
//     y ~ normal(alpha + beta * x, sigma);
// }
// data {
//     int<lower=1> D;
//     int<lower=0> N;
//     int<lower=1> L;
//     array[N] int<lower=0, upper=1> y;
//     array[N] int<lower=1, upper=L> ll;
//     array[N] row_vector[D] x;
// }
// parameters {
//     array[D] real mu;
//     array[D] real<lower=0> sigma;
//     array[L] vector[D] beta;
// }
// mo
// model {
//     for (d in 1:D) {
//     mu[d] ~ normal(0, 100);
//     for (l in 1:L) {
//     beta[l, d] ~ normal(mu[d], sigma[d]);
//     }
//     }
//     for (n in 1:N) {
//     y[n] ~ bernoulli(inv_logit(x[n] * beta[ll[n]]));
//     }
// }