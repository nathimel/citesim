// Hierarchical regression model
// This model follows the same structure as shown in the STAN users guide.
data {
    int<lower=1> D; // Number of single-hierarchical-level dimensions per obs
    int<lower=0> N; // Number of obs
    int<lower=1> L; // Number of categories (levels)
    int<lower=1> K; // Number of secondary categories (inner levels)
    array[N] real y; // Outcome per obs
    array[N] int<lower=1, upper=L> ll; // Category per obs
    array[N] int<lower=1, upper=K> kk; // Second category per obs
    array[N] real x; // Observations
    array[N] row_vector[D] xd; // Observations that only have a single hierarchical level.

    // Test data
    int<lower=0> N_test;
    array[N_test] real x_test;
    array[N_test] row_vector[D] xd_test;
    vector[N_test] y_test;
}
parameters {
    real alpha; // Global intercept
    real<lower=0> sigma; // Global error
    array[L] vector[K] beta; // Slope per level and inner level
    array[K] real mu_beta; // Mean slope across inner levels
    array[K] real<lower=0> sigma_beta; // Std between slopes of diff inner levels
    real mu_inner_beta; // Inner-most slope overall
    real<lower=0> sigma_inner_beta; // Inner-most slope overall

    array[L] vector[D] betad; // Slope per level
    array[D] real mu_betad; // Mean slope across levels
    array[D] real<lower=0> sigma_betad; // Std between slopes of diff levels
}
model {
    // Correlation between inner levels
    mu_beta ~ normal(mu_inner_beta, sigma_inner_beta);

    // Loop through levels and draw the slopes
    for (l in 1:L) {
        beta[l] ~ normal(mu_beta, sigma_beta); // One beta per level and second level
        betad[l] ~ normal(mu_betad, sigma_betad); // One beta per level and dimension
    }
    // Loop through observations and create the y input for each
    vector[N] mu; // The mean for y, which is a function of alpha, beta, etc
    for (n in 1:N) {
        mu[n] = alpha + x[n] * beta[ll[n], kk[n]] + xd[n] * betad[ll[n]];
    }
    // Vectorized normal fn
    y ~ normal(mu, sigma);
}
generated quantities {
    // log_p is used to calculate the log posterior predictive density
    real log_p;
    // Nested block lets us declare a local variable mu that's not saved
    {
        vector[N_test] mu; // The mean for y, which is a function of alpha, beta, etc
        for (n in 1:N_test) {
            mu[n] = alpha + x[n] * beta[ll[n], kk[n]] + xd[n] * betad[ll[n]];
        }
        log_p = normal_lpdf(y_test | mu, sigma);
    }
}