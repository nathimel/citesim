data {
    int<lower=1> D; // Number of dimensions per obs
    int<lower=0> N; // Number of obs
    int<lower=1> L; // Number of categories (levels)
    array[N] real y; // Outcome per obs
    array[N] int<lower=1, upper=L> ll; // Category per obs
    array[N] row_vector[D] x; // Observations

    // Test data
    int<lower=0> N_test;
    array[N_test] row_vector[D] x_test;
    vector[N] y_test;
}
parameters {
    real alpha; // Global intercept
    real<lower=0> sigma; // Global error
    array[L] vector[D] beta; // Slope per level
    array[D] real mu_beta; // Mean slope across levels
    array[D] real<lower=0> sigma_beta; // Std between slopes of diff levels
}
model {
    // Loop through levels and draw the slopes
    for (l in 1:L) {
        beta[l] ~ normal(mu_beta, sigma_beta); // One beta per level and dimension
    }
    // Loop through observations and create the y input for each
    vector[N] mu; // The mean for y, which is a function of alpha, beta, etc
    for (n in 1:N) {
        mu[n] = alpha + x[n] * beta[ll[n]];
    }
    // Vectorized normal fn
    y ~ normal(mu, sigma);
}
generated quantities {
    // log_p is used to calculate the log posterior predictive density
    vector[N] mu; // The mean for y, which is a function of alpha, beta, etc
    for (n in 1:N) {
        mu[n] = alpha + x[n] * beta[ll[n]];
    }
    real log_p = normal_lpdf(y_test | mu, sigma);
}