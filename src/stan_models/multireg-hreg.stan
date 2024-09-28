// Mixed multivariate + hierarchical regression model
// The user can mix and match which predictors to model hierarchically vs which
// to model with regular multivariate regression
data {
    // Shared quantities
    int<lower=0> N; // Number of obs
    array[N] real y; // Outcome per obs

    // Shared test data
    int<lower=0> N_test;
    vector[N] y_test;

    // Multireg quantities
    int<lower=1> K; // Number of predictors/dimensions for multiregression
    matrix[N, K] x_m; // Observations
    matrix[N_test, K] x_m_test; // Test observations

    // Hreg quantities
    int<lower=1> D; // Number of dimensions/predictors for hreg
    int<lower=1> L; // Number of categories (levels)
    array[N] int<lower=1, upper=L> ll; // Category per obs
    array[N] row_vector[D] x_h; // Observations
    array[N_test] row_vector[D] x_h_test; // Test observations
}
// For the regular linear regression we transform some of the data
transformed data {
    matrix[N, K] Q_ast;
    matrix[K, K] R_ast;
    matrix[K, K] R_ast_inverse;
    // thin and scale the QR decomposition
    Q_ast = qr_thin_Q(x_m) * sqrt(N - 1);
    R_ast = qr_thin_R(x_m) / sqrt(N - 1);
    R_ast_inverse = inverse(R_ast);

    // Same for test values.
    matrix[N_test, K] Q_ast_test;
    Q_ast_test = qr_thin_Q(x_m_test) * sqrt(N_test - 1);
}
parameters {
    // Shared params
    real alpha; // Global intercept
    real<lower=0> sigma; // Global error

    // Multireg params
    vector[K] theta; // Coefficients on Q_ast

    // Hreg params
    array[L] vector[D] beta; // Slope per level
    array[D] real mu_beta; // Mean slope across levels
    array[D] real<lower=0> sigma_beta; // Std between slopes of diff levels
}
model {
    // Multivariate regression
    vector[N] mu = alpha + Q_ast * theta;

    // Hierarchical regression
    // Loop through levels and draw the slopes
    for (l in 1:L) {
        beta[l] ~ normal(mu_beta, sigma_beta); // One beta per level and dimension
    }
    // Loop through observations and create the y input for each
    for (n in 1:N) {
        mu[n] += x_h[n] * beta[ll[n]];
    }

    // Vectorized normal fn
    y ~ normal(mu, sigma);
}
generated quantities {
    // Multireg betas
    vector[K] beta_m;
    beta_m = R_ast_inverse * theta; // coefficients on x

    // log_p is used to calculate the log posterior predictive density
    real log_p;
    {
        vector[N] mu = alpha + Q_ast_test * theta;
        for (n in 1:N) {
            mu[n] += x_h_test[n] * beta[ll[n]];
        }
        log_p = normal_lpdf(y_test | mu, sigma);
    }
}