// This model uses the QR reparameterization, which enables easy multivariate regression
data {
    int<lower=0> N; // Number of data items
    int<lower=0> K; // Number of predictors
    matrix[N, K] x; // predictor matrix
    vector[N] y; // outcome vector
    int<lower=0> N_test;
    matrix[N_test, K] x_test;
    vector[N] y_test;
}
transformed data {
    matrix[N, K] Q_ast;
    matrix[K, K] R_ast;
    matrix[K, K] R_ast_inverse;
    // thin and scale the QR decomposition
    Q_ast = qr_thin_Q(x) * sqrt(N - 1);
    R_ast = qr_thin_R(x) / sqrt(N - 1);
    R_ast_inverse = inverse(R_ast);

    // Same for test values.
    matrix[N_test, K] Q_ast_test;
    Q_ast_test = qr_thin_Q(x_test) * sqrt(N_test - 1);
}
parameters {
    real alpha; // Intercept
    vector[K] theta; // Coefficients on Q_ast
    real<lower=0> sigma; // error scale
}
model {
    y ~ normal(Q_ast * theta + alpha, sigma);
}
generated quantities {
    vector[K] beta;
    beta = R_ast_inverse * theta; // coefficients on x

    // We predict a full probability distribution for each x_test
    // This gets crazy because it produces an array of size (N, N_test)
    array[N_test] real y_tilde = normal_rng(Q_ast_test * theta + alpha, sigma);
}
//    log_p is used to calculate the log posterior predictive density
//    real log_p = normal_lpdf(y_test | alpha, sigma);
// }