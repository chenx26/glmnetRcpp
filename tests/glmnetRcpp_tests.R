rm(list = ls())
library(h2o)
library(glmnetRcpp)

# negative log likelihood of exponential distribution

ExpNegativeLogLikelihood = function(x, params) {
        rs = params[["A"]] %*% x
        return(t(params[["b"]]) %*% exp(-rs) + sum(rs))
}

# gradient of the negative log likelihood of exponential distribution

GradExpNegativeLogLikelihood = function(x, params) {
        tmp = -t(params[["A"]]) %*% (exp(-params[["A"]] %*% x) * params[["b"]] - 1)
        return(tmp)
}

# proximal of L1 regularizer
prox_L1 = function(x, threshold) {
        return(sapply(x, function(x)
                sign(x) * max(abs(x) - threshold, 0)))
}

# L1 regularizer
regularizer_L1 = function(x) {
        return(sum(abs(x)))
}

# Elastic Net regularizer
regularizer_ENet = function(x, alpha = 1){
        return(alpha * regularizer_L1(x) + (1 - alpha) / 2 * sum(x^2))
}

# proximal of Elastic Net regularizer, not run
# prox_ENet = function(x, threshold, alpha = 1){
#         return( alpha * t / (1 + t * ( 1 - alpha )))
# }

# complete objective function
ObjFun = function(x,
                  params,
                  SmoothFun = ExpNegativeLogLikelihood,
                  regularizer = regularizer_ENet) {
        return(SmoothFun(x, params) +
                       params[["lambda"]] * regularizer(x, params[["alpha"]]))
}

# predict output given the coefficients
predict_exp_glmnet = function(x, params) {
        return(exp(params[["A"]] %*% x))
}

# fit glmnet with vanilla pgd

fit_glmnet_fixed_lambda_vanilla_pgd = function(A,
                                               b,
                                               x0 = rep(0, ncol(A)),
                                               alpha = 0.5,
                                               lambda = 0,
                                               maxiter = 10000,
                                               ABSTOL = 1e-5,
                                               RELTOL = 1e-2,
                                               t = 1,
                                               beta = 1 / 2) {
        params = list(
                b = b,
                # The response variables
                A = A,
                # The matrix
                alpha = alpha,
                # 1 means lasso
                lambda = lambda
        ) # the regularization coefficient
        prox.lambda = 1
        beta = beta
        max_iter = maxiter
        ABSTOL   = ABSTOL

        # save algorithm state
        nvars = ncol(A)
        x_matrix = matrix(0, nrow = max_iter, ncol = nvars)
        grad_x_matrix = matrix(0, nrow = max_iter, ncol = nvars)
        z_matrix = matrix(0, nrow = max_iter, ncol = nvars)
        lhs_vector = rep(NA, max_iter)
        rhs_vector = rep(NA, max_iter)
        obj_vector = rep(NA, max_iter)

        x = x0
        xprev = x

        for (k in 1:max_iter) {
                while (TRUE) {
                        grad_x = GradExpNegativeLogLikelihood(x, params)
                        threshold = prox.lambda * params[["lambda"]] * params[["alpha"]] /
                                (1+prox.lambda * params[["lambda"]]*(1-params[["alpha"]]))
                        z = prox_L1(x - prox.lambda * grad_x, threshold)
                        lhs = ExpNegativeLogLikelihood(z, params)
                        rhs = ExpNegativeLogLikelihood(x, params)
                        rhs = rhs + t(grad_x) %*% (z - x)
                        rhs = rhs + (1 / (2 * prox.lambda)) * sum((z - x) ^ 2)
                        if (lhs <= rhs) {
                                break
                        }
                        prox.lambda = beta * prox.lambda
                }

                xprev = x
                x = z
                x_matrix[k, ] = xprev
                z_matrix[k, ] = z
                grad_x_matrix[k, ] = grad_x
                lhs_vector[k] = lhs
                rhs_vector[k] = rhs

                obj_vector[k] = ObjFun(x, params)
                if (k > 1)
                        if (abs(obj_vector[k] - obj_vector[k - 1]) < ABSTOL) {
                                break
                        }
        }
        x_matrix = x_matrix[1:k, ]
        z_matrix = z_matrix[1:k, ]
        grad_x_matrix = grad_x_matrix[1:k, ]
        lhs_vector = lhs_vector[1:k]
        rhs_vector = rhs_vector[1:k]

        obj_vector
        grad_x_matrix
        lhs_vector
        rhs_vector
        z_matrix
        x.true
        return(z_matrix[k,])
}

fit_glmnet_fixed_lambda_vanilla_pgd_cv = function(A,
                                           b,
                                           lambda,
                                           x0 = rep(0, ncol(A)),
                                           alpha = 0.5,
                                           maxiter = 1000,
                                           ABSTOL = 1e-4,
                                           RELTOL = 1e-2,
                                           t = 1,
                                           beta = 1/2,
                                           kfold = 5){

        res = list()
        for(ii in 1:kfold){
                train.idx = sample(nrow(A), floor((1 - 1/kfold) * nrow(A)))
                A.train = A[train.idx,]
                b.train = b[train.idx]
                A.test = A[-train.idx,]
                b.test = b[-train.idx]
                x.pred = fit_glmnet_fixed_lambda_vanilla_pgd(A.train,
                                                      b.train,
                                                      x0 = x0,
                                                      alpha = alpha,
                                                      lambda = lambda,
                                                      maxiter = maxiter,
                                                      ABSTOL = ABSTOL,
                                                      RELTOL = RELTOL,
                                                      t = t,
                                                      beta = beta)
                params = list(
                        b = b.test,
                        # The response variables
                        A = A.test,
                        # The matrix
                        alpha = alpha,
                        # 1 means lasso
                        lambda = lambda
                ) # the regularization coefficient
                b.pred = predict_exp_glmnet(x.pred, params)
                rse = sqrt(sum((b.pred - b.test)^2))
                res[[ii]] = list(x.pred = x.pred, A.test = A.test, b.test = b.test,
                                 b.pred = b.pred, rse = rse,
                                 A.train = A.train, b.train = b.train)
        }
        rmse = sapply(res, function(x) x$rse)
        rmse = mean(rmse)
        return(list(rmse = rmse, raw.res = res))
}

# fit glm-en with vanillda PGD, use k-fold cv to estimate the prediction error and automatically find optimal lambda

fit_glmnet_search_lambda_vanilla_pgd_cv = function(A,
                                            b,
                                            n.lambda = 100,
                                            min.lambda.ratio = 10e-4,
                                            x0 = rep(0, ncol(A)),
                                            alpha = 0.5,
                                            maxiter = 1000,
                                            ABSTOL = 1e-4,
                                            RELTOL = 1e-2,
                                            t = 1,
                                            beta = 1/2,
                                            kfold = 5){
        lambda_max = find_lambda_max(A, b, alpha)
        lambdas = 10^(seq(log10(lambda_max * min.lambda.ratio), log10(lambda_max), length.out = n.lambda))
        res = list()
        rmse_vector = rep(0, n.lambda)
        for(ii in 1:n.lambda){
                res[[ii]] = fit_glmnet_fixed_lambda_vanilla_pgd_cv(A,
                                                            b,
                                                            lambda = lambdas[ii],
                                                            x0 = x0,
                                                            alpha = alpha,
                                                            maxiter = maxiter,
                                                            ABSTOL = ABSTOL,
                                                            RELTOL = RELTOL,
                                                            t = t,
                                                            beta = beta,
                                                            kfold = kfold)
                rmse_vector[ii] = res[[ii]]$rmse
        }
        best.idx = which.min(rmse_vector)
        best.lambda = lambdas[best.idx]
        best.x = fit_glmnet_fixed_lambda_vanilla_pgd(A,
                                              b,
                                              x0 = x0,
                                              alpha = alpha,
                                              lambda = best.lambda,
                                              maxiter = maxiter,
                                              ABSTOL = ABSTOL,
                                              RELTOL = RELTOL,
                                              t = t,
                                              beta = beta)
        return(list(best.x = best.x,
                    best.idx = best.idx,
                    best.lambda = best.lambda,
                    rmse = rmse_vector,
                    cv.res = res
        ))
}




find_lambda_max = function(A,
                           b,
                           alpha){
        return(max(abs((b-1)%*%A))/alpha)
}


# find the smallest enet_lambda that garuantees zero solution

find_lambda_max = function(A,
                           b,
                           alpha,
                           scale = 10){
        return(max(abs((b-1)%*%A))/alpha/scale)
}

# generate vector of candidate lambdas

generate_lambda_grid = function(A, b, alpha,
                                n.lambda = 100,
                                min.lambda.ratio = 1e-4
                                ){
        lambda_max = find_lambda_max(A, b, alpha)
        lambdas = 10^(seq(log10(lambda_max * min.lambda.ratio), log10(lambda_max), length.out = n.lambda))
}



######### generate test data
set.seed(20170820)

nobs = 50
nvars = 7

x.true = rnorm(nvars)
x.true[sample(2:nvars, floor(nvars / 2) - 1)] = 0

## random normal matrix A
A = matrix(rnorm(nvars * nobs), ncol = nvars)

# ## nobs same row
# A = matrix(rnorm(nvars), ncol = nvars, nrow = nobs, byrow = TRUE)

## Vandermonde matrix A

A = NULL
for(i in 0:(nvars - 1)){
        A = cbind(A, seq(0.5/nobs, 0.5, by = 0.5/nobs)^i)
}


exp.lambdas = exp(-A %*% x.true)
b = sapply(exp.lambdas, function(x)
        rexp(1, x))


######## use pre-generated data
#### start of the code
# nsim = 100
# train_index = 1 : (nsim * 0.8)
# validate_index = (nsim * 0.8 + 1) : nsim
# A = as.matrix(read.csv("./testdata/VandermondeMatrix.csv"))
# bmat = as.matrix(read.csv("./testdata/b_VandermondeMatrix.csv"))
# x.true = as.matrix(read.csv("./testdata/trueSol.csv"))
# x0 = as.matrix(read.csv("./testdata//startSol.csv"))
# nvars = length(x0)
# b = bmat[,1]

# params = list(b = bmat[,1],  # The response variables
#               A = A,  # The matrix
#               alpha = 1, # 1 means lasso
#               lambda = 0) # the regularization coefficient



####### set parameters
alpha = 0.5
lasso.lambda = 0.1
params = list(
        b = b,
        # The response variables
        A = A,
        # The matrix
        alpha = alpha,
        # 1 means lasso
        lambda = lasso.lambda
) # the regularization coefficient

# parameters for PGD


### use R implementation of vanilla pgd

fit.pgd = fit_glmnet_fixed_lambda_vanilla_pgd(A, b, alpha = alpha,
                                              lambda = lasso.lambda)

### use R implementation of vanilla pgd

fit.pgd.cv = fit_glmnet_fixed_lambda_vanilla_pgd_cv(A, b, alpha = alpha,
                                              lambda = lasso.lambda/10)

fit.pgd.cv$rmse

### use R implementation of vanilla pgd and cross validation

fit.pgd.search.cv = fit_glmnet_search_lambda_vanilla_pgd_cv(A, b, alpha = alpha)
min(fit.pgd.search.cv$rmse)
fit.pgd.search.cv$best.x
fit.pgd.search.cv$best.idx
x.true
glmnet_exp(A, b, alpha = alpha)

#
# ### use bfgs
# fit.bfgs = optim(rep(0, nvars), ObjFun, params = params, method = "BFGS")
#
# ### use h2o
# h2o.init()
# x.h2o.df = as.h2o(data.frame(b, A))
# predictors = colnames(x.h2o.df)[-1]
# response = colnames(x.h2o.df)[1]
# my.glm.lasso = h2o.glm(
#         x = predictors,
#         y = response,
#         family = 'gamma',
#         intercept = FALSE,
#         training_frame = x.h2o.df,
#         ignore_const_cols = TRUE,
#         link = "log",
#         lambda_search = TRUE,
# #        lambda = lasso.lambda,
#         alpha = alpha,
#         standardize = TRUE
# )
# h2o.shutdown(FALSE)
# fit.h2o = my.glm.lasso@model$coefficients
# fit.h2o
# ### compare results
# data.frame(
#         x_true = x.true,
#         x_prox = fit.pgd,
#         x_bfgs = fit.bfgs$par,
#         x_h2o = fit.h2o[-1]
# )

enet_lambdas = generate_lambda_grid(A, b, alpha)
mse_matrix = matrix(0, nrow = length(enet_lambdas), ncol = 3)
h2o.init()

i = 1
for(i in 1:length(enet_lambdas)){
        enet_lambda = enet_lambdas[i]
        params[["lambda"]] = enet_lambda
        ### use R implementation of vanilla pgd

        fit.pgd = fit_glmnet_fixed_lambda_vanilla_pgd(A, b, alpha = alpha,
                                                      lambda = enet_lambda)
        mse.pgd = sum((predict_exp_glmnet(fit.pgd, params) - b)^2)

        ### use bfgs
        fit.bfgs = optim(rep(0, nvars), ObjFun, params = params, method = "BFGS")$par
        mse.bfgs = sum((predict_exp_glmnet(fit.bfgs, params) - b)^2)

        ###use h2o
        x.h2o.df = as.h2o(data.frame(b, A))
        predictors = colnames(x.h2o.df)[-1]
        response = colnames(x.h2o.df)[1]
        my.glm.lasso = h2o.glm(
                x = predictors,
                y = response,
                family = 'gamma',
                intercept = FALSE,
                training_frame = x.h2o.df,
                ignore_const_cols = TRUE,
                link = "log",
                lambda = enet_lambda,
                lambda_search = FALSE,
                alpha = alpha,
                standardize = FALSE
        )
        fit.h2o = my.glm.lasso@model$coefficients
        mse.h2o = sum((predict_exp_glmnet(fit.h2o, params) - b)^2)

        mse_matrix[i,] = c(mse.pgd, mse.bfgs, mse.pgd)
}
mse_matrix
fit.pgd = fit_glmnet_fixed_lambda_vanilla_pgd(A, b, alpha = alpha,
                                              lambda = tail(enet_lambdas,1))

fit.bfgs
# h2o.shutdown(FALSE)
x.true
sum((exp(A%*%x.true) - b)^2)
best.idx = which.min(mse_matrix[,1])
fit.pgd = fit_glmnet_fixed_lambda_vanilla_pgd(A, b, alpha = alpha,
                                    lambda = enet_lambdas[best.idx])

params.bfgs = params
params.bfgs[["lambda"]] = enet_lambdas[best.idx]
fit.bfgs = optim(rep(0, nvars), ObjFun, params = params.bfgs, method = "BFGS")$par
fit.bfgs

h2o.init()
x.h2o.df = as.h2o(data.frame(b, A))
predictors = colnames(x.h2o.df)[-1]
response = colnames(x.h2o.df)[1]
my.glm.lasso = h2o.glm(
        x = predictors,
        y = response,
        family = 'gamma',
        intercept = TRUE,
        training_frame = x.h2o.df,
        ignore_const_cols = TRUE,
        link = "log",
#        lambda = enet_lambda,
        lambda_search = TRUE,
        alpha = alpha,
        standardize = FALSE
)
fit.h2o = my.glm.lasso@model$coefficients
fit.h2o
### compare results
data.frame(
        x_true = x.true,
        x_prox = fit.pgd,
        x_bfgs = fit.bfgs,
        x_h2o = fit.h2o,
        x_prox_cv = fit.pgd.search.cv$best.x
)
h2o.shutdown(FALSE)

### test cpp NLL of exp
ExpNegativeLogLikelihood(x.true, params)
ExpNegativeLogLikelihood_cpp(x.true, params[["A"]], params[["b"]])


### test cpp gradient of NLL of exp
GradExpNegativeLogLikelihood(x.true, params)
GradExpNegativeLogLikelihood_cpp(x.true, params[["A"]], params[["b"]])

### test fit glmnet_cpp with fixed lambda
fit.pgd = fit_glmnet_fixed_lambda_vanilla_pgd(params[["A"]], params[["b"]],
                                              alpha = params[["alpha"]],
                                              lambda = enet_lambdas[best.idx])
fit.pgd
ProxGradDescent_cpp(params[["A"]], params[["b"]],
                    enet_lambdas[best.idx], params[["alpha"]])
x.true


### test fit glmnet_cpp with lambda searching
fit.pgd.cpp = fitGlmCv(params[["A"]], params[["b"]],
         alpha = params[["alpha"]])

data.frame(
  x_true = x.true,
  x_prox = fit.pgd,
  x_bfgs = fit.bfgs,
  x_h2o = fit.h2o,
  x_prox_cv = fit.pgd.search.cv$best.x,
  x_prox_cv_cpp = fit.pgd.cpp
)

library(microbenchmark)

nobs = 100
params_list = list()
for(j in 1:1000){
A = NULL
for(i in 0:(nvars - 1)){
  A = cbind(A, seq(0.5/nobs, 0.5, by = 0.5/nobs)^i)
}


exp.lambdas = exp(-A %*% x.true)
b = sapply(exp.lambdas, function(x)
  rexp(1, x))

####### set parameters
alpha = 0.5
lasso.lambda = 0.1
params_list[[j]] = list(
  b = b,
  # The response variables
  A = A,
  # The matrix
  alpha = alpha,
  # 1 means lasso
  lambda = lasso.lambda
) # the regularization coefficient
}
str(params_list)

r_start = Sys.time()
r_res_list = lapply(params_list,
       function(x)
                fit_glmnet_search_lambda_vanilla_pgd_cv(x[["A"]], x[["b"]])$best.x)
r_end = Sys.time()
r_time = r_end - r_start

cpp_start = Sys.time()
cpp_res_list = lapply(params_list,
       function(x)
         fitGlmCv(x[["A"]], x[["b"]], x[["alpha"]]))
cpp_end = Sys.time()
cpp_time = cpp_end - cpp_start
c(r_time, cpp_time)

h2o.init()

h2o_start = Sys.time()

h2o_res_list = lapply(params_list,
                      function(dat){

b = dat[["b"]]
A = dat[["A"]]
alpha = dat[["alpha"]]
x.h2o.df = as.h2o(data.frame(b, A))
predictors = colnames(x.h2o.df)[-1]
response = colnames(x.h2o.df)[1]
my.glm.lasso = h2o.glm(
  x = predictors,
  y = response,
  family = 'gamma',
  intercept = TRUE,
  training_frame = x.h2o.df,
  ignore_const_cols = TRUE,
  link = "log",
  #        lambda = enet_lambda,
  lambda_search = TRUE,
  alpha = alpha,
  standardize = FALSE
)
return(my.glm.lasso@model$coefficients)
                      }
)
h2o_end = Sys.time()
h2o_time = h2o_end - h2o_start
h2o_time
cpp_time
r_time
h2o_res_list
cpp_res_list
x.true

compute_mean_vector = function(data_list){
  tmp_mat = matrix(unlist(data_list), nrow = length(data_list[[1]]))
  apply(tmp_mat, 1, mean)
}
mean_r_res = compute_mean_vector(r_res_list)
mean_cpp_res = compute_mean_vector(cpp_res_list)
mean_h2o_res = compute_mean_vector(h2o_res_list)
data.frame(x.true = x.true,
           x.r = mean_r_res,
           x.h2o = mean_h2o_res,
           x.cpp = mean_cpp_res)

#### try doing things parallel using foreach
#### note that h2o.glm does not work with foreach
library(doParallel)
library(foreach)
cl <- makeCluster(3)
registerDoParallel(cl)
foreach(i=1:3) %dopar% sqrt(i)


system.time({cpp_foreach_res_list = foreach(i = 1:length(params_list),
                               .packages = 'glmnetRcpp') %do% {
                                 x = params_list[[i]]
                                 fitGlmCv(x[["A"]], x[["b"]], x[["alpha"]])
}
})

system.time({cpp_foreach_res_list = foreach(i = 1:length(params_list),
                                            .packages = 'glmnetRcpp') %dopar% {
                                              x = params_list[[i]]
                                              fitGlmCv(x[["A"]], x[["b"]], x[["alpha"]])
                                            }
})
stopCluster(cl)
compute_mean_vector(cpp_foreach_res_list)
