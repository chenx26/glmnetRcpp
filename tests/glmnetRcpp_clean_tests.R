rm(list = ls())

######### load package
library(glmnetRcpp)
library(h2o)

######### helper function
compute_mean_vector = function(data_list){
  tmp_mat = matrix(unlist(data_list), nrow = length(data_list[[1]]))
  apply(tmp_mat, 1, mean)
}

######### set parameters
set.seed(20170827)
nobs = 50
nvars = 7
ntests = 10
alpha = 0.5
lasso.lambda = 0.1
x.true = rnorm(nvars)
x.true[sample(2:nvars, floor(nvars / 2) - 1)] = 0

######## generate test data

params_list = list()
for(j in 1:ntests){
  ## random normal matrix A
  A = matrix(rnorm(nvars * nobs), ncol = nvars)
  exp.lambdas = exp(-A %*% x.true)
  b = sapply(exp.lambdas, function(x) rexp(1, x))
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

#### try doing things parallel using foreach
#### note that h2o.glm does not work with foreach
library(doParallel)
library(foreach)

h2o.init()
h2o_time_single = system.time({h2o_single_res_list = lapply(params_list,
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
                                                         intercept = FALSE,
                                                         training_frame = x.h2o.df,
                                                         ignore_const_cols = TRUE,
                                                         link = "log",
                                                         #        lambda = enet_lambda,
                                                         lambda_search = TRUE,
                                                         alpha = alpha,
                                                         standardize = FALSE
                                                       )
                                                       return(my.glm.lasso@model$coefficients[-1])
                                                     }
)
}
)
h2o.shutdown(FALSE)

cl <- makeCluster(3)
registerDoParallel(cl)

time_single = system.time({cpp_single_res_list = foreach(i = 1:length(params_list),
                                            .packages = 'glmnetRcpp') %do% {
                                              x = params_list[[i]]
                                              fitGlmCv(x[["A"]], x[["b"]], x[["alpha"]])
                                            }
})

time_multi = system.time({cpp_multi_res_list = foreach(i = 1:length(params_list),
                                            .packages = 'glmnetRcpp') %dopar% {
                                              x = params_list[[i]]
                                              fitGlmCv(x[["A"]], x[["b"]], x[["alpha"]])
                                            }
})
stopCluster(cl)
res_cpp_single = compute_mean_vector(cpp_single_res_list)
res_cpp_multi = compute_mean_vector(cpp_multi_res_list)
res_h2o_single = compute_mean_vector(h2o_single_res_list)

### timings
data.frame(cpp_time_single = time_single[3],
           cpp_time_multi = time_multi[3],
           h2o_time_single = h2o_time_single[3])

### fitted coefficients
data.frame(x.true = x.true,
           x.single = res_cpp_single,
           x.multi = res_cpp_multi,
           x.h2o = res_h2o_single)

