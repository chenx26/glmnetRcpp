#' fit glmnet model for exponetially distributed response data
#'
#' @param A The matrix of independent variables
#' @param b The vector of response variables
#' @param alpha the coefficient of elastic net regularizer (1 means lasso)
#' @param num_lambda size of the lambda grid
#' @param glm_type type of glm model, 1 is exponential, 2 is gamma (not implemented yet)
#' @param max_iter max number of iteration for the prox grad descent optimizer
#' @param abs_tol absolute error threshold for the pgd optimizer
#' @param rel_tol relative error threshold for the pgd optimizer (not used for vanilla PGD)
#' @param normalize_grad swtich for whether to normalize the gradient or not
#' @param k_fold the number of folds for cross validation
#'
#' @return vector of optimal coefficient for the glm model
#' @export
glmnet_exp = function(A,
                      b,
                      ...,
                      alpha.EN = 0.5,
                      num_lambda = 100L,
                      glm_type = 1L,
                      max_iter = 100L,
                      abs_tol = 1.0e-4,
                      rel_tol = 1.0e-2,
                      normalize_grad = FALSE,
                      k_fold = 5L,
                      has_intercept = TRUE,
                      k_fold_iter = 5L){
  return(fitGlmCv(A,
         b,
         alpha = alpha.EN,
         num_lambda = num_lambda,
         glm_type = glm_type,
         max_iter = max_iter,
         abs_tol = abs_tol,
         rel_tol = rel_tol,
         normalize_grad = normalize_grad,
         k_fold = k_fold,
         has_intercept = has_intercept,
         k_fold_iter = k_fold_iter))
}
