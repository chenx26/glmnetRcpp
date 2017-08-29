# DESCRIPTION
This is an efficient Rcpp implementation of the glmnet model for exponentially distributed response variables.

## Proposed GSOC 2017 Standard Errors Project Deliverables

In the proposal for the GSoC project for standard errors 2017, the following 7 deliverables are included:
1. Procedures to compute the following: the negative log-likelihood of exponential distribution, the gradient of the negative log-likelihood of exponential distribution, the proximal gradient function of the elastic net regularization, the negative log-likelihood of gamma distribution (thereby extending the use of the method to the entire gamma family of distributions), the gradient of the negative log-likelihood of gamma distribution.
2. Procedures to perform optimization of GLM-EXP/EN given λ and α.
3. Procedures to perform model selection for λ and α .
4. Compare the results of GLM-EXP/EN with vanilla GLM-EXP, i.e., to find out how much the EN provides.
5. Test results of speed and accuracy of the new procedure compared with the H2O implementation.
6. Implementation with the risk and performance measure functions in PerformanceAnalytics.
7. Use our implementation method above, modified as needed, to allow use of nse methods in PerformanceAnalytics.

## Status of Proposed Deliverables

Of the 7 proposed deliverables, the first 5 have been completed. See the GitHub repo https://github.com/chenx26/glmnetRcpp for the completed package. 

For the last two deliverables, preliminary results have been implemented for sample mean and Sharpe ratio and the performances of our new method have been satisfactory. See section 2.4 and 2.5 of the vignette at https://github.com/chenx26/glmnetRcpp/blob/master/vignettes/glmnetRcpp_vignette.pdf. I expect to complete these deliverables during September2017.


# INSTALLATION
Sys.setenv( "PKG_CXXFLAGS"="-std=c++14" )

libary(devtools)

install_github("chenx26/glmnetRcpp")

# Vignette
See the Vignette folder for timings and accuracy

