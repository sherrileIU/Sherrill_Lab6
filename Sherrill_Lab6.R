### Elizabeth Sherrill
### Statistics S-610
### Homework 7
### Due Friday, November 13th, 2020
library(microbenchmark)

p = c(10,50,100,200,500)
n = 2000

mean = 0
sd = 1
epsilon = rnorm(n, mean, sd)

X = list()
beta = list()
y = list()

for (i in 1:length(p)) {
	X[[ i ]] = matrix(rnorm(n*p[ i ], mean, sd), n, p[ i ])
	beta[[ i ]] = rep(1,p[ i ])
	y[[ i ]] = (X[[ i ]] %*% beta[[ i ]]) + epsilon 
}

######## Newton's method ############################### 
newton_step = function (X, y) {
	newton = solve(t(X) %*% X) %*% t(X) %*% y
	return(newton)
}

newton_mbm = microbenchmark(
NM_10 = newton_step(X[[1]], y[[1]]),
NM_50 = newton_step(X[[2]], y[[2]]),
NM_100 = newton_step(X[[3]], y[[3]]),
NM_200 = newton_step(X[[4]], y[[4]]),
NM_500 = newton_step(X[[5]], y[[5]]),
times = 20L)

newton_mean_runtime = summary(newton_mbm)[[4]]
#plot(newton_mean_runtime ~ p)
#plot(newton_mean_runtime^(1/3) ~ p)

######## Gradient descent ############################### 
neg_loglik = function (beta, y, X) {
	negloglik =   0.5 * (t(y - (X %*% beta)) %*% (y - (X %*% beta)))
	return(negloglik)
}
### returns scalar 1 x 1 value

neg_deriv = function (beta, y, X) {
	negderiv = t(X) %*% (y - (X %*% beta))
	return(negderiv)
}
### returns p x 1 vector


backtracking = function (fn, grad, beta, a, b, X, y) {
	t = 1
	fn_current = fn(beta, y, X)
	grad_current = grad(beta, y, X)
	while(fn(beta + t * grad_current, y, X) > (fn_current + a * t * t(grad_current) %*% grad_current)) {
		t = b * t
	}
	return(t)
}
### returns 1 x 1 scalar

backtrack_desc = function (fn, grad, start, a, b, X, y) {
	beta = start
	step_size = backtracking(fn, grad, beta, a, b, X, y)
	new_beta = beta + step_size * grad(beta, y, X)
	return(new_beta)
}
### returns p x 1 new beta vector


gradient_desc_mbm = microbenchmark(
GD_10 = backtrack_desc(neg_loglik,neg_deriv,start = beta[[1]], a = 0.1, b = 0.2, X = X[[1]], y = y[[1]]),
GD_50 = backtrack_desc(neg_loglik,neg_deriv,start = beta[[2]], a = 0.1, b = 0.2, X = X[[2]], y = y[[2]]),
GD_100 = backtrack_desc(neg_loglik,neg_deriv,start = beta[[3]], a = 0.1, b = 0.2, X = X[[3]], y = y[[3]]),
GD_200 = backtrack_desc(neg_loglik,neg_deriv,start = beta[[4]], a = 0.1, b = 0.2, X = X[[4]], y = y[[4]]),
GD_500 = backtrack_desc(neg_loglik,neg_deriv,start = beta[[5]], a = 0.1, b = 0.2, X = X[[5]], y = y[[5]]),
times = 20L)


grad_desc_mean_runtime = summary(gradient_desc_mbm)[[4]]
#plot(grad_desc_mean_runtime ~ p)
