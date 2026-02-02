# Variable Selection

We have already seen times when we have more predictors than optimal for predicting the response. We discussed an *ad hoc* method of variable selection using $p$-values, which can be useful for explanatory model building. We also looked at projecting the predictors onto a smaller subspace using PCA, which works about as well as projecting randomly onto subspaces in many cases. 

The issue with PCA regression is that the choice of directions of maximal variance may have nothing to do with the response. It **could* be that the predictive value of the predictors is in a direction that has small variance relative to other directions, in which case it would be hidden in one of the later components. For this reason, we will want to consider the relationship of the components in the predictors with the response. That is the idea behind partial least squares.

In the next section, we will consider ridge and lasso regression. These methods attempt to minimize an error function that also includes the magnitudes of the coefficients. The thought is that if a predictor doesn't have much predictive power, then the penalty associated with the coefficient will force the coefficient to be zero, which is the same as eliminating it from the model.

Next, we will talk about the Akaike Information Criterion, which is a classical way of variable selection that also performs well in predictive models.

Finally, we will take a **long** overdue detour and talk about interactions. Interactions really belonged in the classical theory section, but are also useful for predictive modeling. We will discuss their use both in explanatory model building and predictive model building.

## Partial Least Squares

As mentioned above, partial least squares first finds the direction in the predictor space that has maximum covariance with the response. Let's do that using simulations so that we can understand what is going on. Our data is also simulated data.





``` r
set.seed(2162020)
Sigma <- matrix(rnorm(9), nrow = 3)
Sigma <- Sigma %*% t(Sigma)
dd <- as.data.frame(MASS::mvrnorm(500, mu = c(0,0,0), Sigma = Sigma))
names(dd) <- c("x", "y", "z")
cor(dd)
```

```
##            x          y          z
## x  1.0000000  0.3970130 -0.4702539
## y  0.3970130  1.0000000 -0.9928534
## z -0.4702539 -0.9928534  1.0000000
```

``` r
dd <- mutate(dd, response = 2 * (x + 4 *y) + .25 * (x + 6*y + z) + rnorm(500, 0, 7))
summary(lm(response ~., data = dd))
```

```
## 
## Call:
## lm(formula = response ~ ., data = dd)
## 
## Residuals:
##      Min       1Q   Median       3Q      Max 
## -20.1504  -4.6839  -0.2462   4.6686  19.8194 
## 
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept)  -0.1148     0.3203  -0.358 0.720159    
## x             2.3100     0.6733   3.431 0.000652 ***
## y            12.5728     3.1097   4.043 6.11e-05 ***
## z             4.3116     3.9722   1.085 0.278256    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 7.147 on 496 degrees of freedom
## Multiple R-squared:  0.7119,	Adjusted R-squared:  0.7101 
## F-statistic: 408.5 on 3 and 496 DF,  p-value: < 2.2e-16
```

We begin by centering and scaling the predictors.


``` r
dd <- dd %>% mutate(x = scale(x),
             y = scale(y),
             z = scale(z))
```

Next, as in the case of PCA, we will use `optim` to maximize the covariance of the projection of the data set onto a direction on the unit sphere in $R^3$. We begin with some helper functions, which are the inner product of two vectors and the norm of a vector.


``` r
ip <- function(x, y) {
  sum(x * y)
}
mynorm <- function(x) {
  sqrt(sum(x^2))
}
```

Next, we see how to take a random sample from the sphere, project the data onto that direction, and compute the covariance with the response.


``` r
rvec <- rnorm(3)
rvec <- rvec/mynorm(rvec)
dd_p <- apply(dd[,1:3], 1, function(x) ip(x, rvec))
cov(dd_p, dd$response)
```

```
## [1] 0.3228743
```

Now we put this inside of `optim`. We normalize the vector and take the negative absolute value of the covariance in order to insure that the *minimum* found corresponds to the *maximum covariance*.


``` r
ff <- function(rvec) {
  rvec <- rvec/mynorm(rvec)
  dd_p <- apply(dd[,1:3], 1, function(x) ip(x, rvec))
  -abs(cov(dd_p, dd$response))
}
first_comp <- optim(par = c(-.33,-.66, .66), fn = ff, control = list(reltol = .000001))
first_comp
```

```
## $par
## [1] -0.3346406 -0.6749531  0.6739219
## 
## $value
## [1] -16.67199
## 
## $counts
## function gradient 
##       30       NA 
## 
## $convergence
## [1] 0
## 
## $message
## NULL
```

``` r
first_comp$par/mynorm(first_comp$par)
```

```
## [1] -0.3310651 -0.6677415  0.6667212
```

According to this, our estimate for the first PLS component is -0.331, -0.668, 0.667. This is the direction of the data that has the largest covariance with the response. Let's check it against the first component found using the `plsr` function in the `pls` package.


``` r
library(pls)
```

```
## 
## Attaching package: 'pls'
```

```
## The following object is masked from 'package:caret':
## 
##     R2
```

```
## The following object is masked from 'package:stats':
## 
##     loadings
```

``` r
plsmod <- plsr(response ~ x + y + z, data = dd)
loadings(plsmod)
```

```
## 
## Loadings:
##   Comp 1 Comp 2 Comp 3
## x  0.401 -0.942  0.100
## y  0.645  0.281  0.678
## z -0.655 -0.186  0.728
## 
##                Comp 1 Comp 2 Comp 3
## SS loadings     1.006  1.002  1.000
## Proportion Var  0.335  0.334  0.333
## Cumulative Var  0.335  0.669  1.002
```

``` r
plsmod$projection
```

```
##       Comp 1     Comp 2     Comp 3
## x  0.3306078 -0.9135712 0.06426708
## y  0.6676654  0.3579067 0.69202962
## z -0.6670243 -0.2072125 0.72009477
```

We see that the first component that `plsr` got in the `projection` is $(0.331, 0.668, -0.6667)$, which is the negative of what we got, so gives the same line or direction. 

Now, in PCA to get the second component, we looked at all directions perpendicular to the first component and found the one with maximum variance. In PLS, we no longer force the second component to be orthogonal to the first component. That's one difference. But, we do want it to pick up different information than the first component. How do we accomplish that?

A first idea might be to find the direction that has the highest covariance with the **residuals** of the response after predicting on the directions chosen so far. That is pretty close to what happens, but doesn't work exactly. In partial least squares, we first **deflate** the predictor matrix by subtracting out $d d^T M$, where $d$ is the first direction, $d^T$ is the transpose of $d$, and $M$ is the original matrix. Then, we find the direction of maximum covariance between the residuals of the first model and this new matrix. I include the details below, but this section is **OPTIONAL**.

### Optional PLS Stuff































































































