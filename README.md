<h1><div style="text-align: center;"> Coordinate Descent Algorithms for Lasso Penalized Regression[1] </div></h1>
<h4><div style="text-align: center;">By: Aryan Gandhi</div></h4>

<br>

# 1. Introduction

This paper provides an overview of the research paper "Coordinate Descent Algorithms for Lasso Penalized Regression" by Tong Tong Wu and Kenneth Lange.

<br>

## What is Lasso Regression?

Lasso regression is a type of regularized regression that combines the loss function with the $\ell_1$ norm penalty. The purpose of the $\ell_1$ norm penalty is to set some of the coefficient estimates to zero, thus allowing for feature selection and improved interpretability of the model. This is opposed to ridge regression, which uses the $\ell_2$ norm penalty, which does not set coefficients to zero. Therefore, lasso regression can be useful when we have a large number of features and want to select only a few that are most important [2].

<br>

## Lasso Penalized Regression

The objective function for Lasso penalized regression is:
$$f(\theta) = g(\theta) + \lambda \sum_{j=1}^p |\beta_j|$$

where $g(\theta)$ is the loss function.

<br>

Loss function for $\ell_1$ regression:

$$g(\theta) = \min_{\beta}\sum_{i=1}^n | y_i - \mu - \sum_{j=1}^p x_{ij}\beta_j |$$

Loss function for $\ell_2$ regression: $$g(\theta) = \min_{\beta} \frac{1}{2} \sum_{i=1}^n ( y_i - \mu - \sum_{j=1}^p x_{ij}\beta_j )^{2}$$

where $\mu$ is the intercept, $\beta$ is the vector of regression coefficients, $x_i$ is the $i^{th}$ observation, $y_i$ is the $i^{th}$ response value, $n$ is the number of observations, $p$ is the number of predictors and $\lambda$ is a penalty tuning parameter. The lasso penalty encourages sparsity in $\beta$, i.e., some of its elements shrink to zero.

<br>

## Two Main Issues with Lasso Penalized Estimation

1. What is the most effective method of minimizing the objective function?
2. How does one choose the tuning parameter $\lambda$

   To address first issue standard regression methods involve matrix diagonalization, matrix inversion, or, at the very least, the solution of large systems of linear equations.

   Here are the pros and cons of these existing methods:

   ### Pros:

   - Has a closed form

   ### Cons:

   - Computationally expensive due to large matrix operations
   - Instability in high dimensional settings
   - Does not perform model selection

Although the existing methods have closed-form solutions, they suffer from computational inefficiency due to large matrix operations, and they lack the simplicity, speed, and stability of the coordinate descent algorithms presented in this paper. Furthermore, the coordinate descent algorithms excel in high-dimensional settings where the number of predictors (p) far exceeds the number of observations (n). As a result, the limitations of the traditional methods can be significantly mitigated by employing the innovative algorithms introduced in this paper.

To address the second issue cross-validation can be used and the paper introduces an efficient method to achieve an optimal lambda through cross-validation in section 6.

<br>

# 2. Cyclic coordinate descent for $\ell_2$ regression

The Cyclic Coordinant Descent (CCD) algorithm is a simple and efficient optimization method that updates one coordinate at a time while keeping the others fixed. In other words, the algorithm cycles through the parameters and updates each in turn until the algorithm converges to minima for the objective function. This algorithm works well for $\ell_2$ regression as the loss function $g(\theta)$ is differentiable and convex.

<br>

## Algorithm for Cyclic Coordinate Descent

The main steps of the cyclic coordinate descent algorithm for Lasso penalized regression are as follows:

1.  Initialize the coefficients.
2.  For each coefficient in the model:
    - Keep all other coefficients fixed.
    - Update the current coefficient by minimizing the Lasso objective function with respect to this single coefficient.
3.  Check for convergence. If improvement cannot be made in any direction then the objective function has converged to a minima. If the objective function has not converged, go back to step 2.

<br>

## <u>Update Rules for each Parameter</u>

<br>

### Update rule for $\mu$:

$$\hat{\mu} = \frac{1}{n} \sum_{i=1}^n ( y_i - x_i^{t}\beta) = \mu - \frac{1}{n} \frac{\partial}{\partial \mu} g(\theta)$$

where, $\frac{\partial}{\partial \mu} g(\theta) = -\sum_{i=1}^n r_i$

<br>

### Update rule for each coefficient $\beta_k$:

$$\hat{\beta}_{k,-} = \min (0,\beta_k - \frac{\frac{\partial}{\partial \beta_k} g(\theta) - \lambda} {\sum_{i=1}^n x_{ik}^2})$$

$$\hat{\beta}_{k,+} = \max (0,\beta_k - \frac{\frac{\partial}{\partial \beta_k} g(\theta) + \lambda} {\sum_{i=1}^n x_{ik}^2})$$

where, $\frac{\partial}{\partial \beta_k} g(\theta) = -\sum_{i=1}^n r_i x_{ik}$

Note, only one of $\hat{\beta}_{k,-}$ or $\hat{\beta}_{k,+}$ can be nonzero

<br>

## Pros and Cons of Cyclic Coordinate Descent

### Pros:

- Often has faster convergence for $\ell_2$ regression than by greedy coordinate descent
- Handles high-dimensional settings well (p>>n)
- Good at picking relevant predictors
- Computationally efficient as it does not rely on any matrix operations

### Cons:

- Does not have a closed form
- May converge to an inferior local minima instead of a global minima
- Often has slower convergence for $\ell_1$ regression than by greedy coordinate descent

<br>

Although cyclic coordinate descent has it's cons, it's efficiency and numeric stability for $\ell_2$ regression is undeniable. Cyclic coordinate descent for $\ell_2$ regression often works well in practice.

<br>

# 3. Greedy coordinate descent for $l_1$ regression

In greedy coordinate descent, we update the parameter $\theta_k$ giving the most negative value of min{$df_{e_k} (\theta), df_{-ek} (\theta)$}. If none of the coordinate directional derivatives are negative, then no further progress can be made. The greedy coordinate descent algorithm is efficient for $\ell_1$ regression as it capitalizes on the benefits of Edgeworth's Algorithm.

<br>

## Edgeworth's Algorithm

Edgeworth's algorithm leverges the connection between the $\ell_1$ regression loss function $g(\theta) =  \sum_{i=1}^{n} |y_i - \mu - x_i\beta|$ and the roperties of medians to minimize the objective function in an efficient manner.

<br>

### Connction between median and $\ell_1$ regression

$\ell_1$ regression minimizes the sum of absolute differences between the observed responses and the predicted responses. Similarly, the median is the value that minimizes the sum of absolute differences between itself and the data points. Thus we can consider the problem of minimizing $g(\theta) = \sum_{i=1}^{n} |y_i - \mu - x_i\beta|$ as a problem of finding a set of median values.

<br>

### <u>Update Rules for each Parameter</u>

<br>

### Update rule for $\mu$:

While keeping $\beta$ fixed, replace $\mu$ with the sample median of the numbers $z_i = y_i - x_i\beta$

<br>

### Update rule for $\beta$:

For a single predictor model we write the loss function $g(\theta)$:

$$g(\theta) = \sum_{i=1}^{n} |x_i||\frac{y_i - \mu}{x_i} - \beta|$$

by sorting the numbers $z_i = \frac{y_i - x_i}{x_i}$ and finding the weighted median with weight $w_i = |x_i|$ assigned to $z_i$ we are able to minimize $g(\theta)$.

We update $\beta$ by replacing it with the order statistic $z_[i]$ whose index $i$ satisfies:

$$\sum_{j=1}^{i-1} w_{[j]} < \frac{1}{2} \sum_{j=1}^{n} w_{[j]}$$
and
$$\sum_{i}^{j=1} w_{[j]} \geq \frac{1}{2} \sum_{j=1}^{n} w_{[j]}$$

For models with more than one predictor, we update $\beta_k$ like so:

$$g(\theta) = \sum_{i=1}^{n} |x_{ik}||\frac{y_i - \mu - \sum_{j=k}x_{ij}\beta_k}{x_{ik}} -\beta|$$

<br>

## Pros and Cons of Greedy Coordinate Descent

### Pros:

- Usually converges faster for $\ell_1$ regression than by cyclic coordinate descent
- Handles high-dimensional settings well (p>>n)
- Good at picking relevant predictors
- Computationally efficient as it does not rely on any matrix operations

### Cons:

- Does not have a closed form
- May converge to an inferior local minima instead of a global minima. [4]
- Convergence often occurs in a slow seesaw pattern

<br>

Although these concerns appear significant, the algorithm generally performs well on a majority of real-world problems. It is also efficient in identifying relevant predictors and typically converges faster than cyclic coordinate descent for $\ell_2$ regression. In the case of $\ell_1$ regression, greedy coordinate descent significantly outperforms cyclic coordinate descent in terms of speed. This is likely due to greedy coordinate descent prioritizing the update of important predictors first which leads to quicker convergence. Thus, in practice the greedy coordinate descent algorithm is likely an excellent choice when performing $\ell_1$ regression.

<br>

# 4. Convergence of the algorithms

The convergence of Edgeworth's algorithm and cyclic coordinate descent algorithms is an important topic. Previously in the paper, it was mentioned that Edgeworth's algorithm may not converge to a minimum point. For cyclic coordinate descent, textbook treatments assume that the objective function is continuously differentiable, which might not hold for non-differentiable functions like the lasso penalty [5]. For non-differentiable functions, it is possible that not all directional derivatives are nonnegative at the minimum point as required. In other words, although the function should not decrease in any direction at the minimum point, it might still do so for non-differentiable functions.

It would be beneficial to determine a straightforward sufficient condition for a unique minimum point. Uniqueness is commonly demonstrated by showing strict convexity of the objective function. However, this might not hold for underdetermined problems involving lasso penalties because strict convexity can fail. Nevertheless, strict convexity is not required for a unique minimum. According to a discussion with Emanuel Candes, the authors hypothesize that with respect to Lebesgue measure, almost all design matrices result in a unique minimum.

<br>

# 5. $l_2$ regression with group penalties

This section discusses handling grouped effects in linear regression by introducing group penalties. Grouped effects occur when certain predictor variables naturally belong to specific groups, and the objective is to apply coordinated penalties to include or exclude all parameters within a group.

The ideal group penalty is the scaled Euclidean norm $\lambda || \gamma_j||_2$, which couples the parameters, preserves convexity, and works well with cyclic coordinate descent in $\ell_2$ regression. It has a grouping tendency because the directional derivative along the coordinate vector drops from $1$ to $0$ when any parameter within the group is nonzero, making it easier for other parameters within the same group to move away from $0$.

The recommended objective function for $\ell_2$ regression with grouping effects is a combination of $\ell_1$ and $\ell_2$ penalties:

$$f(\theta) = g(\theta) + \lambda_2 * \sum_{j=1}^{q}||\gamma_j||_2 + \lambda_1 * \sum_{j=1}^{q}||\gamma_j||_1$$

where g($\theta$) is the residual sum of squares.

When updating a parameter, if all other parameters in its group are $0$, the standard $\ell_2$ regression update with a Lasso penalty applies. If at least one other parameter in the group is nonzero, majorization is used to create a surrogate function, which is a quadratic plus a Lasso penalty. Minimizing the surrogate function guarantees that the objective function will decrease. Although, the structure of updating parameters is consistent with the same procedure as before.

<br>

## Surrogate Function

$||\gamma_j||_2 < ||\gamma_j^{m}||_2 + \frac{1}{2||\gamma_j^{m}||_2}(||\gamma_j||_2^{2} - ||\gamma_j^{m}||_2)$

<br>

# 6. Selection of the tuning constant $\lambda$

This section explores the process of choosing the tuning constant $\lambda$. A popular method for selecting $\lambda$ is cross-validation, which can be done by examining the cross-validation error curve and selecting the optimal lambda that minimizes the error. K-fold cross-validation partitions the data into k equal subsets and estimates parameters k times, excluding one subset at a time. By averaging the testing error across the k subsets, the cross-validation error is calculated.

Assessing the cross-validation error across a grid of points can be computationally demanding, so techniques like bracketing and golden section search can be employed to speed up the process. A shortcut for determining the optimal $\lambda$ merges bracketing and golden section search. Beginning with a large $\lambda_0$, the algorithm iteratively decreases $\lambda$ by a fixed amount until $c(\lambda_{k+1}) > c(\lambda_k)$ is encountered, establishing $\lambda_{k-1} > \lambda_k > \lambda_{k+1}$, with $\lambda_k$ yielding the lowest value of $c(\lambda)$. Subsequently, the golden section search is applied to minimize $c(\lambda)$ within the range $(\lambda_{k+1}, \lambda_{k-1})$. Initiating the search with a high $\lambda_0$ accelerates the procedure since most coefficients $\beta_j$ are estimated to be $0$ when $\lambda$ values are large.

When dealing with grouped parameters, finding the best pair of tuning parameters ($\lambda_1$, $\lambda_2$) is more challenging. A recommendation is to consider three cases: (a) $\lambda_1 = 0$, (b) $\lambda_2 = 0$, and (c) $\lambda_1 = \lambda_2$, and use bracketing and golden section search for these one-dimensional slices.

When choosing the tuning constant $\lambda$, the initial value of $\theta$ is influenced. For a single $\theta$, it is recommended to set the initial values as $\theta = \bf{0}$ and all residuals $r_i = 0$. As $\lambda$ diminishes, it is expected that current predictors will be maintained and new predictors may be introduced. When estimating $\hat{\theta}$ with a given $\lambda$, it is reasonable to begin with $\hat{\theta}$ and the associated residuals for a slightly smaller $\lambda$ value. This approach leverages existing reliable estimates, decreases the iterations required to achieve convergence, and significantly reduces the total time spent in evaluating the $c(\lambda)$ curve.

<br>

# 7. Analysis of Simulated Data

In this section, we now evaluate the coordinate descent algorithms using simulated data. This simulation will specifically concentrate on evaluating the cyclic coordinate descent algorithm for $\ell_2$ regression in underdetermined settings (p >> n) within a regression model framework [3]. By employing simulated data, we will assess the accuracy of parameter estimates and validate our selection process of the tuning constant $\lambda$.

The simulated data is generated from the regression model:

$$y_i = \mu + \sum_{j=1}^{p} x_{ij}B_{j} + \epsilon_i$$

where we assume the error terms $\epsilon_i$ are independent and follow either a standard normal distribution or a Laplace (double exponential) distribution with scale $1$.

The predictor vectors $x_i$ represent a random sample from a multivariate normal distribution whose marginals are standard normal and whose pairwise correlations are
$$\operatorname{Cov}(X_{ij}, X_{ik}) =\begin{cases}\rho, & \text{if } j \leq 10 \text{ and } k \leq 10, \\0, & \text{otherwise}.\end{cases}$$

The true parameter values are $\mu = 0$, $\beta_j = 1$ for $1 \leq j \leq 5$ and $\beta_j = 0$ for $j > 5$.

Due to the computational limitations of the available resources, the problem sizes of $(p, n) = (5000, 200)$ in the original paper will not be used. Instead problem sizes of $(p, n) = (500,20)$ will be considered. Each senerio will be fitted using $\ell_2$ regression with a lasso penalty and the cyclic coordinate descent algorithm. Addtionally, similar to the original paper, each senerio will be replicated 50 times and the average values for the senerio will be recorded.

<br>

## Simulation Results

![Alt text](code_output.png)

### Column Description:

- $Distribution:$ describes which distribution the error values were sampled from
- $p:$ number of initial predictors
- $n:$ number of data values
- $corr:$ correlation between the predictors as described earlier
- $True\_Error:$ average test error (MSE) of the true model parameters, per replication
- $Lambda:$ average optimal lambda value found
- $Pred\_Error:$ average test error (MSE) of the estimated model parameters, per replication
- $N\_nonzero:$ average number of estimated non-zero parameters, per replication
- $N\_true:$ number of true non-zero model parameters
- $Time:$ average amount of time in seconds, per replication

## Analysis

From the simulation, we see that in each setting cyclic coordinate descent significantly reduces the number of estimated predictors as the initial starting value is 500 while the true number of non-zero predictors is 5. We notice that for both the laplace and normal distribution the number of estimated non-zero predictors is higher when the predictors are correlated. However, the algorithm still significantly reduces the initial large number of predictors to a reasonable amount. Additionally, we see that the average test accuracy for the estimated parameters is slight higher than the true test accuracy. This is most likely because there are a smaller number of data points in this simulation compared to the one in the original paper due to constraints of the available computational resources. Thus, although the exact numbers when comparing this simulation to the one in the original paper are quite different, the overall conclusions are consistent. Cyclic coordinate descent is an efficient algorithm that performs well in high dimensional settings (p >> n) and selects relevant predictors.

<br>

# Conclusion

Lasso penalized regression provides an efficient approach to continuous model selection and estimation, making it an optimal exploratory data analysis tool. It effectively avoids multiple testing problems and benefits from the speed of coordinate descent algorithms, which are well-suited for modern datasets with a much higher number of predictors than data points. Though initially underestimated, coordinate descent methods have proven to be fast due to their avoidance of matrix operations and excluding poor predictors.

However, potential bias in estimating parameters toward zero for small samples requires addressing. This can be done by re-estimating active parameters without the Lasso penalty. Furthermore, deeper understanding of the convergence properties of these algorithms and improving Edgeworth's algorithm without sacrificing speed are still needed. Additionally, Lasso penalized estimation has broader applications including generalized linear models, and ongoing research in logistic regression using cyclic coordinate descent shows promising results [6][7].

# References

[1] Tong Tong Wu, Kenneth Lange "Coordinate descent algorithms for lasso penalized regression," The Annals of Applied Statistics, Ann. Appl. Stat. 2(1), 224-244, (March 2008)

[2] WANG, L., GORDON, M. D. and ZHU, J. (2006a). Regularized least absolute deviations regression and an efficient algorithm for parameter tuning. In Proceedings of the Sixth International Conference on Data Mining (ICDM’06) 690–700. IEEE Computer Society.

[3] WANG, S., YEHYA, N., SCHADT, E. E., WANG, H., DRAKE, T. A. and LUSIS, A. J. (2006b). Genetic and genomic analysis of a fat mass trait with complex inheritance reveals marked sex specificity. PLoS Genet. 2 148–159.

[4] LI, Y. and ARCE, G. R. (2004). A maximum likelihood approach to least absolute deviation regression. EURASIP J. Applied Signal Proc. 2004 1762–1769. MR2131987

[5] RUSZCZYNSKI, A. (2006). Nonlinear Optimization. Princeton Univ. Press. MR2199043

[6] FU, W. J. (1998). Penalized regressions: The bridge versus the lasso. J. Comput. Graph. Statist. 7
397–416. MR1646710

[7] PARK, M. Y. and HASTIE, T. (2006a). L1 regularization path algorithm for generalized linear models. Technical Report 2006-14, Dept. Statistics, Stanford Univ.
