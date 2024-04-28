# Optimization 4: Algorithms SAG-SVRG-Newton-CGD 

---
jupytext:
  cell_metadata_filter: all, -hidden, -heading_collapsed, -run_control, -trusted
  notebook_metadata_filter: all, -jupytext.text_representation.jupytext_version, -jupytext.text_representation.format_version,
    -language_info.version, -language_info.codemirror_mode.version, -language_info.codemirror_mode,
    -language_info.file_extension, -language_info.mimetype, -toc
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
language_info:
  name: python
  nbconvert_exporter: python
  pygments_lexer: ipython3
nbhosting:
  title: Optimization 4: SAG-SVRG-Newton-CGD
  version: '1.0'
---


<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Aymeric DIEULEVEUT</span>
<span>Licence CC BY-NC-ND</span>
</div>

The aim of this material is implement the follogin algorithms, in the same setup as the previous lab, and to provide a global comparison of all the algorithms, interpret rates, etc.

- variance reduced gradient descent (SAG, SVRG)
- newton descent 
- coordinate gradient descent (CD)

applying them on the linear regression and logistic regression models, with ridge penalization.

# Table of content

[1. Introduction](#intro)<br>

[2. Models gradients and losses](#models)<br>

[2.1  Linear regression](#models_regression)<br>
[2.2  Check for Linear regression](#models_regression_check)<br>
[2.3  Logistic regression](#models_logistic)<br>
[2.4  Check for logistic regression](#models_logistic_check)<br>


[3. Solvers](#solvers)<br>

[3.1 Tools for solvers](#tools)<br>
[3.2 Gradient descent](#gd)<br>
[3.3 Stochastic Gradient descent](#sgd)<br>
[3.4 Accelerated Gradient descent](#agd)<br>
[3.5 Heavy ball method](#hb)<br>

### Sections 1 and 2 is similar to the previous lab, you do not have anything to code or answer to.

<a id='intro'></a>
# 1. Introduction

## 1.1. Getting model weights

We'll start by generating sparse vectors and simulating data


```{code-cell} python
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

np.set_printoptions(precision=2) # to have simpler print outputs with numpy
```

## 1.2. Simulation of a linear model


```{code-cell} python
#Import from previous Lab 
from helpers_sim_lin_reg import *
```


```{code-cell} python
n_samples = 500
w0 = np.array([0.5])

X, y = simu_linreg(w0, n_samples=n_samples, corr=0.3, std=0.5)
plt.scatter(X, y)
plt.xlabel(r"$x_i$", fontsize=16)
plt.ylabel(r"$y_i$", fontsize=16)
plt.title("Linear regression simulation", fontsize=18)
plt.scatter(X, y, label='data')
plt.legend()

```

## 1.3. Simulation of a logistic regression model


```{code-cell} python
#Import from previous Lab 
from helpers_sim_log_reg import *
```


```{code-cell} python
n_samples = 500
w0 = np.array([-3, 3.])

X, y = simu_logreg(w0, n_samples=n_samples, corr=.4)

plt.scatter(*X[y == 1].T, color='b', s=10, label=r'$y_i=1$')
plt.scatter(*X[y == -1].T, color='r', s=10, label=r'$y_i=-1$')
plt.legend(loc='upper left')
plt.xlabel(r"$x_i^1$", fontsize=16)
plt.ylabel(r"$x_i^2$", fontsize=16)
plt.title("Logistic regression simulation", fontsize=18)

```

<a id='models'></a>
# 2. Models gradients and losses

We want to minimize a goodness-of-fit function $f$ with ridge regularization, namely
$$
\arg\min_{w \in \mathbb R^d} \Big\{ f(w) + \frac{\lambda}{2} \|w\|_2^2 \Big\}
$$
where $d$ is the number of features and where we will assume that $f$ is $L$-smooth.
We will consider below the following cases.

**Linear regression**, where 
$$
f(w) = \frac 1n \sum_{i=1}^n f_i(w) = \frac{1}{2n} \sum_{i=1}^n (y_i - x_i^\top w)^2 + \frac{\lambda}{2} \|w\|_2^2 = \frac{1}{2 n} \| y - X w \|_2^2 + \frac{\lambda}{2} \|w\|_2^2,
$$
where $n$ is the sample size, $y = [y_1 \cdots y_n]$ is the vector of labels and $X$ is the matrix of features with lines containing the features vectors $x_i \in \mathbb R^d$.

**Logistic regression**, where
$$
f(w) = \frac 1n \sum_{i=1}^n f_i(w) = \frac{1}{n} \sum_{i=1}^n \log(1 + \exp(-y_i x_i^\top w)) + \frac{\lambda}{2} \|w\|_2^2,
$$
where $n$ is the sample size, and where labels $y_i \in \{ -1, 1 \}$ for all $i$.

We need to be able to compute $f(w)$ and its gradient $\nabla f(w)$, in order to solve this problem, as well as $\nabla f_i(w)$ for stochastic gradient descent methods and $\frac{\partial f(w)}{\partial w_j}$ for coordinate descent.

Below is the full implementation for linear regression.

<a id='models_regression'></a>

## 2.1 Linear regression


```{code-cell} python
from numpy.linalg import norm


class ModelLinReg:
    """A class giving first order information for linear regression
    with least-squares loss
    
    Parameters
    ----------
    X : `numpy.array`, shape=(n_samples, n_features)
        The features matrix
    
    y : `numpy.array`, shape=(n_samples,)
        The vector of labels
    
    strength : `float`
        The strength of ridge penalization
    """    
    def __init__(self, X, y, strength):
        self.X = X
        self.y = y
        self.strength = strength
        self.n_samples, self.n_features = X.shape
    
    def loss(self, w):
        """Computes f(w)"""
        y, X, n_samples, strength = self.y, self.X, self.n_samples, self.strength
        return 1 / 2 * norm(y - X.dot(w)) ** 2 / n_samples + strength * norm(w) ** 2 / 2
    
    def grad(self, w):
        """Computes the gradient of f at w"""
        y, X, n_samples, strength = self.y, self.X, self.n_samples, self.strength
        return X.T.dot(X.dot(w) - y) / n_samples + strength * w

    def grad_i(self, i, w):
        """Computes the gradient of f_i at w"""
        x_i = self.X[i]
        return (x_i.dot(w) - y[i]) * x_i + self.strength * w

    def grad_coordinate(self, j, w):
        """Computes the partial derivative of f with respect to 
        the j-th coordinate"""
        y, X, n_samples, strength = self.y, self.X, self.n_samples, self.strength
        return X[:, j].T.dot(X.dot(w) - y) / n_samples + strength * w[j]
    
    def hessian(self, w): # add w for consistency in Newton with logistic in number of args
        X, n_samples, n_features = self.X, self.n_samples, self.n_features
        return(X.T.dot(X))/ n_samples + self.strength * np.eye(n_features)

    def lip(self,w):
        """Computes the Lipschitz constant of the gradients of  f"""
        H = self.hessian(w)
        return norm(H, 2)

    def lip_coordinates(self):
        """Computes the Lipschitz constant of the gradients of f with respect to 
        the j-th coordinate"""
        X, n_samples = self.X, self.n_samples
        return (X ** 2).sum(axis=0) / n_samples + self.strength
        
    def lip_max(self):
        """Computes the maximum of the lipschitz constants of the gradients of f_i"""
        X, n_samples = self.X, self.n_samples
        return ((X ** 2).sum(axis=1) + self.strength).max()
    
    
```

<a id='models_logistic'></a>

## 2.2 Logistic regression


See previous lab for mathematical details


```{code-cell} python

class ModelLogReg:
    """A class giving first order information for logistic regression
    
    Parameters
    ----------
    X : `numpy.array`, shape=(n_samples, n_features)
        The features matrix
    
    y : `numpy.array`, shape=(n_samples,)
        The vector of labels
    
    strength : `float`
        The strength of ridge penalization
    """    
    def __init__(self, X, y, strength):
        self.X = X
        self.y = y
        self.strength = strength
        self.n_samples, self.n_features = X.shape
    
    def loss(self, w):
        """Computes f(w)"""
        y, X, n_samples, strength = self.y, self.X, self.n_samples, self.strength
        return np.mean(np.log(1+np.exp(- y * (X.dot(w))))) + strength * norm(w) ** 2 / 2
       
    def grad(self, w):
        """Computes the gradient of f at w"""
        y, X, n_samples, strength = self.y, self.X, self.n_samples, self.strength
        u = y*np.exp(- y * (X.dot(w)))/(1 + np.exp(- y * (X.dot(w))))
        return - (X.T.dot(u))/n_samples + strength * w
    
    def grad_i(self, i, w):
        """Computes the gradient of f_i at w"""
        x_i = self.X[i]
        strength = self.strength
        u = y[i]*np.exp(- y[i] * (x_i.dot(w)))/(1 + np.exp(- y[i] * (x_i.dot(w))))
        return (- u*x_i + strength * w)
    
    def grad_coordinate(self, j, w):
        """Computes the partial derivative of f with respect to 
        the j-th coordinate"""
        y, X, n_samples, strength = self.y, self.X, self.n_samples, self.strength
        u = y*np.exp(- y * (X.dot(w)))/(1 + np.exp(- y * (X.dot(w))))
        return - (X[:, j].T.dot(u))/n_samples + strength * w[j]
    
    def lip(self,w):
        """Computes the Lipschitz constant of the gradient of  f"""
        X, n_samples = self.X, self.n_samples
        return norm(X.T.dot(X), 2) / (4*n_samples) + self.strength

    def lip_coordinates(self):
        """Computes the Lipschitz constant of the gradient of f with respect to 
        the j-th coordinate"""
        X, n_samples = self.X, self.n_samples
        return (X ** 2).sum(axis=0) / (4*n_samples) + self.strength

    def lip_max(self):
        """Computes the maximum of the lipschitz constants of the gradients of f_i"""
        X, n_samples = self.X, self.n_samples
        return ((X ** 2).sum(axis=1)/4 + self.strength).max()
    
    def hessian(self, w):
        X, n_samples = self.X, self.n_samples
        u = np.exp(- y * (X.dot(w)))/(1 + np.exp(- y * (X.dot(w))))
        M = np.diag(u) @ X
        return (M.T @ M)/n_samples + self.strength * np.eye(self.n_features)
        
```

<a id='solvers'></a>
## 3. Solvers

We now have classes `ModelLinReg` and `ModelLogReg` that allow to compute $f(w)$, $\nabla f(w)$, 
$\nabla f_i(w)$ and $\frac{\partial f(w)}{\partial w_j}$ for the objective $f$
given by linear and logistic regression. We want now to code and compare several solvers to minimize $f$.


```{code-cell} python
def true_parameters(n_features):    
    nnz = 20 # Number of non-zeros coordinates.
    idx = np.arange(n_features)
    w = (-1) ** (idx + 1) * np.exp(-idx / 10.)
    w[nnz:] = 0.
    return w

# Number of features
n_features = 50

# Starting point of all solvers
w0 = np.zeros(n_features)

# Number of iterations
n_iter = 200

# True parameter used to generate data for both logistic/linear regression
W_TRUE = true_parameters(n_features)
```


```{code-cell} python
# Choose which kind of regression to do.

TYPE_REGRESSION = "Linear" # or "Logistic"
# TYPE_REGRESSION = "Logistic"


if TYPE_REGRESSION == "Linear":
    X, y = simu_linreg(W_TRUE, n_samples=1000, corr=0.6)
    model = ModelLinReg(X, y, strength=1e-3)
elif TYPE_REGRESSION == "Logistic":
    X, y = simu_logreg(W_TRUE, n_samples=1000, corr=0.6)
    model = ModelLogReg(X, y, strength=1e-3)
else:
    raise ValueError("The type of regression is incorrect. Must be 'Linear' or 'Logistic'.")
    
```


```{code-cell} python
from scipy.optimize import check_grad


print(check_grad(model.loss, model.grad, w0)) # This must be a number (of order 1e-6)
```

<a id='tools'></a>
## 3.1 Tools for the solvers

The following tools store the loss after each epoch


```{code-cell} python
#Import from previous Lab GD, SGD, AGD, HB
from helpers_and_algorithms import *
```


```{code-cell} python
# Use GD to approximate the minimum value
callback_long = inspector(model, n_iter=10000, verbose=False)
w_star = gd(model, w0, step=1/model.lip_max(), n_iter=10000, callback=callback_long, verbose=False)
obj_min = callback_long.objectives[-1]
```

<a id='sag'></a>
## 3.6. Stochastic average gradient descent

**1) Complete the function `sag` below that implements the stochastic averaged gradient algorithm and test it using the next cell.**


```{code-cell} python
def sag(model, w0, n_iter, step, callback, verbose=True):
    """Stochastic average gradient descent
    """
    w = w0.copy()
    n_samples, n_features = model.n_samples, model.n_features
    gradient_memory = np.zeros((n_samples, n_features)) # one gradient per sample n= 60k,  d= 50M  => 3 10^12
    y = np.zeros(n_features)
    callback(w)
    it = 0
    for idx in range(n_iter):
        
        
        #
        #
        # YOUR CODE OR ANSWER HERE
        #
        #
        
        
        callback(w)
    return w


```


```{code-cell} python
step = 1 / model.lip_max()
callback_sag = inspector(model, n_iter=n_iter)
w_sag = sag(model, w0, n_iter=n_iter, step=step, callback=callback_sag)
```

**2) What is the rate of convergence of SAG? What is the main problem of this algorithm? What is its memory footprint?**

Write your answer below

-

<a id='svrg'></a>
## 3.7. Stochastic variance reduced gradient

**3) Complete the function `svrg` below that implements the stochastic variance reduced gradient algorithm and test it using the next cell.**


```{code-cell} python
def svrg(model, w0, n_iter, step, callback, verbose=True):
    """Stochastic variance reduced gradient descent
    """
    w = w0.copy()
    w_old = w.copy()
    temp_sum = 0
    n_samples = model.n_samples
    callback(w)
    for idx in range(n_iter):        
        
        
        #
        #
        # YOUR CODE OR ANSWER HERE
        #
        #
        
        
        callback(w)
    return 

```


```{code-cell} python
step = 1 / model.lip_max()
callback_svrg = inspector(model, n_iter=n_iter)
w_svrg = svrg(model, w0, n_iter=n_iter,
              step=step, callback=callback_svrg)
```


```{code-cell} python
callbacks_variance_reduction = [callback_sag, callback_svrg]
names_variance_reduction = ["SAG", "SVRG"]

plot_callbacks(callbacks_variance_reduction, names_variance_reduction, obj_min, "Different strategies for variance reduction")

#### ERROR : we are comparing epoch on svrg and sag, and forgetting that in
### svrg we recompute every gradient after each pass on the data
```

**4) What is the rate of SVRG? What is its memory footprint and cost per iteration?**

Write your answer below

-

<a id='newton'></a>


## 3.8 Newton descent

Newton algorithm consist in optimizing the exact taylor approximation of the function at order 2 of the function. To compare it with first order methods, we propose to implement it here. The actualization writes 

$$
w^{t+1} = w^t - \gamma [\nabla^2f(w^t)]^{-1}\nabla f(w^t).
$$


**5) Complete the function `newton` that implements the newton solver and test it.**


```{code-cell} python
def newton(model, w0, n_iter, step, callback, verbose=True):
    """Newton"""
    w = w0.copy()
    n_features = model.n_features
    accumulated_gradient = np.zeros(n_features)
    u =  np.zeros(n_features)
    if verbose:
        print("Lauching Newton solver...")
    callback(w)
    for k in range(n_iter + 1):
        
        #
        #
        # YOUR CODE OR ANSWER HERE
        #
        #
        
        callback(w)
    return w

```


```{code-cell} python
step = 0.8
callback_newton = inspector(model, n_iter=n_iter)
w_newton = newton(model, w0, n_iter=n_iter, step = step, callback=callback_newton)
```


```{code-cell} python
callbacks_newton = [callback_newton]
names_newton = ["Newton"]

plot_callbacks(callbacks_newton, names_newton, obj_min, f"Newton method with stepsize ={step}")
```

**6) Questions on Newton method:**
- What is the rate of convergence of Newton method?
- What is the main problem of this algorithm? 
- How can this be alleviated?
- Try step=1 : How many iterations are necessary for convergence? Why is it the case? Change to logistic model to compare?



Write your answer below

-

<a id='cgd'></a>

## 3.9 Coordinate gradient descent

CGD is considered as the best optimization algorithm. It writes at step $t$

For $k$ in $ 0,\ldots, d$ do:
$
w^{t+1}_k = w^t_k - \gamma_k \partial_{w_k} f(w^t)
$

**7) Complete the function `cgd` below that implements the coordinate gradient descent algorithm and test it using the next cell.**


```{code-cell} python
def cgd(model, w0, n_iter, callback, verbose=True):
    """Coordinate gradient descent
    """
    w = w0.copy()
    n_features = model.n_features
    steps = 1 / model.lip_coordinates()
    if verbose:
        print("Lauching CGD solver...")
    callback(w)
    for k in range(n_iter + 1):
        
        #
        #
        # YOUR CODE OR ANSWER HERE
        #
        #
        
        callback(w)
    return w

```


```{code-cell} python
callback_cgd = inspector(model, n_iter=n_iter)
w_cgd = cgd(model, w0, n_iter=n_iter, callback=callback_cgd)
```

Now we compute the optimal loss $f_\star \triangleq f(w_\star)$.


```{code-cell} python
plot_callbacks([callback_cgd], ["CGD"], obj_min, "Coordinate gradient descent")
```

**6) Questions on CGD method:**
- which one of the three strategies seen in the lecture is implemented here?


Write your answer below

-

<a id='comparison'></a>
# 4. Comparison of all previous algorithms

We reuse the already implemented GD, SGD, AGD, HB from the previous lab

The excess loss will be plotted using the below function.

<a id='sgd'></a>
## 4.1 Recover results for  Stochastic gradient descent, Accelerated GD,  HB

### GD


```{code-cell} python
callback_gd = inspector(model, n_iter=n_iter)
w_gd = gd(model, w0, step= 1/model.lip_max(),  n_iter=n_iter, callback=callback_gd)
```

### SGD
With various strategies, see previous lab


```{code-cell} python
step = 1

callback_sgd_constant = inspector(model, n_iter=n_iter)
callback_sgd_decaying = inspector(model, n_iter=n_iter)
callback_sgd_constant_PR = inspector(model, n_iter=n_iter)
callback_sgd_decaying_PR = inspector(model, n_iter=n_iter)

sgd(model, w0, n_iter=n_iter, step=step, callback=callback_sgd_constant, 
    stepsize_strategy="constant")
sgd(model, w0, n_iter=n_iter, step=step, callback=callback_sgd_decaying, 
    stepsize_strategy="decaying")
sgd(model, w0, n_iter=n_iter, step=step, callback=callback_sgd_constant_PR, 
    stepsize_strategy="constant", pr_averaging=True)
sgd(model, w0, n_iter=n_iter, step=step, callback=callback_sgd_decaying_PR, 
    stepsize_strategy="decaying", pr_averaging=True)


callbacks_sgd = [callback_sgd_constant, callback_sgd_decaying, callback_sgd_constant_PR, callback_sgd_decaying_PR]

names_sgd = ["SGD constant", "SGD decaying", "PR constant", "PR decaying"]

plot_callbacks(callbacks_sgd, names_sgd, obj_min, "Different strategies for SGD")
```

<a id='agd'></a>
### Accelerated gradient descent

With various strategies, see previous lab


```{code-cell} python
callback_agd_constant = inspector(model, n_iter=n_iter)
callback_agd_convex = inspector(model, n_iter=n_iter)
callback_agd_convex_approx = inspector(model, n_iter=n_iter)
callback_agd_strongly_convex = inspector(model, n_iter=n_iter)

agd(model, w0, n_iter=n_iter, callback=callback_agd_constant, momentum_strategy="constant")
agd(model, w0, n_iter=n_iter, callback=callback_agd_convex, momentum_strategy="convex")
agd(model, w0, n_iter=n_iter, callback=callback_agd_convex_approx, momentum_strategy="convex_approx")
agd(model, w0, n_iter=n_iter, callback=callback_agd_strongly_convex, momentum_strategy="strongly_convex")

callbacks_agd = [callback_agd_constant, callback_agd_convex, callback_agd_convex_approx, 
                 callback_agd_strongly_convex]
names_agd = ["AGD constant", "AGD cvx", "AGD cvx approx", "AGD stgly cvx"]

plot_callbacks(callbacks_agd, names_agd, obj_min, "Different strategies for AGD")

```

<a id='hb'></a>
### Heavy ball method


```{code-cell} python
callback_hb = inspector(model, n_iter=n_iter)

heavy_ball_optimized(model, w0, n_iter=n_iter, callback=callback_hb)

plot_callbacks([callback_hb], ["HB"], obj_min, "Heavy ball method")
```

<a id='sgd'></a>
## 4.2 Comparison of all previous algorithms

**7) Plot the values of the loss for the different iteration and for each solver. Comment.**


```{code-cell} python
# NB : Enlever newton et CGD si pas traité

callbacks = [callback_gd, callback_agd_convex, callback_hb, callback_cgd, 
             callback_sgd_decaying, callback_sgd_decaying_PR,
             callback_sag, callback_svrg,
             callback_newton]
names = ["GD", "AGD", "HB", "CGD", 
         "SGD decaying", "SGD decaying PR",
         "SAG", "SVRG",
         "Newton"]

# YOUR CODE OR ANSWER HERE
```


```{code-cell} python

```

**6) Comment on the rate of each algorithm
- which is the best solution in that setup and why
- change the model (logistic vs linear) and rerun everything, are conlusions similar?
- change the noise, number of points, dimension, are conlusions similar



Write your answer below

-

<a id='sgd'></a>
## 4.3Impact of feature correlation

Use this function to answer the 2 following questions


```{code-cell} python
def gd_perf_on_log_reg_wrt_correlation_and_regularization(corr=.6, strength=1e-3):
    X_loc, y_loc = simu_logreg(w0, corr=corr)
    model_loc = ModelLogReg(X_loc, y_loc, strength=strength)
    
    callback_long = inspector(model_loc, n_iter=1000, verbose=False)
    w_long = gd(model_loc, w0, step= 1/model_loc.lip(),  n_iter=1000, callback=callback_long, verbose=False)

    callback_loc = inspector(model_loc, n_iter=n_iter, verbose=False)
    w_loc = gd(model_loc, w0, step= 1/model_loc.lip(),  n_iter=n_iter, callback=callback_loc, verbose=False)
    
    return ((model_loc.loss(w_loc) - model_loc.loss(w_long)) / (model_loc.loss(w0) - model_loc.loss(w_long)))**(1/n_iter)
```

**15) In logistic regression, study the influence of the correlation of the features on the performance of the optimization algorithms. Explain.**


```{code-cell} python
#### Expliquer ce qu'on compare : (f(x^t) - f^*) / (f(x^0) - f(x^*))
```


```{code-cell} python
corrs = np.linspace(.1, 1, 10, endpoint=False)
gd_final_values = list()


#
#
# YOUR CODE OR ANSWER HERE
#
#


plt.plot(corrs, gd_final_values)
plt.xlabel("Correlation")
plt.ylabel("Average progression of GD after 50 iterations")
plt.title("Impact of the correlation of the features \n on the performance of GD on Logistic regression model.")

```

Write your answer below

-

**16) In logistic regression, study the influence of the level of ridge penalization on the performance of the optimization algorithms. Explain.**


```{code-cell} python
strengths = np.logspace(-3, 1, 5)
gd_final_values = list()


#
#
# YOUR CODE OR ANSWER HERE
#
#


plt.plot(strengths, gd_final_values)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Regularization strengths")
plt.ylabel("Average progression of GD after 50 iterations")
plt.title("Impact of the regularization of the logistic regression model \n on the performance of GD.")

```

Write your answer below

-


```{code-cell} python

```

## Bonus: what about the test loss

Generate a test dataset and report the test performance
