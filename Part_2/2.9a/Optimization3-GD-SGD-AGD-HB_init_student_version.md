# Optimization 3: Algorithms  - GD, SGD, AGD, HB
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
  title: Optimization 3: Algorithms  - GD, SGD, AGD, HB
  version: '1.0'
---

 
<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Aymeric DIEULEVEUT</span>
<span>Licence CC BY-NC-ND</span>
</div>

The aim of this material is to code 
- gradient descent (GD)
- stochastic gradient descent (SGD)
- accelerated gradient descent (AGD)
- Heavy Ball methods (HB)
and apply them on the linear regression and logistic regression models, with ridge penalization.

Importantly, interpretation of the rates is one key aspect.


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
from numpy.random import randn
from numpy.random import multivariate_normal
from scipy.linalg import toeplitz
```


```{code-cell} python
def simu_linreg(w0, n_samples=1000, corr=0.5, std=0.5):
    """Simulation of a linear regression model with Gaussian features
    and a Toeplitz covariance, with Gaussian noise.
    
    Parameters
    ----------
    w0 : `numpy.array`, shape=(n_features,)
        Model weights
    
    n_samples : `int`, default=1000
        Number of samples to simulate
    
    corr : `float`, default=0.5
        Correlation of the features
    
    std : `float`, default=0.5
        Standard deviation of the noise
    
    Returns
    -------
    X : `numpy.ndarray`, shape=(n_samples, n_features)
        Simulated features matrix. It contains samples of a centered 
        Gaussian  vector with Toeplitz covariance.
    
    y : `numpy.array`, shape=(n_samples,)
        Simulated labels
    """
    n_features = w0.shape[0]
    # Construction of a covariance matrix
    cov = toeplitz(corr ** np.arange(0, n_features))
    # Simulation of features
    X = multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    # Simulation of the labels
    y = X.dot(w0) + std * randn(n_samples)
    return X, y
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
def sigmoid(t):
    """Sigmoid function (overflow-proof)"""
    idx = t > 0
    out = np.empty(t.size)    
    out[idx] = 1 / (1. + np.exp(-t[idx]))
    exp_t = np.exp(t[~idx])
    out[~idx] = exp_t / (1. + exp_t)
    return out

def simu_logreg(w0, n_samples=1000, corr=0.5):
    """Simulation of a logistic regression model with Gaussian features
    and a Toeplitz covariance.
    
    Parameters
    ----------
    w0 : `numpy.array`, shape=(n_features,)
        Model weights
    
    n_samples : `int`, default=1000
        Number of samples to simulate
    
    corr : `float`, default=0.5
        Correlation of the features

    Returns
    -------
    X : `numpy.ndarray`, shape=(n_samples, n_features)
        Simulated features matrix. It contains samples of a centered 
        Gaussian vector with Toeplitz covariance.
    
    y : `numpy.array`, shape=(n_samples,)
        Simulated labels
    """
    n_features = w0.shape[0]
    cov = toeplitz(corr ** np.arange(0, n_features))
    X = multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    p = sigmoid(X.dot(w0))
    y = np.random.binomial(1, p, size=n_samples)
    # Put the label in {-1, 1}
    y[:] = 2 * y - 1
    return X, y
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
    
    def hessian(self):
        X, n_samples, n_features = self.X, self.n_samples, self.n_features
        return(X.T.dot(X))/ n_samples + self.strength * np.eye(n_features)

    def lip(self):
        """Computes the Lipschitz constant of the gradients of  f"""
        H = self.hessian()
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

<a id='models_regression_check'></a>

## 2.2 Checks for the linear regression model


```{code-cell} python
## Simulation setting
n_features = 50
nnz = 20
idx = np.arange(n_features)
w0 = (-1) ** (idx + 1) * np.exp(-idx / 10.)
w0[nnz:] = 0.

plt.figure(figsize=(5, 3))
plt.stem(w0)
plt.title("Model weights")
```


```{code-cell} python
from scipy.optimize import check_grad

X, y = simu_linreg(w0, corr=0.6)
model = ModelLinReg(X, y, strength=1e-3)
w = np.random.randn(n_features)

print(check_grad(model.loss, model.grad, w)) # This must be a number (of order 1e-6)
```


```{code-cell} python
print("lip=", model.lip())
print("lip_max=", model.lip_max())
print("lip_coordinates=", model.lip_coordinates())
```

<a id='models_logistic'></a>

## 2.3 Logistic regression

**NB**: you can skip these questions and go to the solvers implementation, and come back here later.


**1) Compute (on paper) the gradient $\nabla f$, the gradient of $\nabla f_i$ and the gradient of the coordinate function $\frac{\partial f(w)}{\partial w_j}$ of $f$ for logistic regression (fill the class given below).**

Write your answer below

**2) Fill in the functions below for the computation of $f$, $\nabla f$, $\nabla f_i$ and $\frac{\partial f(w)}{\partial w_j}$ for logistic regression in the ModelLogReg class below.**


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
        
        #
        #
        # YOUR CODE OR ANSWER HERE
        #
        #
        
       
    def grad(self, w):
        """Computes the gradient of f at w"""
        y, X, n_samples, strength = self.y, self.X, self.n_samples, self.strength
        
        #
        #
        # YOUR CODE OR ANSWER HERE
        #
        #
        
    
    def grad_i(self, i, w):
        """Computes the gradient of f_i at w"""
        x_i = self.X[i]
        strength = self.strength
        
        #
        #
        # YOUR CODE OR ANSWER HERE
        #
        #
        
    
    def grad_coordinate(self, j, w):
        """Computes the partial derivative of f with respect to 
        the j-th coordinate"""
        y, X, n_samples, strength = self.y, self.X, self.n_samples, self.strength
        
        #
        #
        # YOUR CODE OR ANSWER HERE
        #
        #
        
    
    def lip(self):
        """Computes the Lipschitz constant of the gradient of  f"""
        X, n_samples = self.X, self.n_samples
        
        #
        #
        # YOUR CODE OR ANSWER HERE
        #
        #
        
        


```

<a id='models_logistic_check'></a>


## 2.4 Checks for the logistic regression model

**3) Use the function `simu_logreg` to simulate data according to the logistic regression model. Check numerically the gradient using the function ``checkgrad`` from ``scipy.optimize``, as we did for linear regression above.**


```{code-cell} python
## Simulation setting
n_features = 50
nnz = 20
idx = np.arange(n_features)
w0 = (-1) ** (idx + 1) * np.exp(-idx / 10.)
w0[nnz:] = 0.

plt.figure(figsize=(5, 3))
plt.stem(w0)
plt.title("Model weights")

from scipy.optimize import check_grad


X, y = simu_logreg(w0, corr=0.6)
model = ModelLogReg(X, y, strength=1e-3)
w = np.random.randn(n_features)
print('Checkgrad returns %.2e' % (check_grad(model.loss, model.grad, w))) # This must be a number (of order 1e-6)

```


```{code-cell} python
print("lip=", model.lip())
print("lip_max=", model.lip_max())
print("lip_coordinates=", model.lip_coordinates())
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


if TYPE_REGRESSION == "Linear":
    X, y = simu_linreg(W_TRUE, corr=0.6)
    model = ModelLinReg(X, y, strength=1e-3)
elif TYPE_REGRESSION == "Logistic":
    X, y = simu_logreg(W_TRUE, corr=0.6)
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
def inspector(model, n_iter, verbose=True):
    """A closure called to update metrics after each iteration.
    Don't even look at it, we'll just use it in the solvers."""
    objectives = []
    it = [0] # This is a hack to be able to modify 'it' inside the closure.
    def inspector_cl(w):
        obj = model.loss(w)
        objectives.append(obj)
        if verbose == True:
            if it[0] == 0:
                print(' | '.join([name.center(8) for name in ["it", "obj"]]))
            if it[0] % (n_iter / 5) == 0:
                print(' | '.join([("%d" % it[0]).rjust(8), ("%.6e" % obj).rjust(8)]))
            it[0] += 1
    inspector_cl.objectives = objectives
    return inspector_cl
```

<a id='gd'></a>
## 3.2 Gradient descent

We start by implementing the simple gradient descent. We will use it to compute the optimal point $w_\star$. 
Next, we will plot for each algorithm the excess loss $f(w) - f(w_\star)$ to check its rate of convergence.

**4) Complete the function `gd` below that implements the gradient descent algorithm and test it using the next cell.**





```{code-cell} python
def gd(model, w0, step,  n_iter, callback, verbose=True):
    """Gradient descent
    """
    #step = 1 / model.lip()
    w = w0.copy()
    w_new = w0.copy()
    if verbose:
        print("Lauching GD solver...")
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

Now we compute the optimal loss $f_\star \triangleq f(w_\star)$.


```{code-cell} python
callback_long = inspector(model, n_iter=10000, verbose=False)
w_star = gd(model, w0, step=1/model.lip(), n_iter=10000, callback=callback_long, verbose=False)
obj_min = callback_long.objectives[-1]
```

The excess loss will be plotted using the below function.


```{code-cell} python
def plot_callbacks(callbacks, names, obj_min, title):

    plt.figure(figsize=(6, 6))
    plt.yscale("log")

    for callback, name in zip(callbacks, names):
        objectives = np.array(callback.objectives)
        objectives_dist = objectives - obj_min    
        plt.plot(objectives_dist, label=name, lw=2)

    plt.tight_layout()
    plt.xlim((0, len(objectives_dist)))
    plt.xlabel("Number of passes on the data", fontsize=16)
    plt.ylabel(r"$f(w_k) - f_\star$", fontsize=16)
    plt.legend(loc='lower left')
    plt.title(title, fontsize=16)
    plt.tight_layout()
    return plt
```


```{code-cell} python
n_iter=50
```


```{code-cell} python
callback_gd = inspector(model, n_iter=n_iter)
w_gd = gd(model, w0, step= 1/model.lip(),  n_iter=n_iter, callback=callback_gd)
```


```{code-cell} python
plot_callbacks([callback_gd], ["GD"], obj_min, "Gradient descent")
```

**5) Which step size did you choose? What is the expected rate of convergence?**

Write your answer below

<a id='sgd'></a>
## 3.3 Stochastic gradient descent

**6) Complete the function `sgd` below that implements the stochastic gradient descent algorithm and test it using the next cell. Implement the Polyak-Ruppert averaging using an online update**

You can implement different strategy for the step size:
- Constant step-size: use $$\gamma = \frac{1}{2L}~.$$
- Decaying step size: use $$ \forall k \in \mathbb N, \gamma_k = \frac{1}{L\sqrt{k + 1}}~.$$


```{code-cell} python
def sgd(model, w0, n_iter, step, callback, stepsize_strategy="constant",
        pr_averaging=False, verbose=True):
    
    """Stochastic gradient descent.
    
    stepsize_strategy:{"constant", "strongly_convex", "decaying"}
        define your own strategies to update (or not) the step size.
    pr_averaging: True if using polyak-ruppert averaging.
    """
    
    mu = model.strength
    w = w0.copy()
    w_averaged = w0.copy()
    callback(w)
    n_samples = model.n_samples
    L = model.lip_max()
    it = 0
    for idx in range(n_iter):
        
        idx_samples = np.random.randint(0, model.n_samples, model.n_samples)
        for i in idx_samples: 
            if stepsize_strategy == "constant":
                stepsize = # YOUR CODE OR ANSWER HERE
            elif stepsize_strategy == "strongly_convex": ##### A enlever si pas traité en cours
                # For strongly-convex (choice in the slides)
                stepsize = # YOUR CODE OR ANSWER HERE
            elif stepsize_strategy == "decaying":
                stepsize = # YOUR CODE OR ANSWER HERE
            else:
                raise ValueError('The strategy is not correct')

            w -= # YOUR CODE OR ANSWER HERE

            if pr_averaging:
                # Polyak-Ruppert averaging
                w_averaged = # YOUR CODE OR ANSWER HERE
            it += 1
        if pr_averaging:
            callback(w_averaged)
        else:
            callback(w) 
    if pr_averaging:
        return w_averaged
    return w



```


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
```


```{code-cell} python
callbacks_sgd = [callback_sgd_constant, callback_sgd_decaying, callback_sgd_constant_PR, callback_sgd_decaying_PR]

names_sgd = ["SGD constant", "SGD decaying", "PR constant", "PR decaying"]

plot_callbacks(callbacks_sgd, names_sgd, obj_min, "Different strategies for SGD")
```

<a id='agd'></a>
## 3.4 Accelerated gradient descent

**7) Complete the function `agd` below that implements the (Nesterov) accelerated gradient descent algorithm and test it using the next cell.**

What choice of momentum coefficient is recommended for AGD ?
- for strongly convex
- for convex functions

Here you can implement different strategy:
- Constant: use an arbitrary value (e.g. 0.9).
- Using the strong convexity (see https://blogs.princeton.edu/imabandit/2014/03/06/nesterovs-accelerated-gradient-descent-for-smooth-and-strongly-convex-optimization/):
$$\beta = \frac{\sqrt{\kappa} - 1}{
\sqrt{\kappa} + 1}, \quad \text{with} \quad \kappa = \frac{L}{\mu}.$$
- Using only convexity (see https://blogs.princeton.edu/imabandit/2018/11/21/a-short-proof-for-nesterovs-momentum/)
$$ \beta_k = \frac{t_{k-1} -1}{t_k}, \quad \text{with} \quad t_k = \frac{1}{2} (1 + \sqrt{1 + 4 t_{k-1}^2}). $$ 
This value can be approximated by $$\beta_k = \frac{k}{k+3}.$$


```{code-cell} python
def agd(model, w0, n_iter, callback, verbose=True, momentum_strategy="constant"):
    """(Nesterov) Accelerated gradient descent.
    
    momentum_strategy: {"constant","convex","convex_approx","strongly_convex"} 
        define your own strategies to update (or not) the momentum coefficient.
    """
    mu = model.strength
    step = 1 / model.lip()
    w = w0.copy()
    w_new = w0.copy()
    # An extra variable is required for acceleration
    z = w0.copy() # the auxiliari point at which the gradient is taken
    t = 1. # Usefull for computing the momentum (beta)
    t_new = 1.    
    if verbose:
        print("Lauching AGD solver...")
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
callback_agd_constant = inspector(model, n_iter=n_iter)
callback_agd_convex = inspector(model, n_iter=n_iter)
callback_agd_convex_approx = inspector(model, n_iter=n_iter)
callback_agd_strongly_convex = inspector(model, n_iter=n_iter)

agd(model, w0, n_iter=n_iter, callback=callback_agd_constant, momentum_strategy="constant")
agd(model, w0, n_iter=n_iter, callback=callback_agd_convex, momentum_strategy="convex")
agd(model, w0, n_iter=n_iter, callback=callback_agd_convex_approx, momentum_strategy="convex_approx")
agd(model, w0, n_iter=n_iter, callback=callback_agd_strongly_convex, momentum_strategy="strongly_convex")
```


```{code-cell} python
callbacks_agd = [callback_agd_constant, callback_agd_convex, callback_agd_convex_approx, 
                 callback_agd_strongly_convex]
names_agd = ["AGD constant", "AGD cvx", "AGD cvx approx", "AGD stgly cvx"]

plot_callbacks(callbacks_agd, names_agd, obj_min, "Different strategies for AGD")

```

<a id='hb'></a>
## 3.5. Heavy ball method

**8) Complete the function `hb` below that implements the Heavy ball (HB) method and test it using the next cell.**


```{code-cell} python
def heavy_ball(model, w0, n_iter, step, momentum, callback, verbose=True):
    
    w = w0.copy()
    w_previous = w0.copy()
    callback(w)
    
    for idx in range(n_iter):
        
        ###################### TODO BLOCK
        
        w_next = w - step * model.grad(w) + momentum * (w - w_previous)
        w_previous = w
        w = w_next
        
        ###################### END TOD BLOCK
        
        callback(w)
    
    return w


```

On strongly convex smooth quadratic functions, the recommended momentum coefficient for Heavy-Ball is
$$
\beta = \left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^2, \quad \text{with} \quad \kappa = \frac{L}{\mu}.
$$

and

$$
\gamma = \frac{2}{L+\mu}(1 + \beta) = \frac{4}{(\sqrt{L} + \sqrt{\mu})^2} \,.
$$

This algorithm is guarantee to verify:

$$
\|w_n - w_\star\|^2 \leq \left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^{2n} \|w_0 - w_\star\|^2
$$

on quadratic functions.

There exists a famous strongly convex smooth function (not quadratic) which Heavy-Ball with the above setting doesn't converge (See [Lessard et al. (2014)](https://arxiv.org/pdf/1408.3595.pdf) Section 4.6, Equation 4.11 and Figure 7).
[Ghadimi et al. (2014)](https://arxiv.org/pdf/1412.7457.pdf) (Theorem 4) provides a convergence rate on general strongly convex smooth functions that is much worse (not accelerated) than NAG. However, HB works pretty well in pratice and is the default momentum implementation in most DL frammeworks.

Below, the function `hb_optimized` implements the Heavy ball method with optimized tunning (for quadratic functions), test it using the next cells.


```{code-cell} python
def heavy_ball_optimized(model, w0, n_iter, callback, verbose=True):
        
    mu = model.strength
    L = model.lip()
    
    gamma = 3.99 / (sqrt(L) + sqrt(mu))**2
    beta = ((sqrt(L) - sqrt(mu)) / (sqrt(L) + sqrt(mu)))**2
    
    print(gamma, beta)
    
    return heavy_ball(model=model,
              w0=w0,
              n_iter=n_iter,
              step=gamma,
              momentum=beta,
              callback=callback,
              verbose=verbose,
             )


```


```{code-cell} python
callback_hb = inspector(model, n_iter=n_iter)

heavy_ball_optimized(model, w0, n_iter=n_iter, callback=callback_hb)
```


```{code-cell} python
plot_callbacks([callback_hb], ["HB"], obj_min, "Heavy ball method")
```


```{code-cell} python

```
