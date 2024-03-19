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
  title: Variance reduction, Newton, CGD
  version: '1.0'
---

 
<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Aymeric DIEULEVEUT</span>
<span>Licence CC BY-NC-ND</span>
</div>

+++

# Gradient Descent Methods - Variance reduction, Newton, CGD

The aim of this material is to code 
- Variance reduced methods
- coordinate gradient descent (CGD)
- newton descent

and apply them on the linear regression and logistic regression models, with ridge penalization.

# Table of content

[1. Introduction](#intro)<br>

[2. Models gradients and losses](#models)<br>

[2.1  Linear regression](#models_regression)<br>
[2.2  Check for Linear regression](#models_regression_check)<br>
[2.3  Logistic regression](#models_logistic)<br>
[2.4  Check for logistic regression](#models_logistic_check)<br>


[3. Solvers](#solvers)<br>

[3.1 Tools for solvers](#tools)<br>
[3.2 Stochastic Average Gradient descent](#sag)<br>
[3.3 Stochastic Variance Reduced Gradient descent](#svrg)<br>
[3.4 Newton descent](#newton)<br>
[3.5 Coordinate Gradient descent](#cgd)<br>


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




    <matplotlib.legend.Legend at 0x10f7554d0>




    
![png](media/Part_2/2.9/Optimization3-Var-Red-Newton-CGD-Solution_7_1.png)
    


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




    Text(0.5, 1.0, 'Logistic regression simulation')




    
![png](media/Part_2/2.9/Optimization3-Var-Red-Newton-CGD-Solution_10_1.png)
    


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




    Text(0.5, 1.0, 'Model weights')




    
![png](media/Part_2/2.9/Optimization3-Var-Red-Newton-CGD-Solution_14_1.png)
    



```{code-cell} python
from scipy.optimize import check_grad

X, y = simu_linreg(w0, corr=0.6)
model = ModelLinReg(X, y, strength=1e-3)
w = np.random.randn(n_features)

print(check_grad(model.loss, model.grad, w)) # This must be a number (of order 1e-6)
```

    4.45572990476208e-06



```{code-cell} python
print("lip=", model.lip())
print("lip_max=", model.lip_max())
print("lip_coordinates=", model.lip_coordinates())
```

    lip= 4.110824241039015
    lip_max= 109.02063655415297
    lip_coordinates= [1.02 1.   0.98 0.99 1.   1.07 1.07 0.99 0.99 0.95 0.94 0.98 0.98 0.99
     1.03 0.98 0.91 1.02 0.99 1.06 0.98 1.01 1.04 1.01 1.01 1.02 1.02 0.98
     1.05 0.99 0.92 0.92 0.93 0.98 1.   1.04 1.03 1.02 1.   1.01 1.   0.98
     0.98 1.02 1.01 1.1  1.01 0.97 1.04 1.04]


<a id='models_logistic'></a>

## 2.3 Logistic regression

**NB**: you can skip these questions and go to the solvers implementation, and come back here later.


**1) Compute (on paper) the gradient $\nabla f$, the gradient of $\nabla f_i$ and the gradient of the coordinate function $\frac{\partial f(w)}{\partial w_j}$ of $f$ for logistic regression (fill the class given below).**


Let $w$ in $\mathbb R^d$.

We have 
$$
f(w) = \frac{1}{n} \sum_{i=1}^n f_i(w) = \frac{1}{n} \sum_{i=1}^n \log(1 + \exp(-y_i x_i^\top w)) + \frac{\lambda}{2} \|w\|_2^2.
$$

Thus $\nabla f(w) = \frac{1}{n} \sum_{i=1}^n \nabla f_i(w)$. 

Let $i \in \{1, ..., n\}$, we have
$$
\nabla f_i(w) = - y_i \frac{1}{1 + \exp(y_i x_i^\top w)} x_i + \lambda w,
$$

and at the end:
$$
\nabla f(w) = -\frac{1}{n}\sum_{i=1}^n y_i \frac{1}{1 + \exp(y_i x_i^\top w)} x_i + \lambda w.
$$

In particular for any $j \in \{1, ..., d\}$,

$$
\frac{\partial f(w)}{\partial w_j} = -\frac{1}{n}\sum_{i=1}^n y_i \frac{1}{1 + \exp(y_i x_i^\top w)} x_{i,j} + \lambda w_j.
$$

We can also compute the hessian. Since for all $i$ in $\{1, ..., n\}~, y_i^2 = 1$, we have:
$$
\nabla^2 f(w) = \frac{1}{n}\sum_{i=1}^n \frac{\exp(y_i x_i^\top w)}{(1 + \exp(y_i x_i^\top w))^2} x_i x_i^\top + \lambda I_d.
$$



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
    
    def lip(self):
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
    
    def hessian(self):
        X, n_samples = self.X, self.n_samples
        u = np.exp(- y * (X.dot(w)))/(1 + np.exp(- y * (X.dot(w))))
        M = np.diag(u) @ X
        return (M.T @ M)/n_samples + self.strength * np.eye(self.n_features)
        


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

    Checkgrad returns 3.22e-07



    
![png](media/Part_2/2.9/Optimization3-Var-Red-Newton-CGD-Solution_22_1.png)
    



```{code-cell} python
print("lip=", model.lip())
print("lip_max=", model.lip_max())
print("lip_coordinates=", model.lip_coordinates())
```

    lip= 1.0956195251895076
    lip_max= 26.68930948544274
    lip_coordinates= [0.24 0.23 0.22 0.25 0.24 0.25 0.26 0.26 0.25 0.26 0.25 0.24 0.24 0.25
     0.25 0.24 0.24 0.24 0.26 0.27 0.26 0.24 0.25 0.26 0.26 0.25 0.25 0.27
     0.24 0.24 0.24 0.24 0.27 0.27 0.25 0.26 0.25 0.25 0.27 0.26 0.27 0.26
     0.26 0.26 0.26 0.25 0.27 0.26 0.24 0.24]


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

    1.7137729689063778e-07


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

<a id='sag'></a>
## 3.6. Stochastic average gradient descent

**9) Complete the function `sag` below that implements the stochastic averaged gradient algorithm and test it using the next cell.**


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
        
        ################## TODO BLOCK
        idx_samples = np.random.randint(0, model.n_samples, model.n_samples)
        for i in idx_samples:      
            y_new = model.grad_i(i, w)
            y += (y_new - gradient_memory[i]) / n_samples
            gradient_memory[i] = y_new
            w -= step * y
        ################## END TODO BLOCK    
        
        callback(w)
    return w


```


```{code-cell} python
step = 1 / model.lip_max()
callback_sag = inspector(model, n_iter=n_iter)
w_sag = sag(model, w0, n_iter=n_iter, step=step, callback=callback_sag)
```

       it    |   obj   
           0 | 9.074961e-01
          40 | 1.336772e-01
          80 | 1.336772e-01
         120 | 1.336772e-01
         160 | 1.336772e-01
         200 | 1.336772e-01


**10) What is the rate of convergence of SAG? What is the main problem of this algorithm? What is its memory footprint?**

################## TODO BLOCK  

SAG is a linear convergent algorithm. Its rate of convergence is equivalent to SGD during first iterations, then equivalent to GD. The main drawback of SAG is it's memory cost. It requires to store $n$ gradient, that is to say a matrix of size $n\times d$.

################## END TODO BLOCK  

<a id='svrg'></a>
## 3.7. Stochastic variance reduced gradient

**11) Complete the function `svrg` below that implements the stochastic variance reduced gradient algorithm and test it using the next cell.**


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
        
        ################## TODO BLOCK
        w_old[:] = temp_sum/n_samples
        mu = model.grad(w)
        temp_sum = 0
        idx_samples = np.random.randint(0, model.n_samples, model.n_samples)
        for i in idx_samples:  
            z_new = model.grad_i(i, w)
            z_old = model.grad_i(i, w_old)  ### w_old is the only thing I keep, I can recompute the gradient f_i (w_old)
            ## SVRG trades memory for computation
            w -= step * (z_new - z_old + mu)
            temp_sum += w
        ################## END TODO BLOCK    
        
        callback(w)
    return 


```


```{code-cell} python
step = 1 / model.lip_max()
callback_svrg = inspector(model, n_iter=n_iter)
w_svrg = svrg(model, w0, n_iter=n_iter,
              step=step, callback=callback_svrg)
```

       it    |   obj   
           0 | 9.074961e-01
          40 | 1.336772e-01
          80 | 1.336772e-01
         120 | 1.336772e-01
         160 | 1.336772e-01
         200 | 1.336772e-01



```{code-cell} python
callbacks_variance_reduction = [callback_sag, callback_svrg]
names_variance_reduction = ["SAG", "SVRG"]


#### ERROR : we are comparing epoch on svrg and sag, and forgetting that in
### svrg we recompute every gradient after each pass on the data
```

<a id='newton'></a>


## 3.8 Newton descent

Newton algorithm consist in optimizing the exact taylor approximation of the function at order 2 of the function. To compare it with first order methods, we propose to implement it here. The actualization writes 

$$
w^{t+1} = w^t - \gamma [\nabla^2f(w^t)]^{-1}\nabla f(w^t).
$$


**12) Complete the function `newton` that implements the newton solver and test it.**


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
        ################## TODO BLOCK
        gradient = model.grad(w)
        hessian = model.hessian()
        direction = np.linalg.inv(hessian).dot(gradient)
        w  -= step * direction
        ################## END TODO BLOCK
        callback(w)
    return w


```


```{code-cell} python
step = 0.1
callback_newton = inspector(model, n_iter=n_iter)
w_newton = newton(model, w0, n_iter=n_iter, step = step, callback=callback_newton)
```

    Lauching Newton solver...
       it    |   obj   
           0 | 9.074961e-01
          40 | 1.338462e-01
          80 | 1.336772e-01
         120 | 1.336772e-01
         160 | 1.336772e-01
         200 | 1.336772e-01



```{code-cell} python
callbacks_newton = [callback_newton]
names_newton = ["Newton"]


```

<a id='cgd'></a>

## 3.9 Coordinate gradient descent

CGD is considered as the best optimization algorithm. It writes at step $t$

For $k$ in $ 0,\ldots, d$ do:
$
w^{t+1}_k = w^t_k - \gamma_k \partial_{w_k} f(w^t)
$

**13) Complete the function `cgd` below that implements the coordinate gradient descent algorithm and test it using the next cell.**


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
        ################## TODO BLOCK
        for j in range(n_features): # 50 features
            w[j] -= steps[j] * model.grad_coordinate(j, w)  # 
        ################## END TODO BLOCK
        callback(w)
    return w

```


```{code-cell} python
callback_cgd = inspector(model, n_iter=n_iter)
w_cgd = cgd(model, w0, n_iter=n_iter, callback=callback_cgd)
```

    Lauching CGD solver...
       it    |   obj   
           0 | 9.074961e-01
          40 | 1.336772e-01
          80 | 1.336772e-01
         120 | 1.336772e-01
         160 | 1.336772e-01
         200 | 1.336772e-01


Now we compute the optimal loss $f_\star \triangleq f(w_\star)$.


```{code-cell} python
callback_long = inspector(model, n_iter=1000, verbose=False)
w_cgd = cgd(model, w0, n_iter=1000, callback=callback_long, verbose=False)
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
plot_callbacks([callback_cgd], ["CGD"], obj_min, "Coordinate gradient descent")
```

    /var/folders/c0/s8t504vd0w52xgnvj9cb82w40000gn/T/ipykernel_4804/3220526258.py:17: UserWarning: The figure layout has changed to tight
      plt.tight_layout()





    <module 'matplotlib.pyplot' from '/Users/aymericdieuleveut/anaconda3/lib/python3.11/site-packages/matplotlib/pyplot.py'>




    
![png](media/Part_2/2.9/Optimization3-Var-Red-Newton-CGD-Solution_52_2.png)
    

