# Gradient Descent Methods - GD, SGD, AGD, HB

The aim of this material is to code 
- coordinate gradient descent (CD)
- gradient descent (GD)
- stochastic gradient descent (SGD)
- accelerated gradient descent (AGD)

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
[3.2 Gradient descent](#gd)<br>
[3.3 Stochastic Gradient descent](#sgd)<br>
[3.4 Accelerated Gradient descent](#agd)<br>
[3.5 Heavy ball method](#hb)<br>

<a id='intro'></a>
# 1. Introduction

## 1.1. Getting model weights

We'll start by generating sparse vectors and simulating data


```python
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

np.set_printoptions(precision=2) # to have simpler print outputs with numpy
```

## 1.2. Simulation of a linear model


```python
from numpy.random import randn
from numpy.random import multivariate_normal
from scipy.linalg import toeplitz
```


```python
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


```python
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




    <matplotlib.legend.Legend at 0x10fd87a50>




    
![png](Optimization2-GD-SGD-AGD-HB-Solution_files/Optimization2-GD-SGD-AGD-HB-Solution_7_1.png)
    


## 1.3. Simulation of a logistic regression model


```python
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


```python
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




    
![png](Optimization2-GD-SGD-AGD-HB-Solution_files/Optimization2-GD-SGD-AGD-HB-Solution_10_1.png)
    


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


```python
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


```python
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




    
![png](Optimization2-GD-SGD-AGD-HB-Solution_files/Optimization2-GD-SGD-AGD-HB-Solution_14_1.png)
    



```python
from scipy.optimize import check_grad

X, y = simu_linreg(w0, corr=0.6)
model = ModelLinReg(X, y, strength=1e-3)
w = np.random.randn(n_features)

print(check_grad(model.loss, model.grad, w)) # This must be a number (of order 1e-6)
```

    3.4019493913804284e-06



```python
print("lip=", model.lip())
print("lip_max=", model.lip_max())
print("lip_coordinates=", model.lip_coordinates())
```

    lip= 4.184918885817737
    lip_max= 133.5350649298213
    lip_coordinates= [1.   0.96 1.   0.9  0.91 0.96 1.01 0.98 0.99 0.96 0.95 0.98 1.03 1.07
     1.07 1.05 0.97 0.97 0.94 0.93 0.96 0.99 1.01 1.01 0.96 0.96 0.98 0.98
     1.   0.99 1.01 1.14 1.05 0.98 0.92 0.99 1.01 1.   1.05 1.04 1.07 0.99
     0.97 1.   1.09 1.11 1.08 0.97 0.97 0.98]


<a id='models_logistic'></a>

## 2.3 Logistic regression

**NB**: you can skip these questions and go to the solvers implementation, and come back here later.


**1) Compute (on paper) the gradient $\nabla f$, the gradient of $\nabla f_i$ and the gradient of the coordinate function $\frac{\partial f(w)}{\partial w_j}$ of $f$ for logistic regression (fill the class given below).**

################## TODO BLOCK

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

################## END TODO BLOCK

**2) Fill in the functions below for the computation of $f$, $\nabla f$, $\nabla f_i$ and $\frac{\partial f(w)}{\partial w_j}$ for logistic regression in the ModelLogReg class below.**


```python
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
        ################## TODO BLOCK
        return np.mean(np.log(1+np.exp(- y * (X.dot(w))))) + strength * norm(w) ** 2 / 2
        ################## END TODO BLOCK
       
    def grad(self, w):
        """Computes the gradient of f at w"""
        y, X, n_samples, strength = self.y, self.X, self.n_samples, self.strength
        ################## TODO BLOCK
        u = y*np.exp(- y * (X.dot(w)))/(1 + np.exp(- y * (X.dot(w))))
        return - (X.T.dot(u))/n_samples + strength * w
       ################## END TODO BLOCK
    
    def grad_i(self, i, w):
        """Computes the gradient of f_i at w"""
        x_i = self.X[i]
        strength = self.strength
        ################## TODO BLOCK
        u = y[i]*np.exp(- y[i] * (x_i.dot(w)))/(1 + np.exp(- y[i] * (x_i.dot(w))))
        return (- u*x_i + strength * w)
        ################## END TODO BLOCK
    
    def grad_coordinate(self, j, w):
        """Computes the partial derivative of f with respect to 
        the j-th coordinate"""
        y, X, n_samples, strength = self.y, self.X, self.n_samples, self.strength
        ################## TODO BLOCK
        u = y*np.exp(- y * (X.dot(w)))/(1 + np.exp(- y * (X.dot(w))))
        return - (X[:, j].T.dot(u))/n_samples + strength * w[j]
        ################## END TODO BLOCK
    
    def lip(self):
        """Computes the Lipschitz constant of the gradient of  f"""
        X, n_samples = self.X, self.n_samples
        ################## TODO BLOCK
        return norm(X.T.dot(X), 2) / (4*n_samples) + self.strength
        ################## ENDO TODO BLOCK

    def lip_coordinates(self):
        """Computes the Lipschitz constant of the gradient of f with respect to 
        the j-th coordinate"""
        X, n_samples = self.X, self.n_samples
        ################## TODO BLOCK
        return (X ** 2).sum(axis=0) / (4*n_samples) + self.strength
        ################## ENDO TODO BLOCK

    def lip_max(self):
        """Computes the maximum of the lipschitz constants of the gradients of f_i"""
        X, n_samples = self.X, self.n_samples
        ################## TODO BLOCK
        return ((X ** 2).sum(axis=1)/4 + self.strength).max()
        ################## ENDO TODO BLOCK
    
    def hessian(self):
        X, n_samples = self.X, self.n_samples
        ################## TODO BLOCK
        u = np.exp(- y * (X.dot(w)))/(1 + np.exp(- y * (X.dot(w))))
        M = np.diag(u) @ X
        return (M.T @ M)/n_samples + self.strength * np.eye(self.n_features)
        ################## END TODO BLOCK
        


```

<a id='models_logistic_check'></a>


## 2.4 Checks for the logistic regression model

**3) Use the function `simu_logreg` to simulate data according to the logistic regression model. Check numerically the gradient using the function ``checkgrad`` from ``scipy.optimize``, as we did for linear regression above.**


```python
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

    Checkgrad returns 1.58e-07



    
![png](Optimization2-GD-SGD-AGD-HB-Solution_files/Optimization2-GD-SGD-AGD-HB-Solution_22_1.png)
    



```python
print("lip=", model.lip())
print("lip_max=", model.lip_max())
print("lip_coordinates=", model.lip_coordinates())
```

    lip= 1.0045384237353767
    lip_max= 26.097945626980646
    lip_coordinates= [0.24 0.23 0.24 0.25 0.24 0.25 0.25 0.26 0.24 0.25 0.25 0.26 0.25 0.26
     0.25 0.23 0.26 0.26 0.24 0.25 0.26 0.25 0.25 0.24 0.24 0.25 0.25 0.25
     0.24 0.24 0.22 0.25 0.24 0.24 0.23 0.26 0.24 0.25 0.26 0.23 0.25 0.25
     0.26 0.24 0.26 0.24 0.25 0.25 0.24 0.25]


<a id='solvers'></a>
## 3. Solvers

We now have classes `ModelLinReg` and `ModelLogReg` that allow to compute $f(w)$, $\nabla f(w)$, 
$\nabla f_i(w)$ and $\frac{\partial f(w)}{\partial w_j}$ for the objective $f$
given by linear and logistic regression. We want now to code and compare several solvers to minimize $f$.


```python
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


```python
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


```python
from scipy.optimize import check_grad


print(check_grad(model.loss, model.grad, w0)) # This must be a number (of order 1e-6)
```

    2.0349870236764773e-07


<a id='tools'></a>
## 3.1 Tools for the solvers

The following tools store the loss after each epoch


```python
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





```python
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
        ################## TODO BLOCK
        w_new[:] = w - step * model.grad(w)
        w[:] = w_new # Remark : does operation inplace
        ################## END TODO BLOCK
        callback(w)
    return w

```

Now we compute the optimal loss $f_\star \triangleq f(w_\star)$.


```python
callback_long = inspector(model, n_iter=10000, verbose=False)
w_star = gd(model, w0, step=1/model.lip(), n_iter=10000, callback=callback_long, verbose=False)
obj_min = callback_long.objectives[-1]
```

The excess loss will be plotted using the below function.


```python
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


```python
n_iter=50
```


```python
callback_gd = inspector(model, n_iter=n_iter)
w_gd = gd(model, w0, step= 1/model.lip(),  n_iter=n_iter, callback=callback_gd)
```

    Lauching GD solver...
       it    |   obj   
           0 | 9.507018e-01
          10 | 3.058695e-01
          20 | 1.735939e-01
          30 | 1.410041e-01
          40 | 1.325032e-01
          50 | 1.301809e-01



```python
plot_callbacks([callback_gd], ["GD"], obj_min, "Gradient descent")
```

    /var/folders/c0/s8t504vd0w52xgnvj9cb82w40000gn/T/ipykernel_4815/3220526258.py:17: UserWarning: The figure layout has changed to tight
      plt.tight_layout()





    <module 'matplotlib.pyplot' from '/Users/aymericdieuleveut/anaconda3/lib/python3.11/site-packages/matplotlib/pyplot.py'>




    
![png](Optimization2-GD-SGD-AGD-HB-Solution_files/Optimization2-GD-SGD-AGD-HB-Solution_40_2.png)
    


**5) Which step size did you choose? What is the expected rate of convergence?**

################## TODO BLOCK

Using $\gamma = 1/L$, we expect a linear convergence rate as we have the following bound:
$$
\forall k \in \mathbb N, f(w_k) - f_\star \leq \left(1 - \frac{\mu}{L} \right)^{2k} (f(w_0) - f_\star)
$$

################## END TODO BLOCK


<a id='sgd'></a>
## 3.3 Stochastic gradient descent

**6) Complete the function `sgd` below that implements the stochastic gradient descent algorithm and test it using the next cell. Implement the Polyak-Ruppert averaging using an online update**

You can implement different strategy for the step size:
- Constant step-size: use $$\gamma = \frac{1}{2L}~.$$
- Decaying step size: use $$ \forall k \in \mathbb N, \gamma_k = \frac{1}{L\sqrt{k + 1}}~.$$


```python
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
                stepsize = step / (2*L)     ####### TODO OPERAND
            elif stepsize_strategy == "strongly_convex": ##### A enlever si pas traité en cours
                # For strongly-convex (choice in the slides)
                stepsize = step / max(mu*(it + 1), L)      ####### TODO OPERAND
            elif stepsize_strategy == "decaying":
                stepsize = step / (L * np.sqrt(it + 1))     ####### TODO OPERAND
            else:
                raise ValueError('The strategy is not correct')

            w -= stepsize * model.grad_i(i, w)      ####### TODO OPERAND

            if pr_averaging:
                # Polyak-Ruppert averaging
                w_averaged = it/(it+1)*w_averaged + 1/(it+1)*w      ####### TODO OPERAND
            it += 1
        if pr_averaging:
            callback(w_averaged)
        else:
            callback(w) 
    if pr_averaging:
        return w_averaged
    return w



```


```python
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

       it    |   obj   
           0 | 9.507018e-01
          10 | 1.511222e-01
          20 | 1.430586e-01
          30 | 1.383071e-01
          40 | 1.443209e-01
          50 | 1.421153e-01
       it    |   obj   
           0 | 9.507018e-01
          10 | 3.996369e-01
          20 | 3.044064e-01
          30 | 2.562626e-01
          40 | 2.260614e-01
          50 | 2.056450e-01
       it    |   obj   
           0 | 9.507018e-01
          10 | 1.347116e-01
          20 | 1.309255e-01
          30 | 1.300001e-01
          40 | 1.297700e-01
          50 | 1.295899e-01
       it    |   obj   
           0 | 9.507018e-01
          10 | 5.166539e-01
          20 | 4.229672e-01
          30 | 3.679644e-01
          40 | 3.306047e-01
          50 | 3.030793e-01





    array([-0.6 ,  0.41, -0.42,  0.4 , -0.3 ,  0.33, -0.29,  0.24, -0.25,
            0.2 , -0.14,  0.15, -0.18,  0.12, -0.12,  0.16, -0.11,  0.1 ,
           -0.07,  0.05,  0.05,  0.  , -0.  , -0.01,  0.01, -0.01,  0.01,
            0.04,  0.  , -0.02, -0.03, -0.02,  0.02,  0.02, -0.05, -0.04,
            0.04,  0.02,  0.01,  0.01,  0.02,  0.02, -0.03,  0.03,  0.02,
           -0.01,  0.  , -0.03,  0.03, -0.03])




```python
callbacks_sgd = [callback_sgd_constant, callback_sgd_decaying, callback_sgd_constant_PR, callback_sgd_decaying_PR]

names_sgd = ["SGD constant", "SGD decaying", "PR constant", "PR decaying"]


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


```python
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
        
        ################## TODO BLOCK
        w_new[:] = z - step * model.grad(z)
        
        if momentum_strategy == "constant":
            beta = 0.9      
        elif momentum_strategy == "convex":
            # See https://blogs.princeton.edu/imabandit/2018/11/21/a-short-proof-for-nesterovs-momentum/
            # Optimal momentum coefficinet for smooth convex
            t_new = (1. + sqrt(1. + 4. * t * t)) / 2. 
            beta = (t - 1) / t_new      
        elif momentum_strategy == "convex_approx":
            beta = k/(k+3) 
        elif momentum_strategy == "strongly_convex":
            # See https://blogs.princeton.edu/imabandit/2014/03/06/nesterovs-accelerated-gradient-descent-for-smooth-and-strongly-convex-optimization/
            if mu>0: ##### Regularization is used as a lower bound on the strong convexity coefficient
                kappa = (model.lip())/(mu) 
                beta = (sqrt(kappa) - 1)/(sqrt(kappa) + 1) # For strongly convex
            else:
                beta = k/(k+3)   
        else:
            raise ValueError('The momentum strategy is not correct')

        z[:] = w_new + beta * (w_new - w)  
        t = t_new  
        w[:] = w_new   
        ################## END TODO BLOCK      
        callback(w)
    return w



```


```python
callback_agd_constant = inspector(model, n_iter=n_iter)
callback_agd_convex = inspector(model, n_iter=n_iter)
callback_agd_convex_approx = inspector(model, n_iter=n_iter)
callback_agd_strongly_convex = inspector(model, n_iter=n_iter)

agd(model, w0, n_iter=n_iter, callback=callback_agd_constant, momentum_strategy="constant")
agd(model, w0, n_iter=n_iter, callback=callback_agd_convex, momentum_strategy="convex")
agd(model, w0, n_iter=n_iter, callback=callback_agd_convex_approx, momentum_strategy="convex_approx")
agd(model, w0, n_iter=n_iter, callback=callback_agd_strongly_convex, momentum_strategy="strongly_convex")
```

    Lauching AGD solver...
       it    |   obj   
           0 | 9.507018e-01
          10 | 1.998992e-01
          20 | 1.339184e-01
          30 | 1.307708e-01
          40 | 1.296213e-01
          50 | 1.292931e-01
    Lauching AGD solver...
       it    |   obj   
           0 | 9.507018e-01
          10 | 1.567916e-01
          20 | 1.319428e-01
          30 | 1.295094e-01
          40 | 1.292887e-01
          50 | 1.292625e-01
    Lauching AGD solver...
       it    |   obj   
           0 | 9.507018e-01
          10 | 1.627660e-01
          20 | 1.316508e-01
          30 | 1.294428e-01
          40 | 1.292826e-01
          50 | 1.292596e-01
    Lauching AGD solver...
       it    |   obj   
           0 | 9.507018e-01
          10 | 3.150460e-01
          20 | 1.573048e-01
          30 | 1.401637e-01
          40 | 1.361139e-01
          50 | 1.315355e-01





    array([-0.95,  0.85, -0.78,  0.69, -0.62,  0.63, -0.52,  0.47, -0.44,
            0.39, -0.34,  0.33, -0.31,  0.21, -0.24,  0.28, -0.22,  0.2 ,
           -0.17,  0.1 ,  0.04,  0.02, -0.02, -0.01,  0.01, -0.01,  0.02,
            0.05, -0.04, -0.01, -0.  , -0.02, -0.  ,  0.02, -0.04, -0.01,
            0.06, -0.01, -0.  ,  0.01,  0.01,  0.  , -0.02,  0.01,  0.04,
           -0.01, -0.02,  0.01,  0.03, -0.03])




```python
callbacks_agd = [callback_agd_constant, callback_agd_convex, callback_agd_convex_approx, 
                 callback_agd_strongly_convex]
names_agd = ["AGD constant", "AGD cvx", "AGD cvx approx", "AGD stgly cvx"]



```

<a id='hb'></a>
## 3.5. Heavy ball method

**8) Complete the function `hb` below that implements the Heavy ball (HB) method and test it using the next cell.**


```python
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


```python
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


```python
callback_hb = inspector(model, n_iter=n_iter)

heavy_ball_optimized(model, w0, n_iter=n_iter, callback=callback_hb)
```

    0.933633086901322 0.9397486207495583
       it    |   obj   
           0 | 9.507018e-01
          10 | 1.101912e+00
          20 | 6.438818e-01
          30 | 1.942308e-01
          40 | 2.644666e-01
          50 | 2.215504e-01





    array([-1.08,  0.95, -0.83,  0.64, -0.66,  0.65, -0.62,  0.44, -0.52,
            0.34, -0.42,  0.31, -0.31,  0.11, -0.26,  0.32, -0.31,  0.21,
           -0.22,  0.14,  0.02,  0.04, -0.06,  0.07, -0.04,  0.01,  0.01,
            0.08, -0.02, -0.04,  0.02,  0.06, -0.03,  0.07, -0.04,  0.02,
            0.1 , -0.05,  0.03,  0.07, -0.03,  0.05,  0.04, -0.07,  0.12,
            0.01, -0.04,  0.04,  0.03, -0.03])




```python

```