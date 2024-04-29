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
  title: 'Optimization 1: visualization, Taylor, convexity'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Aymeric DIEULEVEUT</span>
<span>Licence CC BY-NC-ND</span>
</div>

+++

# Exploring Convex and Non-Convex Functions

## Part 1: Understanding Convexity and Non-Convexity

In this lab, we  explore  properties of convex and non-convex functions. In particular, we  look at their graphical representations, analyze their behavior through tangents and Taylor approximations, and study their level sets and gradients.

### Objectives:
- To understand and visualize convex and non-convex functions.
- To plot tangents and Taylor approximations for given functions.
- To visualize level sets and compute gradients.

## Part 2: Machine learning - link between method and convexity, smoothness, etc.

In the second part, we consider binary classification with 01 loss, logistic loss, hinge loss and a small neural network

### Objectives:
- Relate the choice of the method to the regularity of the optimization problem

## Part 1: Understanding Convexity and Non-Convexity


```{code-cell} python
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Enable inline plotting
%matplotlib inline
```

Let us introduce two simple one dimensional function - feel free to modify those


```{code-cell} python
# Define symbolic variables
x, y = sp.symbols('x y')

# Function definitions (modify as needed for different parts)
def f_quadratic(x):
    return x**2

def f_non_quadratic(x):
    return sp.cos(x) + x**3
```

We use the sympy package to compute the derivatives symbolically


```{code-cell} python
# First and second derivatives using SymPy
def diff_functions(func, symbol):
    first = sp.diff(func, symbol)
    second = sp.diff(first, symbol)
    return first, second
```

#### Using symbolic computation


```{code-cell} python
# to access the gradient of the function at a given point, use  ``func.subs(x, point)``
f_quadratic(x).subs(x, 3), f_non_quadratic(x).subs(x, 3)
```



```{code-cell} python
# Get the first 2 derivatives of f
diff_functions(f_quadratic(x), x)
```




```{code-cell} python
# Evaluate the first derivative at x= -5
derivs = diff_functions(f_quadratic(x), x)

derivs[0].subs(x,-5)
```




### Question 1: 
- recall the formula for Taylor approximations
- Complete the following code to plot the first and second order approaximation of the function?




```{code-cell} python
# Plotting function with tangents and Taylor approximations
def fisrt_order_taylor(func, point):
    
    
    # Access the first two derivatives of f  using diff_functions
    derivs = diff_functions(func, x) #TODO OPERAND
    
    # Compute the 1st order taylor (tangent)
    taylor_fst_ord = derivs[0].subs(x, point)*(x - point) + func.subs(x, point) # TODO OPERAND
    return(taylor_fst_ord)

def second_order_taylor(func, point):
     # Convert symbolic to lambda for numerical evaluation
    f_lamb = sp.lambdify(x, func, 'numpy')
    
    # Access the first two derivatives of f  using diff_functions
    derivs = diff_functions(func, x) #TODO OPERAND
    
    # Compute the 2nd order taylor approximation
    #TODO BLOCK
    taylor_scd_ord = func.subs(x, point)
    for i in range(1, 3):
        taylor_scd_ord += derivs[i-1].subs(x, point) * (x - point)**i / sp.factorial(i)
    #END TODO BLOCK

    return(taylor_scd_ord)

    
def plot_function_and_taylor(func, x_range, point, order=2):
    
    
    taylor_fst_ord = fisrt_order_taylor(func, point) # TODO OPERAND
    taylor_scd_ord = second_order_taylor(func, point)# TODO OPERAND
        
        
    # Convert symbolic to lambda for numerical evaluation
    f_lamb = sp.lambdify(x, func, 'numpy')     
    taylor_fst_ord_lamb = sp.lambdify(x, taylor_fst_ord, 'numpy')# TODO OPERAND
    taylor_scd_ord_lamb = sp.lambdify(x, taylor_scd_ord, 'numpy')# TODO OPERAND

    x_vals = np.linspace(x_range[0], x_range[1], 400)
    y_vals = f_lamb(x_vals)
    taylor_fst_ord_vals = taylor_fst_ord_lamb(x_vals)
    taylor_scd_ord_vals = taylor_scd_ord_lamb(x_vals)

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label='Function')
    plt.plot(x_vals, taylor_fst_ord_vals, '--', label='1st order Taylor (Tangent) at x={}'.format(point))
    plt.plot(x_vals, taylor_scd_ord_vals, ':', label='2nd order Taylor Approximation at x={}'.format(point))
    plt.scatter([point], [f_lamb(point)], color='red') # point of tangency
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Function, Tangent, and Taylor Approximation')
    plt.grid(True)
    plt.show()

# Example usage
plot_function_and_taylor(f_quadratic(x), (-10, 10), 1)
plot_function_and_taylor(f_non_quadratic(x), (-10, 10), 1)
```



### Question 2:
- What can you say about the second order taylor approximation for a quadratic function
- Is the first function convex? How is that visible w.r.t. first order approximations?
- Is the second function convex? How is that visible w.r.t. first order approximations? Make a plot illustrating the non-convexity of the function
- Are those fucntions smooth? Compute an upper bound on the Hessian on the range, and plot the second order quadratic upper bound


```{code-cell} python
def plot_function_second_order_upper_bound(func, x_range, point, order=2):
  
    first_deriv, second_deriv = diff_functions(func, x)      # TODO OPERAND
    second_deriv_lamb = sp.lambdify(x, second_deriv, 'numpy')# TODO OPERAND
    x_vals = np.linspace(x_range[0], x_range[1], 400)
    second_deriv_vals = second_deriv_lamb(x_vals)
    
    M = np.max(second_deriv_vals) # TODO OPERAND
    print(M)
    
    # Compute the 2nd order quadratic upper bound dur to smoothness
    #TODO BLOCK
    Quad_UB_scd_ord = func.subs(x, point)+ first_deriv.subs(x, point) * (x - point)+ M*(x - point)**2 / 2
    #END TODO BLOCK
    
     # Convert symbolic to lambda for numerical evaluation
    f_lamb = sp.lambdify(x, func, 'numpy')     
    taylor_fst_ord = fisrt_order_taylor(func, point) 
    y_vals = f_lamb(x_vals)
    taylor_fst_ord_lamb = sp.lambdify(x, taylor_fst_ord, 'numpy')

    taylor_fst_ord_vals = taylor_fst_ord_lamb(x_vals)
    
    Quad_UB_scd_ord_lamb  = sp.lambdify(x, Quad_UB_scd_ord, 'numpy')  
    Quad_UB_scd_ord_vals = Quad_UB_scd_ord_lamb(x_vals)

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label='Function', lw='2')
    plt.plot(x_vals, taylor_fst_ord_vals, '--', label='1st order Taylor (Tangent) at x={}'.format(point), lw='2')
    plt.plot(x_vals, Quad_UB_scd_ord_vals, ':', label='Quadratic upper bound at x={}'.format(point), lw='2')
    plt.scatter([point], [f_lamb(point)], color='red') # point of tangency
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Function, Tangent, and Taylor Approximation')
    plt.grid(True)
    plt.show()

# Example usage
plot_function_second_order_upper_bound(f_quadratic(x), (-5, 5), 3)
plot_function_second_order_upper_bound(f_non_quadratic(x), (0, 5), 3)
```



### Question 3:
Plot the level sets of the function `f(x, y) = x^2 + y^2` and `g(x, y) = cos(x) + sin(y)`. What do these plots indicate about the convexity of the functions?


```{code-cell} python
# Define the functions for level sets
def f(x, y):
    return x**2 + y**2

def g(x, y):
    return np.cos(x) + np.sin(y)

x_vals = np.linspace(-5, 5, 400)
y_vals = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z_f = f(X, Y)
Z_g = g(X, Y)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)

# Use plt.contour to plot the level sets
plt.contour(X, Y, Z_f, levels=20) #TODO LINE
plt.title('Level sets of $f(x, y) = x^2 + y^2$')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

# Use plt.contour to plot the level sets
plt.subplot(1, 2, 2)
plt.contour(X, Y, Z_g, levels=20) #TODO LINE
plt.title('Level sets of $g(x, y) = \cos(x) + \sin(y)$')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

plt.show()
```



### Question 4:
Compute and plot the gradient of `f(x, y) = x^2 + y^2` at point `(1,1)`. How does the gradient relate to the level sets?


```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Define symbolic variables
x, y = sp.symbols('x y')

# Define the function f(x, y) = x^2 + y^2
f = 4*x**2 + y**2

# Create a lambda function for f
f_lambda = sp.lambdify((x, y), f, "numpy")

# Compute the gradient of f
gradient_f = [sp.diff(f, var) for var in (x, y)] # TODO OPERAND
gradient_f_lambda = [sp.lambdify((x, y), grad, "numpy") for grad in gradient_f] 

# Evaluate the function over a grid
x_vals = np.linspace(-2, 2, 400)
y_vals = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f_lambda(X, Y)

# Evaluate the gradient at the point (1, 1)
gradient_at_point = [g(1, 1) for g in gradient_f_lambda] # TODO OPERAND

# Plotting
plt.figure(figsize=(6, 6))
contour = plt.contour(X, Y, Z, levels=20)
plt.clabel(contour, inline=True, fontsize=8)
plt.quiver(1, 1, -gradient_at_point[0], -gradient_at_point[1], scale=20, color='red', label='Negative Gradient at (1, 1)')
plt.title('Contour Plot of $f(x, y) = x^2 + y^2$ with Gradient')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
```


## Part 2: Machine learning - link between method and convexity, smoothness, etc.

In the second part, we consider binary classification with 01 loss, logistic loss, hinge loss and a small neural network

### Objectives:
- Relate the choice of the method to the regularity of the optimization problem

## Setup for Machine Learning Part of the Lab

We generate a Toy Dataset: we will simulate a simple binary classification task using a logistic model approach.

#### Step 1: Generate the Toy Dataset

We'll start by creating a simple dataset where the feature values are drawn from a normal distribution, and the labels are assigned depending on x.


```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

# Seed for reproducibility
np.random.seed(43)

n_samples =25 
# Generate synthetic data
x = np.random.normal(0, 1, size=n_samples )
y = (2*(x+0.5*np.random.normal(0, 1, size=n_samples ) > 0)-1).astype(int)  # Binary labels in -1 , 1

# Visualize the data
plt.figure(figsize=(8, 5))
plt.scatter(x, y, c=y, cmap='bwr', marker='o', edgecolor='k')
plt.xlabel('Feature Value')
plt.ylabel('Label')
plt.title('Toy Binary Classification Data')
plt.show()
```



## 2.1 One dimensional parameter space

In this subsection, we focus on a single dimension parameter w.
To predict at point x, we use $$g_w(x) = x-w$$

To output a binary value, we then consider the $$sign(g_w(x)) = 2* 1_{g_w(x)>0}-1$$


```{code-cell} python
#Compute the zero one empirical risk (ER) on the data for predictor w
def zero_one_ER(w, x, y):
    #TODO BLOCK
    predictions = 2*(x > w).astype(int)-1
    zero_one_ER = np.mean(predictions != y)
    #END TODO BLOCK
    return (zero_one_ER) 

# Range of w values to evaluate
w_values = np.linspace(x.min(), x.max(), 300)
loss_01 = [zero_one_ER(w, x, y) for w in w_values]

plt.figure(figsize=(8, 5))
plt.plot(w_values, loss_01, label='0-1 Loss')
plt.xlabel('Decision Threshold (w)')
plt.ylabel('Loss')
plt.title('0-1 ER vs. Decision Threshold')
plt.legend()
plt.grid(True)
plt.show()
```



### Question 5:
- Is the 01 empirical risk convex as a function of w?
- Is it differentiable?
- What is its derivative

### Question 6:
- Recall the formula for the logistic loss
- Compute the logistic Empirical Risk


```{code-cell} python
def logistic_ER(w, x, y):
        # TODO BLOCK
    z = x - w
    log_ER = np.mean(np.log(1 + np.exp(- y * z))) 
        #END TODO BLOCK
    return log_ER

loss_logistic = [logistic_ER(w, x, y) for w in w_values]

plt.figure(figsize=(8, 5))
plt.plot(w_values, loss_logistic, label='Logistic Loss', color='green')
plt.xlabel('Decision Threshold (w)')
plt.ylabel('Loss')
plt.title('Logistic ER vs. Decision Threshold')
plt.legend()
plt.grid(True)
plt.show()
```



### Question 7:
- Is the logistic empirical risk convex as a function of w?
- Is it differentiable? Smooth?
- What is its derivative?

### Question 8:
- Recall the formula for the hinge loss
- Compute the hinge Empirical Risk


```{code-cell} python
def hinge_ER(w, x, y):
    # TODO BLOCK
    z = x - w
    h_ER = np.mean(np.maximum(0, 1 -y  * z))
    #END TODO BLOCK
    return  h_ER

loss_hinge = [hinge_ER(w, x, y) for w in w_values]

plt.figure(figsize=(8, 5))
plt.plot(w_values, loss_hinge, label='Hinge Loss', color='orange')
plt.xlabel('Decision Threshold (w)')
plt.ylabel('Loss')
plt.title('Hinge ER vs. Decision Threshold')
plt.legend()
plt.grid(True)
plt.show()
```



### Question 9:
- Is the hinge empirical risk convex as a function of w?
- Is it differentiable? Smooth?


## Putting things together


```{code-cell} python

# Range of w values to evaluate
w_values = np.linspace(x.min(), x.max(), 300)
losses_logistic = [logistic_ER(w, x, y) for w in w_values]
losses_hinge = [hinge_ER(w, x, y) for w in w_values]

# Plotting the loss functions
plt.figure(figsize=(10, 6))
plt.plot(w_values, losses_logistic, label='Logistic Loss', color='green')
plt.plot(w_values, losses_hinge, label='Hinge Loss', color='orange')
plt.plot(w_values, loss_01, label='0-1 Loss')
plt.xlabel('Decision Threshold (w)')
plt.ylabel('Loss')
plt.title('Comparison of Logistic and Hinge Losses')
plt.legend()
plt.grid(True)
plt.show()
```



## 2.1 Two dimensional parameter space

In this subsection, we focus on a single dimension parameter w.
To predict at point x, we use $$g_{w}(x) = <w,x>$$

To output a binary value, we then consider the $$sign(g_w(x)) = 2* 1_{g_w(x)>0}-1$$


```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs
# Generate non-linearly separable data
n_samples = 60
centers = [(-1, -1), (1, 1)]  # Define centers for two clusters
cluster_std = 1.5  # Standard deviation of clusters

X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=42)

# Convert labels from {0, 1} to {-1, 1}
y = 2 * y - 1

# Visualize the dataset
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
plt.title('Non-linearly Separable Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()
```



### Question 10:
- Complete the codes to compute the ER for each loss



```{code-cell} python
def zero_one_ER(w, X, y):
    predictions = np.sign(np.dot(X, w))       #TODO LINE
    zo_ER = np.mean(predictions != y)      #TODO LINE
    return zo_ER

def logistic_ER(w, X, y):
    z = np.dot(X, w)
    logistic_prob = 1 / (1 + np.exp(-y * z))      #TODO LINE
    log_ER =-np.mean(np.log(logistic_prob))      #TODO LINE
    return log_ER       

def hinge_ER(w, X, y):
    z = np.dot(X, w)      #TODO LINE
    H_ER = np.mean(np.maximum(0, 1 - y * z))      #TODO LINE
    return H_ER
```



### Question 10:
- What can you say about the level sets?
- Which ER is convex as a 2 dimensional function?



```{code-cell} python
# Create a grid of weight values
w1 = np.linspace(-4, 4, 100)
w2 = np.linspace(-4, 4, 100)
W1, W2 = np.meshgrid(w1, w2)
losses_01 = np.zeros_like(W1)
losses_logistic = np.zeros_like(W1)
losses_hinge = np.zeros_like(W1)

# Evaluate losses
for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        w = np.array([W1[i, j], W2[i, j]])
        losses_01[i, j] = zero_one_ER(w, X, y)
        losses_logistic[i, j] = logistic_ER(w, X, y)
        losses_hinge[i, j] = hinge_ER(w, X, y)

# Plotting level sets of the losses
plt.figure(figsize=(18, 6))

# 0-1 Loss
plt.subplot(1, 3, 1)
plt.contourf(W1, W2, losses_01, levels=20, cmap='viridis')
plt.colorbar()
plt.title('0-1 Loss')
plt.xlabel('Weight 1')
plt.ylabel('Weight 2')

# Logistic Loss
plt.subplot(1, 3, 2)
plt.contourf(W1, W2, losses_logistic, levels=20, cmap='viridis')
plt.colorbar()
plt.title('Logistic Loss')
plt.xlabel('Weight 1')

# Hinge Loss
plt.subplot(1, 3, 3)
plt.contourf(W1, W2, losses_hinge, levels=20, cmap='viridis')
plt.colorbar()
plt.title('Hinge Loss')
plt.xlabel('Weight 1')

plt.tight_layout()
plt.show()


# Plotting level sets of the losses using contour lines
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
contour_01 = plt.contour(W1, W2, losses_01, levels=50, cmap='viridis')
plt.clabel(contour_01, inline=True, fontsize=8)
plt.title('0-1 Loss')
plt.xlabel('Weight 1')
plt.ylabel('Weight 2')

plt.subplot(1, 3, 2)
contour_logistic = plt.contour(W1, W2, losses_logistic, levels=50, cmap='viridis')
plt.clabel(contour_logistic, inline=True, fontsize=8)
plt.title('Logistic Loss')
plt.xlabel('Weight 1')

plt.subplot(1, 3, 3)
contour_hinge = plt.contour(W1, W2, losses_hinge, levels=50, cmap='viridis')
plt.clabel(contour_hinge, inline=True, fontsize=8)
plt.title('Hinge Loss')
plt.xlabel('Weight 1')

plt.tight_layout()
plt.show()
```
 


### We plot the optimal parameter and the associated decision frontier


```{code-cell} python
# Find the indices of the minimum values
min_idx_01 = np.unravel_index(np.argmin(losses_01), losses_01.shape)
min_idx_logistic = np.unravel_index(np.argmin(losses_logistic), losses_logistic.shape)
min_idx_hinge = np.unravel_index(np.argmin(losses_hinge), losses_hinge.shape)

# Retrieve the optimal weight vectors
optimal_w_01 = np.array([W1[min_idx_01], W2[min_idx_01]])
optimal_w_logistic = np.array([W1[min_idx_logistic], W2[min_idx_logistic]])
optimal_w_hinge = np.array([W1[min_idx_hinge], W2[min_idx_hinge]])
```


```{code-cell} python
def plot_decision_boundary(w, label, color):
    # Create a line from weights
    slope = -w[0] / w[1]
    intercept = 0  # Since we are not using bias
    x_values = np.array([X[:, 0].min(), X[:, 0].max()])
    y_values = slope * x_values + intercept
    plt.plot(x_values, y_values, color=color, label=label)
```


```{code-cell} python
# Plotting the level sets and optimal points
plt.figure(figsize=(18, 6))


# Logistic Loss
plt.subplot(1, 3, 2)
contour_logistic = plt.contour(W1, W2, losses_logistic, levels=50, cmap='viridis')
plt.scatter(optimal_w_logistic[0], optimal_w_logistic[1], color='red', marker='*', s=200, label='Optimal Point')
plt.clabel(contour_logistic, inline=True, fontsize=8)
plt.title('Logistic Loss Level Sets')
plt.xlabel('Weight 1')
plt.legend()

# Hinge Loss
plt.subplot(1, 3, 3)
contour_hinge = plt.contour(W1, W2, losses_hinge, levels=50, cmap='viridis')
plt.scatter(optimal_w_hinge[0], optimal_w_hinge[1], color='red', marker='*', s=200, label='Optimal Point')
plt.clabel(contour_hinge, inline=True, fontsize=8)
plt.title('Hinge Loss Level Sets')
plt.xlabel('Weight 1')
plt.legend()

plt.tight_layout()
plt.show()
```



```{code-cell} python
# Plotting decision boundaries and training points
plt.figure(figsize=(18, 6))

# Logistic Loss
plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k', alpha=0.7)
plot_decision_boundary(optimal_w_logistic, 'Decision Boundary', 'red')
plt.title('Logistic Loss Decision Boundary')
plt.xlabel('Feature 1')
plt.legend()

# Hinge Loss
plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k', alpha=0.7)
plot_decision_boundary(optimal_w_hinge, 'Decision Boundary', 'red')
plt.title('Hinge Loss Decision Boundary')
plt.xlabel('Feature 1')
plt.legend()

plt.tight_layout()
plt.show()
```


