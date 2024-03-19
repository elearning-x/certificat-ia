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
  title: Deep Learning and Optimization
  version: '1.0'
---

 
<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Aymeric DIEULEVEUT</span>
<span>Licence CC BY-NC-ND</span>
</div>

+++

# Deep Learning and Optimization
## Lab 2 - Optimization

In this lab, you will apply different techniques to find the best parameter values to a simple linear regression problem. After defining the empirical risk of the corresponding problem, you will apply a **Grid Search strategy** to output an approximation of the best parameter based on the data set. As this strategy cannot be used for most of real-world data sets, you will then implement and compare **Gradient Descent (GD)** and **Stochastic Gradient Descent (SGD)**. 

We will use the dataset `height_weight_genders.csv`. We have provided
sample code templates that already contain useful snippets of code required for this lab.
You will be working in this notebook by filling in the corresponding functions. The
notebook already provides a lot of template codes, as well as a code to load the data, normalize the
features, and visualize the results.
If you have time, you can look at the files `helpers.py` and `plots.py`, and make sure you understand them.


<<<<<<<<< Temporary merge branch 1
```python
=========
```{code-cell} python
>>>>>>>>> Temporary merge branch 2
# Useful starting lines
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
%load_ext autoreload
%autoreload 2
%precision 2


```

# Part 1 - Loading the data set and preliminary analysis

Enough with simulated data! In this lab, you will be happy to know that we will use a real-world data set. However, first things first, we are going to study a very simple one. We will try to build a linear model of the weight based on the height. Yes, this is crazy ML!

Here is a short (and very basic) video to illustrate Linear Regression : https://www.youtube.com/watch?v=CtsRRUddV2s&ab_channel=VisuallyExplained


<<<<<<<<< Temporary merge branch 1
```python
=========
```{code-cell} python
>>>>>>>>> Temporary merge branch 2
import datetime
from helpers import *

#Load the data
#You need to check that the file helpers.py and height_weight_genders.csv are in the current folder
height, weight, gender = load_data(filename = "height_weight_genders.csv", sub_sample=False, add_outlier=False)
x, mean_x, std_x = standardize(height)

#Create the design matrix and the output vector 
y = weight
num_samples = len(y)
tx = np.c_[np.ones(num_samples), x]
```

We have at our disposal a dataset $\mathcal{D} = (x_i, y_i)_{i=1}^n$ with $n$ elements. In this exercise, we will implement a 1D linear regression which takes the following form:
$$
\forall j \in \{1, \cdots, n\}, y_j ≈ f(x_{j1} ) = w_0 + w_1 x_{j1} .
$$

We will use height as the input variable $x_{n1}$ and weight as the output variable $y_n$. The coefficients $w_0$ and $w_1$ are also called model parameters. Note this 1D regression contains 2 parameters: the slope and the intercept, this is the reason why tx is a set of 2D inputs, containing a column of 1.

Let us start by the array data type in NumPy. We store all the $(y_n, x_{n1})$ pairs in a vector and a matrix
as shown below:

$$
y =  \begin{bmatrix}
y_1 \\
y_2 \\
\vdots \\
y_n
\end{bmatrix} 
\qquad 
\tilde{X} = 
\begin{bmatrix}
1 & x_{11} \\
1 & x_{21} \\
\vdots & \vdots \\
1 & x_{n1} \\
\end{bmatrix} \,.
$$

We use the following cost function:
$$
\mathcal L : w \mapsto \frac{1}{2n} \| y - \tilde X w \|^2_2 \,.
$$



<<<<<<<<< Temporary merge branch 1
```python
=========
```{code-cell} python
>>>>>>>>> Temporary merge branch 2
plt.plot(tx[:, 1], y, '.')

plt.xlabel('Height')
plt.ylabel('Weight')
```

**1) To understand this data format, answer the following warmup questions:**

- How many input variables are we going to use?
- How many observations does the data set contain?
- What does each column of $\tilde X$ represent ?
- What does each row of $\tilde X$ represent ?
- Why do we have 1’s in $\tilde X$ ?
- If we have heights and weights of 3 people, what would be the size of $y$ and $\tilde X$ ? What would
$\tilde X_{32}$ represent ? 


In helpers.py, we have already provided code to form arrays for $y$ and $\tilde X$. Have a look at the code, and make sure you understand how they are constructed.


<<<<<<<<< Temporary merge branch 1
```python
=========
```{code-cell} python
>>>>>>>>> Temporary merge branch 2

#
#
# YOUR CODE OR ANSWER HERE
#
#

```

**2) We want to build a linear model to predict the weight as a function of the height. We consider the square loss. What is the optimization problem we want to solve?**


#
#
# YOUR CODE OR ANSWER HERE
#
#


**3) Compute the cost functions by running the code below.**


<<<<<<<<< Temporary merge branch 1
```python
=========
```{code-cell} python
>>>>>>>>> Temporary merge branch 2
def compute_loss(y, tx, w, loss="mse"):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    mse is used by default.
    """
    
    if loss == "mse":
        e = # YOUR CODE OR ANSWER HERE
        mse = # YOUR CODE OR ANSWER HERE
        return mse
    
    elif loss == "mae":
        e = # YOUR CODE OR ANSWER HERE
        mae = # YOUR CODE OR ANSWER HERE
        return mae
    else:
        raise ValueError("\"loss\" argument must be either \"mse\" or \"mae\". {} not permitted".format(loss))


```

**4) Is it possible to solve exactly the previous optimization problem? Justify.**


#
#
# YOUR CODE OR ANSWER HERE
#
#


**5) Compute the least square estimate $\hat{\beta}$ and the value of the loss at optimum. Comment.**


<<<<<<<<< Temporary merge branch 1
```python
=========
```{code-cell} python
>>>>>>>>> Temporary merge branch 2
def compute_exat_solution(y, tx):
    #
    # YOUR CODE OR ANSWER HERE
    #
    
w_star = compute_exat_solution(y, tx)
loss_at_opt = compute_loss(y, tx, w_star)
print("The exact solution of the least square problem is", w_star)
print("The value of the loss at optimum is {:.2f}.".format(loss_at_opt))


```

### Comments:


#
#
# YOUR CODE OR ANSWER HERE
#
#


In the following, we will compute and plot the **excess loss** $ \mathcal  L(w)- \mathcal L(w^{\star})$ instead of the loss, as most theoretical results provide guarantees on the excess loss.

# 2) Adopting a Grid Search approach

**6) We are now going to implement a Grid Search to find an approximate solution of our problem. Use the two following cells to do a Grid Search. What parameter can you tune to optimize the precision/computation time of Grid Search?**


<<<<<<<<< Temporary merge branch 1
```python
=========
```{code-cell} python
>>>>>>>>> Temporary merge branch 2
#Function that takes the data set and a list of parameters as input and output losses corresponding to each pair of parameters

def grid_search(y, tx, w0, w1):
    """Algorithm for grid search."""
    loss = np.zeros((len(w0), len(w1)))
    # compute loss for each combinationof w0 and w1.
    for ind_row, row in enumerate(w0):
        for ind_col, col in enumerate(w1):
            w = np.array([row, col])
            loss[ind_row, ind_col] = compute_loss(y, tx, w)-loss_at_opt
    return loss
```

Let us play with the grid search demo now!


<<<<<<<<< Temporary merge branch 1
```python
=========
```{code-cell} python
>>>>>>>>> Temporary merge branch 2
from grid_search import generate_w, get_best_parameters
from plots import grid_visualization

# Generate the grid of parameters to be swept
grid_w0, grid_w1 = generate_w(num_intervals=10)

# Start the grid search
start_time = datetime.datetime.now()
grid_losses = grid_search(y, tx, grid_w0, grid_w1)

# Select the best combinaison
loss_star, w0_star, w1_star = get_best_parameters(grid_w0, grid_w1, grid_losses)
end_time = datetime.datetime.now()
execution_time = (end_time - start_time).total_seconds()

# Print the results
print("Grid Search: loss*={l}, w0*={w0}, w1*={w1}, execution time={t:.3f} seconds".format(
      l=loss_star, w0=w0_star, w1=w1_star, t=execution_time))

# Plot the results
fig = grid_visualization(grid_losses, grid_w0, grid_w1, mean_x, std_x, height, weight)
fig.set_size_inches(10.0, 6.0)
fig.savefig("grid_plot")  # Optional saving
```

Discuss with your peers :

- Does this look like a good estimate ? Why not ? What is the problem ? Why is the MSE plot not smooth ?
- Repeat the above exercise by changing the grid spacing to 10 instead of 50. Compare the new fit to the old one.
- How does increasing the number of values affect the computational cost ? How fast or slow does your code run ?


#
#
# YOUR CODE OR ANSWER HERE
#
#


# 3) Implementing Gradient Descent

Here is a short (and very basic) video to illustrate GD : https://www.youtube.com/watch?v=qg4PchTECck&ab_channel=VisuallyExplained

**7) The first thing to do when implementing a Gradient Descent is to define the gradient of the loss. Then,  fill in the functions `compute_gradient` below and check that your implementation is correct.**

## Comments:


#
#
# YOUR CODE OR ANSWER HERE
#
#



<<<<<<<<< Temporary merge branch 1
```python
=========
```{code-cell} python
>>>>>>>>> Temporary merge branch 2
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = # YOUR CODE OR ANSWER HERE
    grad = # YOUR CODE OR ANSWER HERE
    return grad


print("Let's verify the gradient in the optimum as a sanity check!")
grad_opt = compute_gradient(y, tx, w_star)
print("The gradient at optimum is ", grad_opt, "which is equal to zero.")
```

**8) Fill in the function `gradient_descent` below.**

As we know from the lecture notes, the update rule for gradient descent at step $k$ is
$$
w_{k+1} = w_k - \gamma \nabla \mathcal L (w_k)
$$
where $\gamma > 0$ is the step size, and $\nabla \mathcal L \in \mathbb{R}^2$ is the gradient vector.



<<<<<<<<< Temporary merge branch 1
```python
=========
```{code-cell} python
>>>>>>>>> Temporary merge branch 2
def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute excess loss and gradient
        grad = # YOUR CODE OR ANSWER HERE
        loss = # YOUR CODE OR ANSWER HERE
        # gradient w by descent update
        w = # YOUR CODE OR ANSWER HERE
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): excess loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=round(loss, 4), w0=round(w[0], 4), w1=round(w[1], 4)))
    return losses, ws


```

**Test your gradient descent function through gradient descent demo shown below.**


<<<<<<<<< Temporary merge branch 1
```python
=========
```{code-cell} python
>>>>>>>>> Temporary merge branch 2
# from gradient_descent import *
from plots import gradient_descent_visualization

# Define the parameters of the algorithm.
max_iters = 31
gamma = 0.8

# Initialization
w_initial = np.array([0, 0])

# Start gradient descent.s
start_time = datetime.datetime.now()
gradient_losses, gradient_ws = gradient_descent(y, tx, w_initial, max_iters, gamma)
end_time = datetime.datetime.now()

# Print result
exection_time = (end_time - start_time).total_seconds()
print("Gradient Descent: execution time=%.3f seconds for %i iterations" %(exection_time, max_iters))

```


<<<<<<<<< Temporary merge branch 1
```python
=========
```{code-cell} python
>>>>>>>>> Temporary merge branch 2
# Time Visualization
from ipywidgets import IntSlider, interact
def plot_figure(n_iter):
    fig = gradient_descent_visualization(
        gradient_losses, gradient_ws, grid_losses, grid_w0, grid_w1, mean_x, std_x, height, weight, n_iter)
    fig.set_size_inches(10.0, 6.0)

interact(plot_figure, n_iter=IntSlider(min=1, max=len(gradient_ws)))
```

## Comments: 

- Is the cost being minimized ?
- Is the algorithm converging ? What can be said about the convergence speed ?
- How good are the final values of $w_1$ and $w_0$ found ?

##############

- The difference between the loss and its optimum reaches machine precision after 13 iterations.
- The algorithm converges as the excess loss reaches zero after 6 iterations (or machine precision after 13 iterations). 
- The values of $w_0$ and $w_1$ are very good, thus the linear line approximated well the point cloud.

##############

**9) Plot the evolution of the logarithm of the excess loss as a function of the number of iterations. What can be said about the convergence speed ?**


<<<<<<<<< Temporary merge branch 1
```python
=========
```{code-cell} python
>>>>>>>>> Temporary merge branch 2

#
#
# YOUR CODE OR ANSWER HERE
#
#


plt.xlabel('number of GD iterations')
plt.ylabel('log(L(w)-L(w^*))')

print("The algorithm converges and the convergence is linear in a log scale")



```

**10) Did we expect this behavior for the loss? Justify.**


#
#
# YOUR CODE OR ANSWER HERE
#
#



<<<<<<<<< Temporary merge branch 1
```python
=========
```{code-cell} python
>>>>>>>>> Temporary merge branch 2
# Computation of the Hessian matrix


hessian = # YOUR CODE OR ANSWER HERE


print("The hessian is:\n", hessian)

```

**11) Is the theoretical rate verified?**

HINT: Run gradient descent with a step size $\gamma=0.5$.


#
#
# YOUR CODE OR ANSWER HERE
#
#


**12) What is the maximal step size $\gamma$ you can choose? Try different values for $\gamma$. What do you notice?**


<<<<<<<<< Temporary merge branch 1
```python
=========
```{code-cell} python
>>>>>>>>> Temporary merge branch 2
def plot_excess_loss_of_gd_for_different_gamma(gammas):
    
    #
    #
    # YOUR CODE OR ANSWER HERE
    #
    #
    
    plt.title('Evolution of the excess loss with the number of iterations for various gamma')
    plt.xlabel('number of GD iterations')
    plt.ylabel('log(L(w)-L(w^*))')
    plt.legend(gammas)
    
plot_excess_loss_of_gd_for_different_gamma(gammas=[.1, .5, .9, 1., 1.2, 2, 2.2])

```


<<<<<<<<< Temporary merge branch 1
```python
=========
```{code-cell} python
>>>>>>>>> Temporary merge branch 2

```


#
#
# YOUR CODE OR ANSWER HERE
#
#


# 4) Implementing Stochastic Gradient Descent

Stochastic Gradient Descent is a classical extension of the Gradient Descent, particularly useful when dealing with very large data sets (large number of observations). 

Here is a short (and very basic) video to illustrate SGD : https://www.youtube.com/watch?v=UmathvAKj80&ab_channel=VisuallyExplained

**13) Fill in the following code to implement Stochastic Gradient Descent.**


<<<<<<<<< Temporary merge branch 1
```python
=========
```{code-cell} python
>>>>>>>>> Temporary merge branch 2
def compute_stochastic_gradient(y, tx, w, batch_size):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""

    # Draw randomly a batch of indices
    batch = # YOUR CODE OR ANSWER HERE
    
    # Extract the corresponding data
    y_batch = # YOUR CODE OR ANSWER HERE
    tx_batch = # YOUR CODE OR ANSWER HERE
    
    # Finally use the same formula as above to compute the gradient, this time only on the selected batch
    # YOUR CODE OR ANSWER HERE



def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent."""
    # Define parameters to store w and loss
    
    w = initial_w
    ws = [w]
    losses = [compute_loss(y, tx, w)]
        
    for n_iter in range(max_iters):
        # Compute a stochastic gradient and loss
        grad = # YOUR CODE OR ANSWER HERE
        # CONSTANT STEP SIZE
        w = # YOUR CODE OR ANSWER HERE
        
        
        # Calculate loss
        loss = # YOUR CODE OR ANSWER HERE
        
        # Store w and loss
        ws.append(w)
        losses.append(loss)


    print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws


```


<<<<<<<<< Temporary merge branch 1
```python
=========
```{code-cell} python
>>>>>>>>> Temporary merge branch 2
# Define the parameters of the algorithm.
max_iters = 100000
gamma = 0.5
batch_size = 16

# Initialization
w_initial = np.array([0, 0])

# Start SGD.
start_time = datetime.datetime.now()
sgd_losses, sgd_ws = stochastic_gradient_descent(
    y, tx, w_initial, batch_size, max_iters, gamma)
end_time = datetime.datetime.now()

# Print result
exection_time = (end_time - start_time).total_seconds()
print("SGD: execution ime=%.3f seconds for %i iterations" %(exection_time, max_iters))

```


<<<<<<<<< Temporary merge branch 1
```python
=========
```{code-cell} python
>>>>>>>>> Temporary merge branch 2
# Time Visualization
from ipywidgets import IntSlider, interact
def plot_figure(n_iter):
    fig = gradient_descent_visualization(
        sgd_losses, sgd_ws, grid_losses, grid_w0, grid_w1, mean_x, std_x, height, weight, n_iter)
    fig.set_size_inches(10.0, 6.0)

interact(plot_figure, n_iter=IntSlider(min=1, max=len(gradient_ws)))
```

**14) Plot the evolution of the logarithm of the excess loss as a function of the number of iterations.**


<<<<<<<<< Temporary merge branch 1
```python
=========
```{code-cell} python
>>>>>>>>> Temporary merge branch 2
# YOUR CODE OR ANSWER HERE


plt.title('Evolution of the excess loss with the number of iterations')
plt.xlabel('number of SGD iterations')
plt.ylabel('log(L(w)-L(w^*))')


```

**15) How does the choice of the step size impact the convergence of SGD?**


<<<<<<<<< Temporary merge branch 1
```python
=========
```{code-cell} python
>>>>>>>>> Temporary merge branch 2
def plot_excess_loss_of_sgd_for_different_gamma(gammas):
    
    #
    #
    # YOUR CODE OR ANSWER HERE
    #
    #
    
    plt.title('Evolution of the excess loss with the number of iterations for gamma={}'.format(gamma))
    plt.xlabel('number of GD iterations')
    plt.ylabel('log(L(w)-L(w^*))')
    plt.legend(gammas)
```


<<<<<<<<< Temporary merge branch 1
```python
=========
```{code-cell} python
>>>>>>>>> Temporary merge branch 2
plot_excess_loss_of_sgd_for_different_gamma(gammas=[.1, .2, .5, 1.])

```

### Comments:


#
#
# YOUR CODE OR ANSWER HERE
#
#


**16) Plot the evolution of the log of the excess loss for SGD and for GD in the same graph. 
What is the complexity per iteration of GD, SGD? Compare the theoretical complexity and the time required per iteration for the algorithm. Interpret.**


<<<<<<<<< Temporary merge branch 1
```python
=========
```{code-cell} python
>>>>>>>>> Temporary merge branch 2
plt.figure(figsize=(10,10))

#
#
# YOUR CODE OR ANSWER HERE
#
#

plt.title('Evolution of the excess loss with the number of passes on the data = theoretical complexity')
plt.xlabel('number of points used iterations')
plt.ylabel('log(L(w)-L(w^*))')


```

### Comments:


#
#
# YOUR CODE OR ANSWER HERE
#
#


Another approach is to use decaying step size for SGD. This way, when we get closer to the neighborhood of the optimal point, the step size is reduced, reducing oscillation and improving the convergence toward the optimum.

**17) Plot the evolution of the log of the excess loss for SGD with decaying step size, and for GD in the same graph.**


<<<<<<<<< Temporary merge branch 1
```python
=========
```{code-cell} python
>>>>>>>>> Temporary merge branch 2
def stochastic_gradient_descent_decaying_step_sizes(y, tx, initial_w, batch_size, gammas):
    """Stochastic gradient descent."""
    # Define parameters to store w and loss
    
    w = initial_w
    ws = [w]
    losses = [compute_loss(y, tx, w)]
        
    for n_iter in range(len(gammas)):
        # compute a stochastic gradient and loss with DECAYING STEP SIZE
        
        #
        #
        # YOUR CODE OR ANSWER HERE
        #
        #
        

    print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws


sgd_losses, _ = stochastic_gradient_descent_decaying_step_sizes(y, tx, [0, 0], 16, 1/np.arange(1, 10 * num_samples+1))
sgd_losses_squared, _ = stochastic_gradient_descent_decaying_step_sizes(y, tx, [0, 0], 16, 1/np.sqrt(np.arange(1, 10 * num_samples+1)))
```


<<<<<<<<< Temporary merge branch 1
```python
=========
```{code-cell} python
>>>>>>>>> Temporary merge branch 2
plt.figure(figsize=(10,10))

#
#
# YOUR CODE OR ANSWER HERE
#
#


plt.title('Evolution of the excess loss with the number of passes on the data = theoretical complexity')
plt.xlabel('number of points used iterations')
plt.ylabel('log(L(w)-L(w^*))')


```

As shown in the lecture notes, we could also compute the **averaged iterate**: 
$$\bar w_k= \frac{1}{k} \sum_{i=1}^k w_i,$$
which reduces the effect of the noise and improves a lot the convergence. No need to store all previous iterations, the average iterate $\bar w_k$ can be computed online via
$$\bar w_k= \frac{k-1}{k} \bar w_{k-1} +\frac{1}{k} w_k.$$

**18) Plot the evolution of the log of the excess loss for SGD with Polyak-Ruppert averaging, and for GD in the same graph.**


<<<<<<<<< Temporary merge branch 1
```python
=========
```{code-cell} python
>>>>>>>>> Temporary merge branch 2
def stochastic_gradient_descent_rupert_averaging(y, tx, initial_w, batch_size, gammas):
    """Stochastic gradient descent."""
    # Define parameters to store w and loss
    
    w = initial_w
    ws = [w]
    losses = [compute_loss(y, tx, w)]
    
    wave = np.array(initial_w)
    
    for n_iter in range(len(gammas)):
        
        #
        #
        # YOUR CODE OR ANSWER HERE
        #
        #
        

    print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws


```


<<<<<<<<< Temporary merge branch 1
```python
=========
```{code-cell} python
>>>>>>>>> Temporary merge branch 2
sgd_losses, _ = stochastic_gradient_descent_rupert_averaging(y, tx, [0, 0], 16, 1/np.arange(1, 10 * num_samples+1))
sgd_losses_sqrt, _ = stochastic_gradient_descent_rupert_averaging(y, tx, [0, 0], 16, 1/np.sqrt(np.arange(1, 10 * num_samples+1)))


plt.figure(figsize=(10,10))
plt.semilogy(num_samples * np.arange(10), gradient_losses[:10], 'r', label='GD')
plt.semilogy(sgd_losses[:10 * num_samples], label='SGD with Rupert avering and 1/k decaying')
plt.semilogy(sgd_losses_sqrt[:10 * num_samples], label='SGD with Rupert avering and 1/sqrt(k) decaying')

plt.legend()
plt.title('Evolution of the excess loss with the number of passes on the data = theoretical complexity')
plt.xlabel('number of points used iterations')
plt.ylabel('log(L(w)-L(w^*))')

```

In both final iterate and average iterate, we observe a good behavior of SGD that converges.
However, its convergence is still asymptotically slower than GD, and SGD is mostly useful in the first steps.

This is a reason why some people increase the batch size during the training to reduce the noise and hence getting faster along the training.

**19) We have solved a very specific problem since it is quadratic and the Hessian of the risk is the identity. For most problems, the hessian matrix won't be the identity. We could create a problem in which the Hessian is a non-diagonal covariance matrix, by having 2 explanatory variables that are not independent. For example, we can use height and height^3. You can re-run the entire lab with $tx$ re-defined with those two features.**


<<<<<<<<< Temporary merge branch 1
```python
=========
```{code-cell} python
>>>>>>>>> Temporary merge branch 2

#
#
# YOUR CODE OR ANSWER HERE
#
#

```

This matrix has eigenvalues 1,81 and 0,19, thus a condition number of 9. We expect to have a maximal step size of 2/L, with L=1,81 that is around 1.1.
