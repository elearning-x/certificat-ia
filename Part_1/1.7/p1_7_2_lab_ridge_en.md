---
jupytext:
  cell_metadata_filter: all, -hidden, -heading_collapsed, -run_control, -trusted
  notebook_metadata_filter: all, -jupytext.text_representation.jupytext_version, -jupytext.text_representation.format_version, -language_info.version, -language_info.codemirror_mode.version, -language_info.codemirror_mode, -language_info.file_extension, -language_info.mimetype, -toc
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
  title: 'Lab session on Ridge Regression'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa Bedin<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

## Modules



### Importing modules



Import the modules pandas (as `pd`) and numpy (as `np`)
Import the sub-module `pyplot` from `matplotlib` as `plt`
Import the function `StandardScaler` from `sklearn.preprocessing`
Import the function `Ridge` from `sklearn.linear_model`
Import the function `RidgeCV` from `sklearn.linear_model`
Import the function `Pipeline` from `sklearn.pipeline`
Import the function `cross_val_predict` from `sklearn.model_selection`
Import the function `KFold` from `sklearn.model_selection`


```{code-cell} python

```

## Ridge Regression on Ozone Data



### Importing Data



Import the ozone data `ozonecomplet.csv` (in Fun Campus, data is in `data/`) and remove the last two qualitative variables
Summarize each variable using methods `astype` on DataFrame columns and `describe` on DataFrame instance




```{code-cell} python

```

### Creating numpy Arrays



Create numpy arrays `y` and `X` using instance methods `iloc` or `loc` (using the underlying `values` attribute of DataFrame)




```{code-cell} python

```

### Centering and Scaling



Center and scale variables using `StandardScaler` with the following steps:

1.  Create an instance using `StandardScaler`
2.  Fit the instance using `fit` method with the numpy array `X`
3.  Transform the array `X` to a scaled array using `transform` method




```{code-cell} python

```

### Ridge Regression Calculation for $\lambda=0.00485$



1.  Estimation/fitting: Use centered and scaled data for $X$ and vector `y` to estimate Ridge regression model:
    -   Instantiate a `Ridge` model (note: in scikit-learn, $\lambda$ is denoted as $\alpha$ for Ridge, Lasso, and Elastic-Net)
    -   Fit the model with $\lambda=0.00485$ using `fit` method
2.  Display $\hat\beta(\lambda)$
3.  Predict a value for $x^*=(17, 18.4, 5, 5, 7, -4.3301, -4, -3, 87)'$ (the second row of the initial table)




```{code-cell} python

```

### Pipeline



Since we need to remove mean and divide by standard deviation for new values, let&rsquo;s automate the process:

-   Verify that `scalerX.transform(X[1,:].reshape(1, 10))` yields `Xcr[1,:]`
-   Automate the sequence &ldquo;transform X&rdquo; followed by &ldquo;modeling&rdquo; using a [Pipeline](https://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html):
    1.  Create a `StandardScaler` instance
    2.  Create a Ridge Regression instance
    3.  Create a `Pipeline` instance with `steps` argument as a list of tuples (step name and instance from previous steps)
    4.  Fit this pipeline instance using `fit` method with `X` and `y` data
    5.  Retrieve $\hat\beta(\lambda)$ by accessing the &ldquo;ridge&rdquo; (chosen step name) coordinate of the `named_steps` attribute
    6.  Retrieve adjustment for $x^*$




```{code-cell} python

```

### Coefficient Evolution with $\lambda$



#### Calculating a $\lambda$ grid



Create a grid similar to lasso grid, based on:

1.  Calculate maximum value $\lambda_0 = \arg\max_{i} |[X'y]_i|/n$
2.  Use a power of 10 grid, with exponents varying from 0 to -4 (typically, 100 regularly spaced values)
3.  Multiply the grid by $\lambda_0$
4.  For Ridge regression, multiply the above grid (lasso grid) by $100$ or $1000$

Create this grid using `np.linspace`, `transpose` method, `dot` and `max` (don&rsquo;t forget the `shape` attribute for $n$)




```{code-cell} python

```

#### Plotting Coefficient Evolution with $\lambda$



Plot the coefficients $\hat\beta(\lambda)$ against the logarithm of $\lambda$ values from the grid




```{code-cell} python

```

### Optimal $\hat\lambda$ (by 10-fold Cross Validation)



#### Splitting into 10 Folds



Split the data into 10 folds using the [KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold) function, creating an instance named `kf`




```{code-cell} python

```

#### Selecting Optimal $\hat\lambda$



1.  Create a DataFrame `res` with 100 columns of zeros
2.  Loop through all folds; use `split` method on `kf` with `X` data
3.  For each fold iteration:
    
    1.  Estimate Ridge models for each $\lambda$ from the grid using data from 9 training folds
    2.  Predict the data from the validation fold
    3.  Store the predicted values in corresponding rows of `res` for the 100 Ridge models
    
    Calculate the optimal model (and $\hat\lambda$) based on the sum of squared error (SSE) $\|Y - \hat Y(\lambda)\|^2$ using the `apply` method on `res` and `argmin`




```{code-cell} python

```

#### Visualizing SSE evolution with $\lambda$



Plot the logarithm of $\lambda$ values from the grid on the x-axis and the calculated SSE (previous question) on the y-axis




```{code-cell} python

```

#### Quick Modeling



1.  The previous questions can be combined quickly using `cross_val_predict` (manually calculate the grid):
2.  For RidgeCV, use the `'neg_mean_squared_error'` loss in the `scoring` argument and get almost the same
3.  Construct a score &ldquo;sum of quadratic errors per fold&rdquo; using `make_scorer` and use it in RidgeCV to get the result of the first question (manually calculate the grid)




```{code-cell} python

```
