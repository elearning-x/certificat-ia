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
  title: 'Lab session on Lasso and Elastic-Net'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa Bedin<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

+++

## Modules



### Importing Modules



-   Import the modules pandas (as `pd`) and numpy (as `np`)
-   Import the sub-module `pyplot` from `matplotlib` as `plt`
-   Import the function `StandardScaler` from `sklearn.preprocessing`
-   Import the function `Lasso` from `sklearn.linear_model`
-   Import the function `LassoCV` from `sklearn.linear_model`
-   Import the function `ElasticNet` from `sklearn.linear_model`
-   Import the function `ElasticNetCV` from `sklearn.linear_model`
-   Import the function `cross_val_predict` from `sklearn.model_selection`
-   Import the function `KFold` from `sklearn.model_selection`



```{code-cell} python

```

## Lasso Regression on Ozone Data



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

1.  Create an instance using `StandardScaler` and name it `scalerX`
2.  Fit the instance using `fit` method with the numpy array `X`
3.  Transform the array `X` to a centered and scaled array using `transform` method




```{code-cell} python

```

### Coefficient Evolution with $\lambda$



The function `LassoCV` directly provides the $\lambda$ grid (unlike Ridge). Use this function on centered and scaled data to retrieve the grid (use the `alphas_` attribute). Then, iterate through the grid to estimate coefficients $\hat\beta(\lambda)$ for each $\lambda$ value.




```{code-cell} python

```

### Optimal $\hat\lambda$ Selection (10-fold Cross Validation)



#### Splitting into 10 Blocks



Split the dataset into 10 blocks using the [KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold) function (named `kf`).




```{code-cell} python

```

#### Optimal $\hat\lambda$ Selection



Find the optimal $\hat\lambda$ by computing the &ldquo;sum of quadratic errors per block&rdquo; score using `cross_val_predict` (provide the grid to `Lasso`).




```{code-cell} python

```

#### Retrieving Results from Previous Step



Use the `LassoCV` function and the `kf` object to retrieve the optimal $\hat\lambda$ (10-fold cross validation).




```{code-cell} python

```

#### Prediction



Use Ridge regression with the optimal $\hat\lambda$ to predict ozone concentration for $x^*=(18, 18, 18, 5, 5, 6, 5, -4, -3, 90)'$.




```{code-cell} python

```

## Elastic-Net Regression



Repeat the same questions from the previous exercise using the same data with a balance between L1 and L2 norms of 1/2 (`l1_ratio`).




```{code-cell} python

```
