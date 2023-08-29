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
  title: 'Solutions to Lab session on Lasso and Elastic-Net'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa Bedin<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

## Modules



-   Import the pandas module (as `pd`)
-   Import the numpy module (as `np`)
-   Import the `pyplot` submodule from `matplotlib` (as `plt`)
-   Import the `StandardScaler` function from `sklearn.preprocessing`
-   Import the `Lasso` function from `sklearn.linear_model`
-   Import the `LassoCV` function from `sklearn.linear_model`
-   Import the `ElasticNet` function from `sklearn.linear_model`
-   Import the `ElasticNetCV` function from `sklearn.linear_model`
-   Import the `cross_val_predict` function from `sklearn.model_selection`
-   Import the `KFold` class from `sklearn.model_selection`



```{code-cell} python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
```

## Lasso Regression on Ozone Data



### Data Import



-   Import the ozone data `ozonecomplet.csv`
-   Drop the last two qualitative variables and provide a summary of each variable using `astype` on the DataFrame column and the `describe` method on the DataFrame instance




```{code-cell} python
ozone = pd.read_csv("data/ozonecomplet.csv", header=0, sep=";")
ozone = ozone.drop(['nomligne', 'Ne', 'Dv'], axis=1)
ozone.describe()
```

### Creating numpy Arrays



Create numpy arrays `y` and `X` using the `iloc` or `loc` instance methods from the DataFrame, utilizing the underlying numpy array accessed through the `values` attribute




```{code-cell} python
y = ozone.O3.values
X = ozone.iloc[:,1:].values
```

### Standardization and Scaling



Standardize and scale the variables using the `StandardScaler` function following these steps:

1.  Create an instance using `StandardScaler`
2.  Fit the instance using the `fit` method with the numpy array `X`
3.  Transform the numpy array `X` to the standardized and scaled array using the `transform` method




```{code-cell} python
scalerX = StandardScaler().fit(X)
Xcr = scalerX.transform(X)
```

### Coefficient Evolution with $\lambda$



The `LassoCV` function directly provides the $\lambda$ grid (unlike ridge). Use this function on the standardized and scaled data to retrieve the grid (using the `alphas_` attribute). Then loop through the grid to estimate the coefficients $\hat\beta(\lambda)$ for each $\lambda$ value.




```{code-cell} python
rl = LassoCV().fit(Xcr, y)
alphas_lasso = rl.alphas_
lcoef = []
for ll in alphas_lasso:
    rl = Lasso(alpha=ll).fit(Xcr, y)
    lcoef.append(rl.coef_)
```

Plot the coefficients against the logarithm of the $\lambda$ values




```{code-cell} python
plt.plot(np.log(alphas_lasso), lcoef)
plt.show()
```

### Optimal $\hat \lambda$ Selection (10-fold Cross-Validation)



-   Use the `KFold` function to split the data into 10 blocks (instance named `kf`)
-   Find the optimal $\hat \lambda$ using the &ldquo;sum of squared errors per block&rdquo; score with `cross_val_predict` (supply the grid to `Lasso`)




```{code-cell} python
kf = KFold(n_splits=10, shuffle=True, random_state=0)
res = pd.DataFrame(np.zeros((X.shape[0], len(alphas_lasso))))
for j, ll in enumerate(alphas_lasso):
    res.iloc[:,j] = cross_val_predict(Lasso(alpha=ll), Xcr, y, cv=kf)
sse = res.apply(lambda x: ((x-y)**2).sum(), axis=0)
print(alphas_lasso[sse.argmin()])
```

    0.7727174033372736

#### Retrieve the results of the previous question



Using `LassoCV` and the `kf` instance to find the optimal $\hat \lambda$ (10-fold Cross-Validation)




```{code-cell} python
rl = LassoCV(cv=kf).fit(Xcr, y)
print(rl.alpha_)
```

    0.7727174033372736

Here, the objective function is $\mathrm{R}^2$ per block (not the sum of squared errors), and the same $\hat \lambda$ is obtained (though not guaranteed in all cases)



### Prediction



Use the optimal $\hat \lambda$ from Lasso regression to predict the ozone concentration for $x^*=(18, 18, 18, 5, 5, 6, 5, -4, -3, 90)'$




```{code-cell} python
xet = np.array([[18, 18, 18, 5, 5, 6, 5, -4, -3, 90]])
xetcr = scalerX.transform(xet)
print(rl.predict(xetcr))
```

    [85.28390512]

## Elastic-Net Regression



### Data Import



-   Import the ozone data `ozonecomplet.csv`
-   Drop the last two qualitative variables and provide a summary of each variable using `astype` on the DataFrame column and the `describe` method on the DataFrame instance




```{code-cell} python
ozone = pd.read_csv("data/ozonecomplet.csv", header=0, sep=";")
ozone = ozone.drop(['nomligne', 'Ne', 'Dv'], axis=1)
ozone.describe()
```

### Creating numpy Arrays



Create numpy arrays `y` and `X` using the `iloc` or `loc` instance methods from the DataFrame, utilizing the underlying numpy array accessed through the `values` attribute




```{code-cell} python
y = ozone.O3.values
X = ozone.iloc[:,1:].values
```

### Standardization and Scaling



Standardize and scale the variables using the `StandardScaler` function following these steps:

1.  Create an instance using `StandardScaler`
2.  Fit the instance using the `fit` method with the numpy array `X`
3.  Transform the numpy array `X` to the standardized and scaled array using the `transform` method




```{code-cell} python
scalerX = StandardScaler().fit(X)
Xcr = scalerX.transform(X)
```

### Coefficient Evolution with $\lambda$



-   Ajust the model for each value of $\lambda$ using the `ElasticNetCV` function
-   Plot the coefficients against the logarithm of the $\lambda$ values




```{code-cell} python
ren = ElasticNetCV().fit(Xcr, y)
alphas_elasticnet = ren.alphas_
lcoef = []
for ll in alphas_elasticnet:
    ren = ElasticNet(alpha=ll).fit(Xcr, y)
    lcoef.append(ren.coef_)
```

Plotting the results




```{code-cell} python
plt.plot(np.log(alphas_elasticnet), lcoef)
plt.show()
```

It can be seen that coefficients (in absolute value) are shrunk to zero as $\lambda$ increases.



### Optimal $\hat \lambda$ Selection (10-fold Cross-Validation)



-   Use the `KFold` function to split the data into 10 blocks (instance named `kf`)
-   Find the optimal $\hat \lambda$ using the &ldquo;sum of squared errors per block&rdquo; score with `cross_val_predict` (supply the grid to `ElasticNet`)




```{code-cell} python
kf = KFold(n_splits=10, shuffle=True, random_state=0)
res = pd.DataFrame(np.zeros((X.shape[0], len(alphas_elasticnet))))
for j, ll in enumerate(alphas_elasticnet):
    res.iloc[:,j] = cross_val_predict(ElasticNet(alpha=ll), Xcr, y, cv=kf)
sse = res.apply(lambda x: ((x-y)**2).sum(), axis=0)
print(alphas_elasticnet[sse.argmin()])
```

    0.41048105093488396

#### Retrieve the results of previous question



Retrieve the results using `ElasticNetCV` and the `kf` instance to find the optimal $\hat \lambda$ (10-fold Cross-Validation)




```{code-cell} python
ren = ElasticNetCV(cv=kf).fit(Xcr, y)
print(ren.alpha_)
```

    0.41048105093488396

### Prediction



Use the optimal $\hat \lambda$ from Elastic-Net regression to predict the ozone concentration for $x^*=(18, 18, 18, 5, 5, 6, 5, -4, -3, 90)'$




```{code-cell} python
xet = np.array([[18, 18, 18, 5, 5, 6, 5, -4, -3, 90]])
xetcr = scalerX.transform(xet)
print(ren.predict(xetcr))
```

    [87.15292087]

The prediction differs (from the lasso) due to the use of a different model.


