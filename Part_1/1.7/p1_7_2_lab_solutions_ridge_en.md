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
  title: 'Solutions to Lab session on Ridge Regression'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa BEDIN<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

## Modules



### Importing Modules



Import the modules pandas (as `pd`) and numpy (as `np`)
Import the sub-module `pyplot` from `matplotlib` as `plt`
Import the functions `StandardScaler` from `sklearn.preprocessing`
Import the function `Ridge` from `sklearn.linear_model`
Import the function `RidgeCV` from `sklearn.linear_model`
Import the function `Pipeline` from `sklearn.pipeline`
Import the function `cross_val_predict` from `sklearn.model_selection`
Import the function `KFold` from `sklearn.model_selection`
Import the function `make_scorer` from `sklearn.metrics`




```{code-cell} python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
```

## Ridge Regression on Ozone Data



### Importing Data



Import the ozone data `ozonecomplet.csv` and remove the last two qualitative variables
Summarize each variable using methods `astype` on DataFrame columns and `describe` on DataFrame instance




```{code-cell} python
ozone = pd.read_csv("data/ozonecomplet.csv", header=0, sep=";")
ozone = ozone.drop(['nomligne', 'Ne', 'Dv'], axis=1)
ozone.describe()
```

### Creating numpy Arrays



Create numpy arrays `y` and `X` using instance methods `iloc` or `loc` (using the underlying `values` attribute of DataFrame)




```{code-cell} python
y = ozone.O3.values
X = ozone.iloc[:,1:].values
```

### Centering and Scaling



Center and scale variables using `StandardScaler` with the following steps:

1.  Create an instance with the `StandardScaler` function and name it `scalerX`
2.  Fit the instance using the `fit` method with the numpy array `X`
3.  Transform the array `X` to a centered and scaled array using the `transform` method




```{code-cell} python
scalerX = StandardScaler().fit(X)
Xcr= scalerX.transform(X)
```

### Ridge Regression for $\lambda=0.00485$



1.  Estimation/adjustment: Use centered and scaled data for `X` and vector `y` to estimate the Ridge regression model:
    -   Instantiate a `Ridge` model using the same name




```{code-cell} python
ridge = Ridge(alpha=0.00485)
```

We recall that $\lambda$ parameter of ridge regression is called
     $\alpha$ in scikit-learn.

-   Estimate the model with $\lambda=0.00485$ using the `fit` instance method




```{code-cell} python
ridge.fit(Xcr, y)
```

1.  Display $\hat\beta(\lambda)$ coefficients




```{code-cell} python
print(ridge.coef_)
```

1.  Predict a value for $x^*=(17, 18.4, 5, 5, 7, -4.3301, -4, -3, 87)'$




```{code-cell} python
print(ridge.predict(Xcr[1,:].reshape(1, 10)))
```

### Using Pipeline



-   Verify that `scalerX.transform(X[1,:].reshape(1, 10))` gives `Xcr[1,:]`. However, the sequence of &ldquo;X transformation&rdquo; followed by &ldquo;modeling&rdquo; can be automated using a  [Pipeline](https://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html)




```{code-cell} python
np.all(np.abs( scalerX.transform(X[1,:].reshape(1, 10))[0,:] - Xcr[1,:])<1e-10)
```

-   Create a `Pipeline` instance
    1.  Create an instance for StandardScaler




```{code-cell} python
cr = StandardScaler()
```

1.  Create an instance for Ridge




```{code-cell} python
ridge = Ridge(alpha=0.00485)
```

1.  Create a `Pipeline` instance with the `steps` argument, where each step is a tuple with the step&rsquo;s name (e.g., `"cr"` or `"ridge"`) and the instance of the step (created in the previous steps)




```{code-cell} python
pipe = Pipeline(steps=[("cr", cr) , ("ridge",  ridge)])
```

-   Fit the pipeline instance with the `fit` instance method using the data `X` and `y`




```{code-cell} python
pipe.fit(X,y)
```

-   Retrieve $\hat\beta(\lambda)$ by accessing the `"ridge"` coordinate (chosen step&rsquo;s name) from the `named_steps` attribute of the pipeline object




```{code-cell} python
er=pipe.named_steps["ridge"]
print(er.coef_)
```

-   Predict the adjustment for $x^*$




```{code-cell} python
print(pipe.predict(X[1,:].reshape(1,10)))
```

### Coefficient Evolution with $\lambda$



#### Calculating a $\lambda$ Grid



The classic grid for ridge is constructed similarly to the one for Lasso:

1.  Calculate the maximum value $\lambda_0 = \arg\max_{i} |[X'y]_i|/n$
2.  Create a grid using powers of 10, with exponents ranging from 0 to -4 (usually 100 evenly spaced values)
3.  Multiply the grid by $\lambda_0$
4.  For Ridge regression, the grid is often multiplied by $100$ or $1000$




```{code-cell} python
llc = np.linspace(0, -4, 100)
l0 = np.abs(Xcr.transpose().dot(y)).max()/X.shape[0]
alphas_ridge = l0*100*10**(llc)
```

#### Plotting the Evolution of $\hat\beta(\lambda)$



First coefficients list




```{code-cell} python
lcoef = []
for ll in alphas_ridge:
    pipe = Pipeline(steps=[("cr", StandardScaler()) , ("ridge",  Ridge(alpha=ll))]).fit(X,y)
    er = pipe.named_steps["ridge"]
    lcoef.append(er.coef_)
```

or without pipeline




```{code-cell} python
lcoef = []
for ll in alphas_ridge:
    rr = Ridge(alpha=ll).fit(Xcr,y)
    lcoef.append(rr.coef_)
```

Plotting




```{code-cell} python
plt.plot(np.log(alphas_ridge), lcoef)
plt.show()
```

As $\lambda$ increase the values of  $\hat\beta(\lambda)$ are shrunk (toward the origin)



### Optimal $\hat\lambda$ (10-fold Cross Validation)



#### Separating into 10 Blocks



Split the dataset into 10 blocks using the [KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold) function: create an instance of `KFold` named `kf`.




```{code-cell} python
kf = KFold(n_splits = 10, shuffle=True, random_state=0)
```

#### Selecting the Optimal $\hat\lambda$



1.  Create a DataFrame `res` with 100 columns filled with zeros
2.  Loop through all blocks; use the `split` instance method on `kf` with data `X`
3.  For each block:
    1.  Estimate Ridge models for each $\lambda$ on the training data (9 blocks)
    2.  Predict the validation data for each $\lambda$
    3.  Store the predicted values in the corresponding rows of `res` for the 100 Ridge models




```{code-cell} python
kf = KFold(n_splits=10, shuffle=True, random_state=0)
res = pd.DataFrame(np.zeros((X.shape[0], len(alphas_ridge))))
for app_index, val_index in kf.split(X):
    Xapp = Xcr[app_index,:]
    yapp = y[app_index]
    Xval = Xcr[val_index,:]
    yval = y[val_index]
    for j, ll in enumerate(alphas_ridge):
        rr = Ridge(alpha=ll).fit(Xapp, yapp)
        res.iloc[val_index,j] = rr.predict(Xval)
```

#### Optimal $\hat\lambda$ Selection



Using the squared error $\|Y - \hat Y(\lambda)\|^2$, determine the best model (and therefore the optimal $\hat\lambda$) using the `apply` instance method on `res` and `argmin`.




```{code-cell} python
sse = res.apply(lambda x: ((x-y)**2).sum(), axis=0)
print(alphas_ridge[sse.argmin()])
```

#### Graphical Representation



Plot the logarithms of the values of $\lambda$ on the grid against the calculated squared errors from the previous step.




```{code-cell} python
plt.plot(np.log(alphas_ridge), sse, "-")
```

### Quick Modeling



The previous questions can be chained more quickly using `cross_val_predict`

1.  Chain the previous questions using `cross_val_predict` (calculate the grid manually)
    Build the grid




```{code-cell} python
scalerX = StandardScaler().fit(X)
Xcr= scalerX.transform(X)
llc = np.linspace(0, -4, 100)
l0 = np.abs(Xcr.transpose().dot(y)).max()/X.shape[0]
alphas_ridge = l0*100*10**(llc)
```

Cross Validation 10 fold




```{code-cell} python
kf = KFold(n_splits=10, shuffle=True, random_state=0)
resbis = pd.DataFrame(np.zeros((X.shape[0], len(alphas_ridge))))
for j, ll in enumerate(alphas_ridge):
    resbis.iloc[:,j] = cross_val_predict(Ridge(alpha=ll),Xcr,y,cv=kf)
```

and results as in previous questions

1.  Build the grid




```{code-cell} python
scalerX = StandardScaler().fit(X)
Xcr= scalerX.transform(X)
llc = np.linspace(0, -4, 100)
l0 = np.abs(Xcr.transpose().dot(y)).max()/X.shape[0]
alphas_ridge = l0*100*10**(llc)
```

We use `RidgeCV` with `kf` (as always)




```{code-cell} python
kf = KFold(n_splits=10, shuffle=True, random_state=0)
modele_ridge = RidgeCV(alphas=alphas_ridge, cv=kf, scoring = 'neg_mean_squared_error').fit(Xcr, y)
```

The result is a ridge model already fitted with $\hat\lambda$




```{code-cell} python
print(modele_ridge.alpha_)
```

1.  If we prefer the sum of squared error (SSE) we need to
    construct a loss function and a score object:




```{code-cell} python
def my_custom_loss_func(y_true, y_pred):
    sse = np.sum((y_true - y_pred)**2)
    return sse
myscore = make_scorer(my_custom_loss_func, greater_is_better=False)
```

We can use this score using:




```{code-cell} python
kf = KFold(n_splits=10, shuffle=True, random_state=0)
modele_ridge = RidgeCV(alphas=alphas_ridge, cv=kf, scoring = myscore).fit(Xcr, y)
```

And the result is a ridge model already fitted and $\hat\lambda$ is




```{code-cell} python
print(modele_ridge.alpha_)
```
