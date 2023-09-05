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
  title: 'Lab Session on Inference'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa BEDIN<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

+++

## Python Modules



Import the modules: pandas (as `pd`), numpy (as `np`), matplotlib.pyplot (as `plt`), and statsmodels.formula.api (as `smf`).




```{code-cell} python

```

## Confidence Intervals (CI)



### Data Importation



Import the data from eucalyptus into the pandas DataFrame `eucalypt`. Use `read_csv` from `numpy`. In Fun Campus, the datasets are located in the directory `data/`.




```{code-cell} python

```

### Simple Regression



Perform a simple linear regression where `circ` is the explanatory variable and `ht` is the dependent variable. Store the result in the object `reg` [ using `ols` from `smf`, `fit` method of the `OLS` class ].




```{code-cell} python

```

### Coefficient CIs



Obtain the 95% confidence intervals for the coefficients [ using the `conf_int` method for the fitted instance/model ].




```{code-cell} python

```

### Prediction CIs



Create a grid of 100 new observations evenly spaced between the minimum and maximum of `circ`. Calculate a 95% CI for these 100 new observations $y^*$ (predict the values using the `get_prediction` method on the estimated instance/model and use the `conf_int` method on the prediction result).




```{code-cell} python

```

### Expectation CI



For the same grid of `circ` values as in the previous question, propose a 95% CI for the expectations $X^*\beta$.




```{code-cell} python

```

### Representation of CIs



Using the 100 predicted observations above and their CIs (both observations and expectations), represent on the same graph:

-   The observations
-   The CI for the predictions
-   The CI for the expectations.

[ Use `plt.plot` and `plt.legend` ]




```{code-cell} python

```

## Confidence Intervals for Two Coefficients



The objective of this lab is to plot the confidence region for parameters and observe the difference with two univariate CIs. For this lab, we will also need the following modules in addition to the standard ones.




```{code-cell} python
import math
from scipy.stats import f
```

### Data Import



Import the ozone data into the pandas DataFrame `ozone` [ `read_csv` from `numpy` ]. In Fun Campus, the datasets are located in the `data/` directory.




```{code-cell} python

```

### Model with 3 Variables



Estimate a regression model explaining the maximum ozone concentration of the day (variable `O3`) with

-   the temperature at noon denoted as `T12`
-   the east-west wind speed denoted as `Vx`
-   the noon cloudiness `Ne12`

along with the constant term as always.
[ Use `ols` from `smf`, `fit` method of the `OLS` class, and `summary` method for the fitted instance/model. ]




```{code-cell} python

```

### Confidence Region for All Variables



Let&rsquo;s focus on the first two variables `T12` and `Vx`, denoted here as $\beta_2$ and $\beta_3$ (the coefficient $\beta_1$ is for the constant/intercept variable).

Define
$F_{2:3}= \|\hat\beta_{2:3} - \beta_{2:3}\|^2_{\hat V_{\hat\beta_{2:3}}^{-1}}$
and introduce the following notation:
$\hat V_{\hat\beta_{2:3}}=\hat\sigma [(X'X)^{-1}]_{2:3,2:3} = \hat\sigma \Sigma$.
Also note that $\Sigma=U\Lambda U'$ and
$\Sigma^{1/2}=U\Delta^{1/2} U'$ ($U$ is an orthogonal matrix of the eigenvectors of $\Sigma$, and $\Delta$ is a diagonal matrix of positive or non-negative eigenvalues).

1.  Show that $F_{2:3,2:3}$ follows a Fisher distribution $\mathcal{F}(2,n-4)$. Calculate its 95% quantile using the `f` function from the `scipy.stats` sub-module (use the `isf` method).




```{code-cell} python

```

1.  Deduce that the confidence region for $\beta_{1:2}$ is the image of a
    disk by a matrix. Calculate this matrix in Python [ use the `cov_params` method for the `modele3` instance, functions `eigh` from the `np.linalg` sub-module, `np.matmul`, `np.diag`, `np.sqrt` ].




```{code-cell} python

```

1.  Generate 500 points on the circle [ `cos` and `sin` from `np` ]




```{code-cell} python

```

1.  Transform these points using the matrix to obtain the confidence ellipse.




```{code-cell} python

```

1.  Plot the ellipse [  `plt.fill` (for the ellipse), `plt.plot` (for the center) ]




```{code-cell} python

```

### Univariate CIs



Add the &ldquo;confidence rectangle&rdquo; from the 2 univariate CIs to the ellipse by obtaining the `Axes` using `plt.gca()`, creating the rectangle `patch` with `matplotlib.patches.Rectangle`, and adding it with `ax.add_artist`.




```{code-cell} python

```

## Confidence Intervals and Bootstrap



The goal of this lab is to construct a confidence interval using the Bootstrap.



### Data Import



Import the ozone data into the pandas DataFrame `ozone` [ `read_csv` from `numpy` ]. In Fun Campus, the datasets are located in the `data/` directory.




```{code-cell} python

```

### Model with 3 Variables



Estimate a regression model explaining the maximum ozone concentration of the day (variable `O3`) with

-   the temperature at noon denoted as `T12`
-   the east-west wind speed denoted as `Vx`
-   the noon cloudiness `Ne12`

along with the constant term as always.
[ Use `ols` from `smf`, `fit` method of the `OLS` class, and `summary` method for the fitted instance/model. ]



### Bootstrap and CI



#### Calculation of the Empirical Model: $\hat Y$ and $\hat\varepsilon$



Store the residuals in the object `residus` and the adjustments in `ychap`



#### Bootstrap Sample Generation



The regression model generating the $Y_i$ ($i\in\{1,\cdots,n\}$) is as follows:
$$
Y_i = \beta_1 +  \beta_2 X_{i2} +   \beta_3 X_{i3} +   \beta_4 X_{i4} +  \varepsilon_i
$$
where the distribution of $\varepsilon_i$ (denoted as $F$) is unknown.

For instance, if we had $B=1000$ samples, we could estimate $\beta$ $B$ times and observe the variability of $\hat\beta$ using these $B$ estimates to calculate empirical quantiles at levels $\alpha/2$ and $1-\alpha/2$, thereby obtaining a confidence interval.

Of course, we only have a single $n$-sample, and if we want to generate $B$ samples, we would need to know $\beta$ and $F$. The idea of the bootstrap is to replace the unknown $\beta$ and $F$ with $\hat\beta$ (the least squares estimator) and $\hat F$ (an estimator of $F$), generate $B$ samples, calculate the $B$ estimates $\hat\beta^*$, observe the variability of $\hat\beta^*$, and obtain empirical quantiles at levels $\alpha/2$ and $1-\alpha/2$ to form a confidence interval.

Let&rsquo;s generate $B=1000$ bootstrap samples.

1.  For each value of $b\in\{1,\cdots,B\}$, draw independently with replacement from the residuals of the regression $n$ values. Let $\hat\varepsilon^{(b)}$ be the resulting vector.
2.  Add these residuals to the adjustment $\hat Y$ to obtain a new sample $Y^*$. Using the data $X$ and $Y^*$, obtain the least squares estimation $\hat\beta^{(b)}$.
3.  Store the value $\hat\beta^{(b)}$ in row $b$ of the numpy array `COEFF`.
    [ Create an instance of the random number generator using `np.random.default_rng`, use the `randint` method on this instance; create a copy of the appropriate columns from `ozone` using the `copy` method to use `smf.ols`, and populate this DataFrame with the sample. ]




```{code-cell} python

```

### Bootstrap CI



From the $B=1000$ values $\hat\beta^{(b)}$, propose a 95% confidence interval using [ `np.quantile` ].




```{code-cell} python

```

## Eucalyptus Height Modeling



### Data Import



Import the eucalyptus data into the pandas DataFrame `eucalypt` using [ `read_csv` from `numpy` ]. In Fun Campus, the datasets are located in the `data/` directory.




```{code-cell} python

```

### Two Regressions



In previous labs, we performed various modeling tasks. For single-variable modeling, we chose the square root model (see Simple Regression lab). Later, we introduced multiple regression, and now we will compare these two models.

1.  Perform a simple linear regression where the square root of `circ` is the explanatory variable and `ht` is the dependent variable. Store the result in the object `regsqrt`.
2.  Perform a multiple linear regression where the square root of `circ` and `circ` itself are the explanatory variables, and `ht` is the dependent variable. Store the result in the object `reg`.
    [ Use `ols` from `smf`, `fit` method of the `OLS` class. ]




```{code-cell} python

```

### Comparison



1.  Compare these two models using a $T$ test [ use the `summary` method ].




```{code-cell} python

```

1.  Compare these two models using an $F$ test [ `stats.anova_lm` from the `statsmodels.api` submodule ].




```{code-cell} python

```

## Does Age Influence Leisure Time?



An investigation was conducted on 40 individuals to study the relationship between leisure time (estimated by the respondent as the number of hours per day available for oneself) and age. The results of this survey are contained in the file `temps_libre.csv` (in Fun Campus, datasets are located in the `data/` directory). We aim to determine if these two variables are related.

1.  What is the data type of the variables?




```{code-cell} python

```

1.  How is the most common relationship between these two variables calculated?




```{code-cell} python

```

1.  How do we test if age has an influence on leisure time using regression? Perform this test and draw a conclusion.




```{code-cell} python

```

1.  Represent the data and discuss the rationale behind the previous test.




```{code-cell} python

```

## Does Obesity Influence Blood Pressure?



An investigation was conducted on 102 individuals to study the relationship between obesity (estimated by the ratio of a person&rsquo;s weight to the ideal weight obtained from the New York Metropolitan Life Tables) and blood pressure in millimeters of mercury. The results of this survey are contained in the file `obesite.csv` (in Fun Campus, data is located in the `data/` directory). We aim to determine if these two variables are related.

1.  What is the data type of the variables?




```{code-cell} python

```

1.  How is the most common relationship between these two variables calculated?

2.  How do we test if obesity has an influence on blood pressure using regression? Perform this test and draw a conclusion.




```{code-cell} python

```
