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
  title: 'Solutions to Lab Session on Inference'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa Bedin<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

+++

## Python Modules



### Importing Python Modules



Import the modules: pandas (as `pd`), numpy (as `np`), matplotlib.pyplot (as `plt`), and statsmodels.formula.api (as `smf`).




```{code-cell} python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
```

## Confidence Intervals (CI)



#### Importing Data



Import the &rsquo;eucalyptus.txt&rsquo; data into the pandas DataFrame `eucalypt`.




```{code-cell} python
eucalypt = pd.read_csv("data/eucalyptus.txt", header=0, sep=";")
```

#### Simple Regression



Perform a simple linear regression where `circ` is the explanatory variable and `ht` is the dependent variable. Store the result in the object `reg`.




```{code-cell} python
reg = smf.ols("ht ~ 1 + circ", data=eucalypt).fit()
```

#### Confidence Intervals for Coefficients



Obtain the 95% confidence intervals for the coefficients using the `conf_int` method on the fitted model.




```{code-cell} python
reg.conf_int(alpha=0.05)
```

#### Prediction Intervals



##### Creating a Grid of New Observations



Create a grid of 100 new observations evenly spaced between the minimum and maximum values of `circ`. Calculate a 95% prediction interval for these new observations $y^*$ (predict the values using the `get_prediction` method on the fitted model and use the `conf_int` method on the prediction result).




```{code-cell} python
grille = pd.DataFrame({"circ": np.linspace(eucalypt["circ"].min(), eucalypt["circ"].max(), 100)})
calculprev = reg.get_prediction(grille)
ICobs = calculprev.conf_int(obs=True, alpha=0.05)
```

##### Confidence Intervals for Expected Values



For the same grid of `circ` values as in the previous question, propose a 95% confidence interval for the expected values $X^*\beta$.




```{code-cell} python
ICdte = calculprev.conf_int(obs=False, alpha=0.05)
```

#### Plotting Confidence Intervals



Using the 100 predicted observations and their confidence intervals (observations and expected values), plot on the same graph:

-   The observations
-   The confidence interval for predictions
-   The confidence interval for expected values.




```{code-cell} python
prev = calculprev.predicted_mean
plt.plot(eucalypt["circ"], eucalypt["ht"], 'o', color='xkcd:light grey')
plt.plot(grille['circ'], prev, 'k-', lw=2, label="E(Y)")
lesic, = plt.plot(grille['circ'], ICdte[:, 0], linestyle='--', color='xkcd:cerulean', label=r"$\mathbb{E}(Y)$")
plt.plot(grille['circ'], ICdte[:, 1], linestyle='--', color='xkcd:cerulean')
lesic2, = plt.plot(grille['circ'], ICobs[:, 0], linestyle='-.', color='xkcd:grass', label=r"$Y$")
plt.plot(grille['circ'], ICobs[:, 1], linestyle='-.', color='xkcd:grass')
plt.legend(handles=[lesic, lesic2], loc='upper left')
```

## Confidence Intervals for Two Coefficients



#### Data Import



Import the &rsquo;ozone.txt&rsquo; data into the pandas DataFrame `ozone`.




```{code-cell} python
ozone = pd.read_csv("data/ozone.txt", header=0, sep=";")
```

#### Model with 3 Variables



Estimate a regression model explaining the maximum ozone concentration of the day (variable `O3`) with

-   the temperature at noon denoted as `T12`
-   the east-west wind speed denoted as `Vx`
-   the noon cloudiness `Ne12`

along with the constant term as always.
[Use `ols` from `smf`, `fit` method of the `OLS` class, and `summary` method for the fitted instance/model.]




```{code-cell} python
modele3 = smf.ols("O3 ~ T12 + Vx + Ne12",data=ozone).fit()
```

#### Confidence Region for All Variables



Let&rsquo;s focus on the first two variables `T12` and `Vx`, denoted here as $\beta_2$ and $\beta_3$ (the coefficient $\beta_1$ is for the constant/intercept variable).

Define
$F_{2:3}= \|\hat\beta_{2:3} - \beta_{2:3}\|^2_{\hat V_{\hat\beta_{2:3}}^{-1}}$
and introduce the following notation:
$\hat V_{\hat\beta_{2:3}}=\hat\sigma [(X'X)^{-1}]_{2:3,2:3} = \hat\sigma \Sigma$.
Also note that $\Sigma=U\Lambda U'$ and
$\Sigma^{1/2}=U\Delta^{1/2} U'$ ($U$ is an orthogonal matrix of the eigenvectors of $\Sigma$, and $\Delta$ is a diagonal matrix of positive or non-negative eigenvalues).

1.  Show that $F_{2:3,2:3}$ follows a Fisher distribution $\mathcal{F}(2,n-4)$. Calculate its 95% quantile using the `f` function from the `scipy.stats` sub-module (use the `isf` method).




```{code-cell} python
f.isf(0.05, 2, modele3.nobs - 2)
```

1.  Deduce that the confidence region for $\beta_{1:2}$ is the image of a
    disk by a matrix. Calculate this matrix in Python [use the `cov_params` method for the `modele3` instance, functions `eigh` from the `np.linalg` sub-module, `np.matmul`, `np.diag`, `np.sqrt` ].




```{code-cell} python
hatSigma = modele3.cov_params().iloc[1:3,1:3]
   valpr,vectpr = np.linalg.eigh(hatSigma)
   hatSigmademi = np.matmul(vectpr, np.diag(np.sqrt(valpr)))
```

1.  Generate 500 points on the circle [=cos= and `sin` from =np=]




```{code-cell} python
theta = np.linspace(0, 2 * math.pi, 500)
   rho = (2 * f.isf(0.05, 2, modele3.nobs - 2))**0.5
   x = rho * np.cos(theta)
   y = rho * np.sin(theta)
   XX = np.array([x, y])
```

1.  Transform these points using the matrix to obtain the confidence ellipse.




```{code-cell} python
ZZ = np.add(np.matmul(hatSigmademi, XX).transpose(), np.array(modele3.params[1:3]))
```

1.  Plot the ellipse [ `plt.fill` (for the ellipse), `plt.plot` (for the center) ]




```{code-cell} python
plt.fill(ZZ[:, 0], ZZ[:, 1], facecolor='yellow', edgecolor='black', linewidth=1)
   plt.plot(modele3.params[1], modele3.params[2], "+")
```

#### Univariate CIs



Add the &ldquo;confidence rectangle&rdquo; from the 2 univariate CIs to the ellipse by obtaining the `Axes` using `plt.gca()`, creating the rectangle `patch` with `matplotlib.patches.Rectangle`, and adding it with `ax.add_artist`.




```{code-cell} python
ICparams = modele3.conf_int(alpha=0.025)
  from matplotlib.patches import Rectangle
  plt.fill(ZZ[:, 0], ZZ[:, 1], facecolor='yellow', edgecolor='black', linewidth=1)
  plt.plot(modele3.params[1], modele3.params[2], "+")
  ax = plt.gca()
  r = Rectangle(ICparams.iloc[1:3, 0],
                ICparams.diff(axis=1).iloc[1, 1],
                ICparams.diff(axis=1).iloc[2, 1],
                fill=False)
  ax.add_artist(r)
```

We can see that using 2 univariate confidence intervals (and thus assuming that the variables are independent) is not suitable, and there are points within the rectangular confidence region that are not within the confidence ellipse and vice versa.



## Confidence Intervals and Bootstrap



The goal of this lab is to construct a confidence interval using the Bootstrap.



#### Data Import



Import the ozone data into the pandas DataFrame `ozone` [=read<sub>csv</sub>= from `numpy=]. In FunStudio, the datasets are located in the =data/` directory.




```{code-cell} python
ozone = pd.read_csv("data/ozone.txt", header=0, sep=";")
```

#### Model with 3 Variables



Estimate a regression model explaining the maximum ozone concentration of the day (variable `O3`) with

-   the temperature at noon denoted as `T12`
-   the east-west wind speed denoted as `Vx`
-   the noon cloudiness `Ne12`

along with the constant term as always.
[Use `ols` from `smf`, `fit` method of the `OLS` class, and `summary` method for the fitted instance/model.]

    modele3 = smf.ols("O3 ~ T12 + Vx + Ne12",data=ozone).fit()



#### Bootstrap and CI



##### Calculation of the Empirical Model: $\hat Y$ and $\hat\varepsilon$



Store the residuals in the object `residus` and the adjustments in `ychap`




```{code-cell} python
ychap = modele3.fittedvalues
residus = modele3.resid
```

##### Bootstrap Sample Generation



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
    [Create an instance of the random number generator using `np.random.default_rng`, use the `randint` method on this instance; create a copy of the appropriate columns from `ozone` using the `copy` method to use `smf.ols`, and populate this DataFrame with the sample.]




```{code-cell} python
B =1000
COEFF = np.zeros((B, 4))
n = ozone.shape[0]
rng = np.random.default_rng(seed=1234)
ozoneetoile = ozone[["O3", "T12" , "Vx",  "Ne12"]].copy()
for  b in range(B):
    resetoile = residus[rng.integers(n, size=n)]
    O3etoile = np.add(ychap.values ,resetoile.values)
    ozoneetoile.loc[:,"O3"] = O3etoile
    regboot = smf.ols("O3 ~ 1+ T12 + Vx + Ne12", data=ozoneetoile).fit()
    COEFF[b] = regboot.params.values

COEFF.shape
```

#### Bootstrap CI



From the $B=1000$ values $\hat\beta^{(b)}$, propose a 95% confidence interval using [=np.quantile=].




```{code-cell} python
pd.DataFrame(np.quantile(COEFF, [0.025, 0.975], axis=0).T)
```

## Eucalyptus Height Modeling



#### Data Import



Import the eucalyptus data into the pandas DataFrame `eucalypt` using [=read<sub>csv</sub>= from `numpy=]. In FunStudio, the datasets are located in the =data/` directory.




```{code-cell} python
eucalypt = pd.read_csv("data/eucalyptus.txt", header=0, sep=";")
```

#### Two Regressions



In previous labs, we performed various modeling tasks. For single-variable modeling, we chose the square root model (see Simple Regression lab). Later, we introduced multiple regression, and now we will compare these two models.

1.  Perform a simple linear regression where the square root of `circ` is the explanatory variable and `ht` is the dependent variable. Store the result in the object `regsqrt`.
2.  Perform a multiple linear regression where the square root of `circ` and `circ` itself are the explanatory variables, and `ht` is the dependent variable. Store the result in the object `reg`.
    [Use `ols` from `smf`, `fit` method of the `OLS` class.]




```{code-cell} python
regsqrt = smf.ols('ht~I(np.sqrt(circ))', data=eucalypt).fit()
reg = smf.ols('ht~I(np.sqrt(circ)) + circ', data=eucalypt).fit()
```

#### Comparison



1.  Compare these two models using a $T$ test [use the `summary` method].




```{code-cell} python
reg.summary()
```

The row `circ` of the table provides the estimation of the coefficient
   $\hat\beta_3$, the estimated standard deviation of the coefficient, the value of the
   $t$ statistic for the test $\mathrm{H}_0: \beta_3=0$ against
   $\mathrm{H}_1: \beta_3\neq 0$, which is -8.336 in this case, and its nearly zero critical probability.
   Hence, we reject $\mathrm{H}_0$, and the model `reg` appears to be a better fit.

1.  Compare these two models using an $F$ test [=stats.anova<sub>lm</sub>= from the `statsmodels.api` submodule].




```{code-cell} python
import statsmodels.api as sm
   sm.stats.anova_lm(regsqrt,reg)
```

We obtain the same results as before (since $F=t^2$).



## Does Age Influence Leisure Time?



An investigation was conducted on 40 individuals to study the relationship between leisure time (estimated by the respondent as the number of hours per day available for oneself) and age. The results of this survey are contained in the file `temps_libre.csv` (in FunStudio, datasets are located in the `data/` directory). We aim to determine if these two variables are related.

1.  What is the data type of the variables?




```{code-cell} python
tpslibre = pd.read_csv("data/temps_libre.csv", header=0, sep=";")
   tpslibre.columns = [ "age", "tempslibre" ]
   tpslibre.describe()
```

Both variables are quantitatives. We do modify the &rsquo;.&rsquo; as it is bad understood by `smf`.

1.  How is the most common relationship between these two variables calculated?




```{code-cell} python
tpslibre.corr()
```

The measure of the relationship is the linear correlation (whose square is the R2), and here it is very weak, suggesting that there is no linear relationship between the two variables.

1.  How do we test if age has an influence on leisure time using regression? Perform this test and draw a conclusion.




```{code-cell} python
reg = smf.ols("tempslibre~1+age", data=tpslibre).fit()
   reg.summary()
```

The row `age` of the table provides the estimation of the coefficient
    $\hat\beta_2$, the estimated standard deviation of the coefficient, the value of the
    $t$ statistic for the test $\mathrm{H}_0: \beta_2=0$ against
    $\mathrm{H}_1: \beta_2\neq 0$, which is 0.285 in this case, and its critical probability
    which is 0.777. Therefore, we retain $\mathrm{H}_0$, and it seems that there is no linear relationship.

1.  Represent the data and discuss the rationale behind the previous test.




```{code-cell} python
plt.plot(tpslibre.age, tpslibre.tempslibre, "*")
```

Clearly, we observe two regimes: between 30 and 60 years, there is little leisure time, and before 30 or after 60, there is more leisure time. There is an influence of age, but it&rsquo;s not linear (more like piecewise constant). The previous test is inappropriate.



## Does Obesity Influence Blood Pressure?



An investigation was conducted on 102 individuals to study the relationship between obesity (estimated by the ratio of a person&rsquo;s weight to the ideal weight obtained from the &ldquo;New York Metropolitan Life Tables&rdquo;) and blood pressure in millimeters of mercury. The results of this survey are contained in the file `obesite.csv` (in FunStudio, data is located in the `data/` directory). We aim to determine if these two variables are related.

1.  What is the data type of the variables?




```{code-cell} python
obesite = pd.read_csv("data/obesite.csv", header=0, sep=";")
   obesite.describe()
```

Both variables are quantitatives.

1.  How is the most common relationship between these two variables calculated?




```{code-cell} python
obesite.corr()
```

Linear correlation seems moderate here.

1.  How do we test if obesity has an influence on blood pressure using regression? Perform this test and draw a conclusion.




```{code-cell} python
plt.plot(obesite.obesite, obesite.pression, "o")
```

Even though the points don&rsquo;t exactly lie on a straight line, we can still consider that this model might roughly fit. Let&rsquo;s perform simple regression and a nullity slope $t$ test.




```{code-cell} python
reg = smf.ols("pression~1+obesite", data=obesite).fit()
   reg.summary()
```

The row `obesity` of the table provides the estimation of the coefficient
   $\hat\beta_2$, the estimated standard deviation of the coefficient, the value of the
   $t$ statistic for the test $\mathrm{H}_0: \beta_2=0$ against
   $\mathrm{H}_1: \beta_2\neq 0$, which is 3.45 in this case, and its critical probability
   which is 0.001. Hence, we reject $\mathrm{H}_0$, and there seems to be a linear relationship (albeit roughly; this model won&rsquo;t predict blood pressure effectively).


