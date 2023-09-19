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
  title: Solutions to Lab Session on Multivariate Regression
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa BEDIN<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

# Python Modules
Import modules pandas (as `pd`), numpy (as `np`), matplotlib.pyplot (as `plt`) and statsmodels.formula.api (as `smf`). 
Also import `Axes3D` from `mpl_toolkits.mplot3d`.


```{code-cell} python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.formula.api as smf
```

# Multiple Regression on Ozone data (2 variables)

## Data import
Import ozone data into pandas `ozone` DataFrame.


```{code-cell} python
ozone = pd.read_csv("data/ozone.txt", header=0, sep=";")
```

## 3D representation
We're interested in building an ozone forecasting model using 
multiple regression. This regression will explain
the maximum ozone concentration of the day (variable `O3`) by 
- the temperature at midday (`T12`)
- wind speed on the East-West axis, noted `Vx`.
Let's graph the data with `O3` on the z axis, 
`T12` on the x-axis and `Vx` on the y-axis.


```{code-cell} python
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax = Axes3D(fig)
ax.scatter(ozone["T12"], ozone["Vx"],ozone["O3"])
ax.set_xlabel('T12')
ax.set_ylabel('Vx')
ax.set_zlabel('O3')
```

## Forecasting model
Write the above model.

\$y_i = \beta_1 + \beta_2 X_i + \beta_3 Z_i + \varepsilon_i\$
where 
- \$X_i\$ is the \$i^e\$ observation of the explanatory variable `T12` and
- \$Z_i\$ is the \$i^e\$ observation of the explanatory variable `Vx`.
- \$X_i\$ is the \$i^e\$ observation for the explanatory variable `O3`.
- \$varepsilon_i\$ is the \$i^e\$ coordinate of the error vector
  \$varepsilon\$
Traditionally, as is the case here, we always introduce the constant 
(variable associated with \$\beta_1\$).

## Model estimation
Use OLS to estimate the parameters of the model described above and summarize them.


```{code-cell} python
reg = smf.ols('O3~T12+Vx', data=ozone).fit()
reg.summary()
```

# Multiple Regression (course model) for Ozone Data

## Data import
Import ozone data into pandas `ozone` DataFrame


```{code-cell} python
ozone = pd.read_csv("data/ozone.txt", header=0, sep=";")
```

## Course model estimation
We are interested in building an ozone forecasting model using 
multiple regression. This regression will explain
the maximum ozone concentration of the day (variable `O3`) by 
- temperature at noon, noted `T12
- cloud cover at midday, noted as `Ne12
- the wind speed on the East-West axis, noted `Vx`.
Traditionally, we always introduce the constant (and do so here too).
Estimate the OLS model and summarize.


```{code-cell} python
reg = smf.ols('O3~T12+Ne12+Vx', data=ozone).fit()
reg.summary()
```

## Variability 
- Among the estimators of the coefficients of the effects of the variables
(excluding the constant) that is the most variable.
- Variability is indicated by
  - parameter variance
  - the standard deviation of the parameter
  - the estimated variance of the parameter
  - the parameter's estimated standard deviation
- Display estimate for \$\sigma^2\$

Reading the summary, the `std err` column gives the estimated
estimated standard deviations of the coordinates of \$\hat \beta\$ and the largest 
is that associated with the `Ne12` variable.


```{code-cell} python
reg.scale
```

# Multiple regression on eucalyptus data

## Importing data
Import eucalytus data into pandas `eucalypt` DataFrame


```{code-cell} python
eucalypt = pd.read_csv("data/eucalyptus.txt", header=0, sep=";")
```

## data representation
Represent point cloud


```{code-cell} python
plt.plot(eucalypt["circ"],eucalypt["ht"],'o')
plt.xlabel("circ")
plt.ylabel("ht")
```

## Forecast model
Estimate (by OLS) the linear model explaining the height (`ht`) 
by the circumference variable (`circ`) and the square root of circumference.
circumference.  You can use
operations and functions in formulas
(see https://www.statsmodels.org/stable/example_formulas.html)


```{code-cell} python
regmult = smf.ols("ht ~ circ +  np.sqrt(circ)", data = eucalypt).fit()
regmult.summary()
```

## Graphical representation of the model
Graph the data, the forecast using the above model and the forecast using the
the forecast by the simple regression models seen in the "two models" exercise
in the simple regression tutorial.


```{code-cell} python
reg = smf.ols('ht~circ', data=eucalypt).fit()
regsqrt = smf.ols('ht~I(np.sqrt(circ))', data=eucalypt).fit()
```


```{code-cell} python
sel = eucalypt['circ'].argsort()
xs = eucalypt.circ.iloc[sel]
ys1 = regmult.predict()[sel]
ys2 = reg.predict()[sel]
ys3 = regsqrt.predict()[sel]
plt.plot(eucalypt['circ'], eucalypt['ht'], "o", xs, ys1, "-", xs, ys2, "--", xs, ys3, "-.")
```
