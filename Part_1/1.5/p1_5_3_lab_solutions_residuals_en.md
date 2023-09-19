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
  title: Solutions to Lab Session on Residuals
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa BEDIN<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

# Python modules
Import modules pandas (as `pd`) numpy (as `np`)
matplotlib.pyplot (as `plt`), statsmodels.formula.api (as `smf`)
and statsmodels.api (as `sm`)


```{code-cell} python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
```

# Multiple Rgression (course model)

## Data import
Import ozone data into pandas `ozone` DataFrame
\[`read_csv` from `pandas`\]


```{code-cell} python
ozone = pd.read_csv('data/ozone.txt', sep=';')
```

## Course model estimation
We are interested in building an ozone forecasting model using 
multiple regression. This regression will explain
the maximum ozone concentration of the day (variable `O3`) by 
- temperature at noon, noted `T12`
- cloud cover at midday, noted as `Ne12`
- the wind speed on the East-West axis, noted `Vx`.
Traditionally, we always introduce the constant (and do so here too).
Estimate the OLS model and summarize.

\[Use the `ols` method from `smf`, the `fit` method from the `OLS` class and the 
method `summary` for the adjusted instance/model\]


```{code-cell} python
reg = smf.ols('O3~T12+Ne12+Vx', data=ozone).fit()
reg.summary()
```

## Residuals \$\varepsilon\$
Display residuals graph (`resid` attribute of estimated model)
(with \$\hat y\$  (`predict` method of estimated model) on the x-axis and \$\varepsilon\$ on the y-axis).

\[plt `plot`\]


```{code-cell} python
plt.plot(reg.predict(), reg.resid, 'o')
```

No visible residue structuring. The thickness (standard deviation) of the points seems to be 
the same, but these residuals by construction do not have the same variance, 
so it's tricky to conclude on the \$mathrm{V}(\varepsilon_i)=\sigma^2\$ hypothesis.
What's more, the scale of the ordinates depends on the problem, so these residuals are not very practical.

## Studentized Residuals
Display the graph of residuals studentized by cross-validation (with \$\hat y\$ on the x-axis and 
\$\varepsilon\$ on the ordinate). To do this, use the `get_influence` function/method 
which will return an object (let's call it `infl`) with a `resid_studentized_external` attribute containing the desired residues.


```{code-cell} python
infl = reg.get_influence()
plt.plot(reg.predict(), infl.resid_studentized_external, 'o')
```

No visible residue structuring. The thickness (standard deviation) of the points seems to be 
the same, so the hypothesis $\mathrm{V}(\varepsilon_i)=\sigma^2\$ appears to be correct. No points outside [-2,2], so no outliers.

## Leverage points
Represent \$h_{ii}\$ with `plt.stem` according to line number
\[`np.arange`, DataFrame `shape` attribute, instance attribute 
`hat_matrix_diag` for `infl`\]


```{code-cell} python
ozone.head()
```


```{code-cell} python
n_data = ozone.shape[0]
plt.stem(np.arange(n_data), infl.hat_matrix_diag)
```

No \$h_{ii}\$ significantly larger 
than the others, so the experimental design is correct.

# R²
We are interested in building an ozone forecasting model using 
multiple regression. However, we're not sure a priori
which variables are useful. Let's build several models.

## Estimation of the course model
This regression will explain
the maximum ozone concentration of the day (variable `O3`) by 
- temperature at noon, noted `T12`
- cloud cover at midday, noted as `Ne12`
- the wind speed on the East-West axis, noted `Vx`.
Traditionally, we always introduce the constant (and do so here too).
Estimate the OLS model and summarize.

\[Use the `ols` method from `smf`, the `fit` method from the `OLS` class and the 
method `summary` for the adjusted instance/model\]


```{code-cell} python
reg = smf.ols('O3~T12+Ne12+Vx', data=ozone).fit()
reg.summary()
```

## Estimation of another model
This regression will explain
the maximum ozone concentration of the day (variable `O3`) by 
- temperature at noon, noted `T12`
- the temperature at 3pm, `T15`
- cloudiness at noon, noted as `Ne12`
- wind speed on east-west axis noted `Vx`
- the maximum of the previous day `O3v`.
Traditionally, we always introduce the constant (do so here too).
Estimate the OLS model and summarize.


```{code-cell} python
reg5 = smf.ols('O3~T12+T15+Ne12+Vx+O3v', data=ozone).fit()
reg5.summary()
```

## Compare R2
Compare the R2 of the 3- and 5-variable models 
and explain why this was expected.


```{code-cell} python
reg.rsquared, reg5.rsquared
```

The R2 increases with the number of variables added. Thus, R2 score cannot be used to compare fits for models with different numbers of variables.

# Partial residuals (to go further)
This exercise demonstrates the practical usefulness of the partial residuals discussed in TD.
The data are in the files `tprespartial.dta` and
`tpbisrespartiel.dta`, the aim of this exercise is to show that the analysis of partial
of partial residuals can improve modeling.

## Import data
You have one variable to explain \$Y\$
and four explanatory variables in the file `tprespartiel.dta`.


```{code-cell} python
tpres = pd.read_csv('data/tprespartiel.dta', sep=';')
tpres.head()
```

## Estimation
OLS estimation of model parameters \$Y_i= \beta_0 + \beta_1 X_{i,1}+ \cdots+
\beta_4 X_{i,4} + \varepsilon_i.\$
\[`ols` from `smf`, method `fit` from class `OLS` and 
method `summary` for the adjusted instance/model\]


```{code-cell} python
reg = smf.ols('Y~X1+X2+X3+X4', data=tpres).fit()
reg.summary()
```

## Analyze partial residuals
What do you think of the results?
\[In the `plot_ccpr_grid` sub-module of `sm.graphics`, partial residuals are called
called "Component-Component plus Residual"
(CCPR) in the statsmodels module...\]


```{code-cell} python
sm.graphics.plot_ccpr_grid(reg)
plt.show()
```

Clearly, the graph for the variable `X4` does not show
points arranged along a straight line or a structureless cloud. 
It shows a \$x\mapsto x^2\$ type of structuring.

## Model improvement 
Replace $X_4$ by $X_5=X_4^2$ in the previous model. What do you think of
  the new model? You can compare this model with the one
  previous question.
\[the `ols` method of the `smf` class, the `fit` method of the `OLS` class and the 
 instance attribute `rsquared`\]
You can use the
operations and functions in formulas
(see https://www.statsmodels.org/stable/example_formulas.html)


```{code-cell} python
reg2 = smf.ols('Y~X1+X2+X3+np.square(X4)', data=tpres).fit()
reg2.summary()
```

## Analyze partial residuals
Analyze the partial residuals of the new model and note that
that they appear to be correct.
\[In the `plot_ccpr_grid` sub-module of `sm.graphics`, partial residuals are called
called "Component-Component plus Residual"
(CCPR) in the statsmodels module...\]


```{code-cell} python
sm.graphics.plot_ccpr_grid(reg2)
plt.show()
```

The graphs show points with no obvious structure
or arranged along straight lines. The model would appear to be correct. We can compare 
compare them (same number of variables) by R2


```{code-cell} python
reg.rsquared, reg2.rsquared
```

## Do the same for `tp2bisrespartiel`.


```{code-cell} python
tpbis = pd.read_csv('data/tpbisrespartiel.dta', sep=';')
tpbis.head()
```


```{code-cell} python
reg = smf.ols('Y~X1+X2+X3+X4', data=tpbis).fit()
reg.summary()
```


```{code-cell} python
sm.graphics.plot_ccpr_grid(reg)
plt.show()
```


```{code-cell} python
reg2 = smf.ols('Y~X1+X2+X3+np.sin(2*np.pi*X4)', data=tpbis).fit()
reg2.summary()
```


```{code-cell} python
sm.graphics.plot_ccpr_grid(reg2)
plt.show()
```
