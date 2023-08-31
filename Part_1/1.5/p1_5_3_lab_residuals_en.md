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
  title: 'Lab Session on Residuals'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa Bedin<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

+++

# Python modules
Import modules pandas (as `pd`) numpy (as `np`)
matplotlib.pyplot (as `plt`), statsmodels.formula.api (as `smf`)
and statsmodels.api (as `sm`)


```{code-cell} python

```

# Multiple Rgression (course model)

### Data import
Import ozone data into pandas `ozone` DataFrame
\[`read_csv` from `pandas`\]. Datasets in Fun Campus are located in `data/` directory.


```{code-cell} python

```

### Course model estimation
We are interested in building an ozone forecasting model using 
multiple regression. This regression will explain
the maximum ozone concentration of the day (variable `O3`) by 
- temperature at noon, noted `T12`
- cloud cover at midday, noted as `Ne12`
- the wind speed on the East-West axis, noted `Vx`.
Traditionally, we always introduce the constant (and do so here too).
Estimate the OLS model and summarize.

[Use the `ols` method from `smf`, the `fit` method from the `OLS` class and the 
method `summary` for the adjusted instance/model]


```{code-cell} python

```

### Residuals \$\varepsilon\$
Display residuals graph (`resid` attribute of estimated model)
(with \$\hat y\$  (`predict` method of estimated model) on the x-axis and \$\varepsilon\$ on the y-axis).

[plt `plot`]


```{code-cell} python

```

### Studentized Residuals
Display the graph of residuals studentized by cross-validation (with \$\hat y\$ on the x-axis and 
\$\varepsilon\$ on the ordinate). To do this, use the `get_influence` function/method 
which will return an object (let's call it `infl`) with a `resid_studentized_external` attribute containing the desired residues.


```{code-cell} python

```

### Leverage points
Represent \$h_{ii}\$ with `plt.stem` according to line number
[`np.arange`, DataFrame `shape` attribute, instance attribute 
`hat_matrix_diag` for `infl`]


```{code-cell} python

```

# R²
We are interested in building an ozone forecasting model using 
multiple regression. However, we're not sure a priori
which variables are useful. Let's build several models.

### Estimation of the course model
This regression will explain
the maximum ozone concentration of the day (variable `O3`) by 
- temperature at noon, noted `T12`
- cloud cover at midday, noted as `Ne12`
- the wind speed on the East-West axis, noted `Vx`.
Traditionally, we always introduce the constant (and do so here too).
Estimate the OLS model and summarize.

[Use the `ols` method from `smf`, the `fit` method from the `OLS` class and the 
method `summary` for the adjusted instance/model]


```{code-cell} python

```

### Estimation of another model
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

```

### Compare R2
Compare the R2 of the 3- and 5-variable models 
and explain why this was expected.


```{code-cell} python
.
```

# Partial residuals (to go further)
This exercise demonstrates the practical usefulness of the partial residuals discussed in TD.
The data are in the files `tprespartial.dta` and
`tpbisrespartiel.dta`, the aim of this exercise is to show that the analysis of partial
of partial residuals can improve modeling.

### Import data
You have one variable to explain \$Y\$
and four explanatory variables in the file `tprespartiel.dta`. Datasets in Fun Campus are located in `data/` directory.


```{code-cell} python

```

### Estimation
OLS estimation of model parameters \$Y_i= \beta_0 + \beta_1 X_{i,1}+ \cdots+
\beta_4 X_{i,4} + \varepsilon_i.\$
[`ols` from `smf`, method `fit` from class `OLS` and 
method `summary` for the adjusted instance/model]


```{code-cell} python

```

### Analyze partial residuals
What do you think of the results?
[In the `plot_ccpr_grid` sub-module of `sm.graphics`, partial residuals are called
called "Component-Component plus Residual"
(CCPR) in the statsmodels module...


```{code-cell} python

```

### Model improvement 
Replace $X_4$ by $X_5=X_4^2$ in the previous model. What do you think of
  the new model? You can compare this model with the one
  previous question.
[the `ols` method of the `smf` class, the `fit` method of the `OLS` class and the 
 instance attribute `rsquared`]
You can use the
operations and functions in formulas
(see https://www.statsmodels.org/stable/example_formulas.html)


```{code-cell} python

```

### Analyze partial residuals
Analyze the partial residuals of the new model and note that
that they appear to be correct.
[In the `plot_ccpr_grid` sub-module of `sm.graphics`, partial residuals are called
called "Component-Component plus Residual"
(CCPR) in the statsmodels module...


```{code-cell} python

```

Do the same for `tp2bisrespartiel`.


```{code-cell} python

```
