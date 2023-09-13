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
  title: 'Lab Session on Multivariate Regression'
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

```

# Multiple Regression on Ozone data (2 variables)

## Data import
Import ozone data into pandas `ozone` DataFrame
\[`read_csv` of `pandas`\]. On Fun Campus the path is `data/ozone.txt`.


```{code-cell} python

```

## 3D representation
We're interested in building an ozone forecasting model using 
multiple regression. This regression will explain
the maximum ozone concentration of the day (variable `O3`) by 
- the temperature at midday (`T12`)
- wind speed on the East-West axis, noted `Vx`.
Let's graph the data with `O3` on the z axis, 
`T12` on the x-axis and `Vx` on the y-axis.

[`figure` and its `add_subplot` method `scatter` method of the `Axes` class]


```{code-cell} python

```

## Forecasting model
Write the above model.



## Model estimation
Use OLS to estimate the parameters of the model described above and summarize them.
[`ols` from `smf`, `fit` method from the `OLS` class and 
`summary` method for the adjusted instance/model]


```{code-cell} python

```

# Multiple Regression (course model) for Ozone Data

## Data import
Import ozone data into pandas `ozone` DataFrame
\[`read_csv` of `pandas`\]. On Fun Campus the path is `data/ozone.txt`.


```{code-cell} python

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
[`ols` from `smf`, `fit` method from the `OLS` class and 
`summary` method for the adjusted instance/model]


```{code-cell} python

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
[`scale` attribute of the adjusted model]


```{code-cell} python

```

# Multiple regression on eucalyptus data

## Data import
Import ozone data into pandas `eucalypt` DataFrame
\[`read_csv` of `pandas`\]. On Fun Campus the path is `data/eucalyptus.txt`.


```{code-cell} python

```

## data representation
Represent point cloud
[`plot` of plt and `xlabel` and `ylabel` of `plt`]


```{code-cell} python

```

## Forecast model
Estimate (by OLS) the linear model explaining the height (`ht`) 
by the circumference variable (`circ`) and the square root of circumference.
circumference.  You can use
operations and functions in formulas
(see https://www.statsmodels.org/stable/example_formulas.html)


```{code-cell} python

```

## Graphical representation of the model
Graph the data, the forecast using the above model and the forecast using the
the forecast by the simple regression models seen in the "two models" exercise
in the simple regression tutorial.
[`argsort` instance method for DataFrame columns,`plot` and `xlabel` and `ylabel` from `plt`.]


```{code-cell} python

```
