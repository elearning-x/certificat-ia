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
  title: 'Lab Session on Univariate Regression'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa BEDIN<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

# Modules python
Load pandas (as pd) numpy (as np) matplotlib.pyplot (as plt) et statsmodels.formula.api (as smf).

# Simple Regression

## Data Loading
Load eucalyptus data into a pandas DataFrame `eucalypt`
\[`read_csv` of `pandas`\]. On Fun Campus the path is `data/eucalyptus.txt`.


```{code-cell} python

```

## Point Cloud
Plot the point cloud with `circ` as abscissa and `ht` as ordinate
\[`plt.plot`\]


```{code-cell} python

```

## Simple Regression
Perform a simple linear regression where `circ` is the explanatory variable
and `ht` is the variable to be explained. Store the result
in the `reg` object and 
1. summarize this modeling;
2. display the attribute containing the OLS-estimated parameters of the line;
3. display the attribute containing the estimated standard deviation of the error.

\[`ols` of `smf`, method `fit` of `OLS` class, 
method `summary` applied to the adjusted model,
attributes `params` and `scale`.\]


```{code-cell} python

```

## Residuals
Plot the residuals with
1. the `circ` variable on the x-axis and the residuals on the y-axis;
2. on the x-axis, the fit \$\hat y\$$ and on the y-axis, the residuals;
3. on the x-axis, the table line number (index) and on the y-axis, the residuals.

\[`plt.plot`, `predict` method for adjusted instance/model and `np.arange` to generate line numbers using the `shape` attribute attribute of the DataFrame\]


```{code-cell} python

```

# Estimatation variability

## Data Loading
Load eucalyptus data into a pandas DataFrame `eucalypt`
\[`read_csv` of `pandas`\]. On Fun Campus the path is `data/eucalyptus.txt`.


```{code-cell} python

```

## Estimation on \$n=100\$ datapoints
Create two empty lists `beta1` and `beta2`.
Perform the following steps 500 times
1. Randomly draw 100 rows from the table `eucalypt` without replacement.
2. On this draw perform a simple linear regression
   where `circ` is the explanatory variable and `ht` the variable to be explained. Store estimated parameters in `beta1` and `beta2`.
   
\[create a random generator `np.random.default_rng` instance\]


```{code-cell} python

```

## Variability of \$\hat \beta_2\$
Represent the variability of the random variable \$\hat \beta_2\$.
\[a fonction of `plt`...\]


```{code-cell} python

```

## Dependance of \$\hat \beta_1\$ and \$\hat \beta_2\$
Plot the \$\hat \beta_1\$ and \$\hat \beta_2\$ pairs and note the
note the variability of the estimate and the correlation
between the two parameters.
\[a fonction of `plt`...\]


```{code-cell} python

```

# Two Models

## Data Loading
Load eucalyptus data into a pandas DataFrame `eucalypt`
\[`read_csv` of `pandas`\]. On Fun Campus the path is `data/eucalyptus.txt`.


```{code-cell} python

```

## Point Cloud
Plot the point cloud with `circ` as abscissa and `ht` as ordinate
and note that the points are not exactly on a straight line
a straight line, but rather a "square root" curve.
\[`plt.plot`\]


```{code-cell} python

```

## Two simple regressions
1. Perform a simple linear regression where `circ` is the explanatory variable
   the explanatory variable and `ht` the variable to be explained.
   Store the result in the `reg` object.
2. Perform a simple linear regression where the square root of `circ` is the explanatory variable and `ht` is the explanatory variable.
   is the explanatory variable and `ht` is the variable to be explained.
   Store the result in the `regsqrt` object. You can use
   operations and functions in formulas
   (see https://www.statsmodels.org/stable/example_formulas.html)
\[`ols` method from `smf`, `fit` method from `OLS` class, 
`summary` method for the adjusted instance/model\]


```{code-cell} python

```

## Comparison
Add the 2 fits (the straight line and the square root) to the scatterplot
and choose the best model.
\[method `argsort` on a DataFrame column and `plt.plot`\]


```{code-cell} python

```
