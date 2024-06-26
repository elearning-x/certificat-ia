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
  title: 'Lab Session on Logistic Regression'
  version: '1.0'
---

```{list-table} 
:header-rows: 0
:widths: 33% 34% 33%

* - ![Logo](media/logo_IPParis.png)
  - Lisa BEDIN<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER
  - Licence CC BY-NC-ND
```

+++

# Python Modules

## Importing Python Modules

Import the following modules: pandas (as `pd`), numpy (as `np`), matplotlib.pyplot (as `plt`), and statsmodels.formula.api (as `smf`).

```{code-cell} python

```


# Logistic Regression


## Importing Data

Import the data from `artere.txt` into the pandas DataFrame `artere` using `read_csv` from `numpy`. The file path on Fun Campus is `data/artere.txt`. Besides age and the presence (1) or absence (0) of cardiovascular disease (`chd`), there is a qualitative variable with 8 categories representing age groups (`agegrp`).

```{code-cell} python

```


## Scatter Plot

Plot a scatter plot with age on the x-axis and `chd` on the y-axis using `plt.plot`.

```{code-cell} python

```


## Logistic Regression

Perform a logistic regression with `age` as the explanatory variable and `chd` as the binary response variable. Store the result in the `reg` object. Steps:

1.  Perform a summary of the model.
2.  Display the parameters estimated by logistic regression.

\[`logit` from `smf`, method `fit`, method `summary` from instance/fitted model, attribute `params`.\]

```{code-cell} python

```


## Prediction and Estimated Probabilities

Display the predictions for the sample data using the `predict` method (without arguments) on the `modele` model. What does this vector represent?

-   Probability of having a disease for each age value in the sample.
-   Probability of not having a disease for each age value in the sample.
-   Prediction of the disease/non-disease state for each age value in the sample.

```{code-cell} python

```

Display the prediction of the disease status (sick/healthy) with the indicator that $\hat p(x)>s$, where $s$ is the classic threshold of 0.5.

```{code-cell} python

```


## Confusion Matrix

Display the estimated confusion matrix for the sample data using a threshold of 0.5.

```{code-cell} python

```

\[method `pred_table` on modeling and/or method `predict` on modeling and `pd.crosstab` on a two columns DataFrame created for this purpose\]


## Residuals

Graphically represent the deviance residuals:

1.  Age on the x-axis and deviance residuals on the y-axis (using the `resid_dev` attribute of the model).
2.  Make a random permutation on row index and use it on the x-axis and use the residuals on the y-axis (using `plt.plot`, `predict` method on the fitted model, and `np.arange` to generate row numbers using the `shape` attribute of the DataFrame ; create an instance of the default random generator using `np.random.default_rng` and use `rng.permutation`

on row index).

```{code-cell} python

```


# Data Simulation: Variability of $\hat \beta_2$


## Simulation

1.  Generate $n=100$ values of $X$ uniformly between 0 and 1.
2.  For each value $X_i$, simulate $Y_i$ according to a logistic model with parameters $\beta_1=-5$ and $\beta_2=10$.

\[ create an instance of the default random generator using `np.random.default_rng` and use `rng.binomial` \]

```{code-cell} python

```


## Estimation

Estimate the parameters $\beta_1$ and $\beta_2$.

```{code-cell} python

```


## Estimation Variability

Repeat the previous two steps 500 times and observe the variability of $\hat \beta_2$ using an appropriate graph.

```{code-cell} python

```


# Two Simple Logistic Regressions


## Importing Data

Import the data from `artere.txt` into the pandas DataFrame `artere` using `read_csv` from `numpy`. The file path on Fun Campus is `data/artere.txt`. Besides age and the presence (1) or absence (0) of cardiovascular disease (`chd`), there is a qualitative variable with 8 categories representing age groups (`agegrp`).

```{code-cell} python

```


## Two Logistic Regressions

1.  Perform a simple logistic regression with `age` as the explanatory variable and `chd` as the binary response variable.
2.  Repeat the same with the square root of `age` as the explanatory variable.

```{code-cell} python

```


## Comparison

Add both the linear and "square root" adjustments to the scatter plot and choose the best model based on a numerical criterion (using `argsort` on a DataFrame column and `plt.plot` ; using summary results).

```{code-cell} python

```