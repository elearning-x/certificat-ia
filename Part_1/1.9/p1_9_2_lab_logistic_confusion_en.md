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
  title: 'Lab Session on Logistic Regression, Threshold and Confusion Matrix'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa BEDIN<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

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

## Logistic Regression

Perform a logistic regression with `age` as the explanatory variable and `chd` as the binary response variable. Store the result in the `modele` object.

```{code-cell} python

```

## Confusion Matrix

Display the estimated confusion matrix for the sample data using a threshold of 0.5.

\[method `pred_table` on modelling and/or method `predict` on modelling and `pd.crosstab` on a DataFrame of 2 columns created for that purpose\]

```{code-cell} python

```

## Residuals

Graphically represent the deviance residuals:

1.  Age on the x-axis and deviance residuals on the y-axis (using the `resid_dev` attribute of the model).
2.  Make a random permutation on row index and use it on the x-axis and use the residuals on the y-axis (using `plt.plot`, `predict` method on the fitted model, and `np.arange` to generate row numbers using the `shape` attribute of the DataFrame ; create an instance of the default random generator using `np.random.default_rng` and use `rng.permutation`

on row index).
```{code-cell} python

```

