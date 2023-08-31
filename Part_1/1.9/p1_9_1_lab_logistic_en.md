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

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa Bedin<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

+++

## Python Modules

+++

### Importing Python Modules



Import the following modules: pandas (as `pd`), numpy (as `np`), matplotlib.pyplot (as `plt`), and statsmodels.formula.api (as `smf`).




```{code-cell} python

```

## Logistic Regression



### Importing Data



Import the data from `artere.txt` into the pandas DataFrame `artere` using `read_csv` from `numpy`. The file path on FunStudio is `data/artere.txt`. Besides age and the presence (1) or absence (0) of cardiovascular disease (`chd`), there is a qualitative variable with 8 categories representing age groups (`agegrp`).




```{code-cell} python

```

### Scatter Plot



Plot a scatter plot with age on the x-axis and `chd` on the y-axis using `plt.plot`.




```{code-cell} python

```

### Logistic Regression



Perform a logistic regression with `age` as the explanatory variable and `chd` as the binary response variable. Store the result in the `reg` object. Steps:

1.  Perform a summary of the model.
2.  Display the parameters estimated by logistic regression.




```{code-cell} python

```

### Prediction and Estimated Probabilities



Display the predictions for the sample data using the `predict` method (without arguments) on the `reg` model. What does this vector represent?

-   Probability of having a disease for each age value in the sample.
-   Probability of not having a disease for each age value in the sample.
-   Prediction of the disease/non-disease state for each age value in the sample.




```{code-cell} python

```

Numerically verify that the prediction with the argument `which='prob'` is simply an indicator that $\hat p(x)>s$, where $s$ is the classic threshold of 0.5.




```{code-cell} python

```

### Confusion Matrix



Display the estimated confusion matrix for the sample data using a threshold of 0.5.




```{code-cell} python

```

### Residuals



Graphically represent the deviance residuals:

1.  Age on the x-axis and deviance residuals on the y-axis (using the `resid_dev` attribute of the model).
2.  Row index on the x-axis and residuals on the y-axis (using `plt.plot`, `predict` method on the fitted model, and `np.arange` to generate row numbers using the `shape` attribute of the DataFrame).




```{code-cell} python

```

## Data Simulation: Variability of $\hat \beta_2$



### Simulation



1.  Generate $n=100$ values of $X$ uniformly between 0 and 1.
2.  For each value $X_i$, simulate $Y_i$ according to a logistic model with parameters $\beta_1=-5$ and $\beta_2=10$.




```{code-cell} python

```

### Estimation



Estimate the parameters $\beta_1$ and $\beta_2$.




```{code-cell} python

```

### Estimation Variability



Repeat the previous two steps 500 times and observe the variability of $\hat \beta_2$ using an appropriate graph.




```{code-cell} python

```

## Two Simple Logistic Regressions



### Importing Data



Import the data from `artere.txt` into the pandas DataFrame `artere` using `read_csv` from `numpy`. The file path on FunStudio is `data/artere.txt`. Besides age and the presence (1) or absence (0) of cardiovascular disease (`chd`), there is a qualitative variable with 8 categories representing age groups (`agegrp`).




```{code-cell} python

```

### Two Logistic Regressions



1.  Perform a simple logistic regression with `age` as the explanatory variable and `chd` as the binary response variable.
2.  Repeat the same with the square root of `age` as the explanatory variable.




```{code-cell} python

```

### Comparison



Add both the linear and &ldquo;square root&rdquo; adjustments to the scatter plot and choose the best model based on a numerical criterion (using `argsort` on a DataFrame column and `plt.plot` ; using summary results).




```{code-cell} python

```
