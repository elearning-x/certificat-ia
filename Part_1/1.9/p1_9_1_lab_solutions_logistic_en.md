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
  title: 'Solutions to Lab Session on Logistic Regression'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa BEDIN<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

+++

## Python Modules



### Importing Python Modules



Import the following modules: pandas (as `pd`), numpy (as `np`), matplotlib.pyplot (as `plt`), and statsmodels.formula.api (as `smf`).




```{code-cell} python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
```

## Logistic Regression



### Importing Data



Import the data from `artere.txt` into the pandas DataFrame `artere` using `read_csv` from `numpy`. The file path on Fun Campus is `data/artere.txt`. Besides age and the presence (1) or absence (0) of cardiovascular disease (`chd`), there is a qualitative variable with 8 categories representing age groups (`agegrp`).




```{code-cell} python
artere = pd.read_csv("data/artere.txt", header=0, sep=" ")
```

### Scatter Plot



Plot a scatter plot with age on the x-axis and `chd` on the y-axis using `plt.plot`.




```{code-cell} python
plt.scatter(x="age", y="chd", c="chd", data=artere)
plt.show()
```

### Logistic Regression



Perform a logistic regression with `age` as the explanatory variable and `chd` as the binary response variable. Store the result in the `reg` object. Steps:

1.  Perform a summary of the model.




```{code-cell} python
modele = smf.logit('chd~age', data=artere).fit()
print(modele.summary())
```

The modeling has a log-likelihood of -53.677. The modeling using only the intercept (named as the Null model here) have a log-likelihood of -68.331. Adding `age` to the Null model (and obtaining our `modele`) leads to a substantial improvement of the log-likelihood.

1.  Display the parameters estimated by logistic regression.




```{code-cell} python
print(modele.params)
```

### Prediction and Estimated Probabilities



Display the predictions for the sample data using the `predict` method (without arguments) on the `reg` model. What does this vector represent?

-   Probability of having a disease for each age value in the sample.
    (YES we modelise the probability of $Y=1$ and in this data sample we have $Y_i$ whenever the person have the CHD)
-   Probability of not having a disease for each age value in the sample. (NO)
-   Prediction of the disease/non-disease state for each age value in the sample (NO, by default the function returns the estimated probability of $Y=1$)




```{code-cell} python
modele.predict()
```

Display the prediction of the disease status (sick/healthy) with the indicator that $\hat p(x)>s$, where $s$ is the classic threshold of 0.5.




```{code-cell} python
print(modele.predict()>0.5)
```

### Confusion Matrix



Display the estimated confusion matrix for the sample data using a threshold of 0.5.

The first method is




```{code-cell} python
yhat = modele.predict()>0.5
pd.crosstab(index=df['Age'], columns=df['Grade'])
```

but a direct method can be used (only for fitted confusion matrix)




```{code-cell} python
modele.pred_table(threshold=0.5)
```

### Residuals



Graphically represent the deviance residuals:

1.  Age on the x-axis and deviance residuals on the y-axis (using the `resid_dev` attribute of the model).
2.  Make a random permutation on row index and use it on the x-axis and use the residuals on the y-axis (using `plt.plot`, `predict` method on the fitted model, and `np.arange` to generate row numbers using the `shape` attribute of the DataFrame ; create an instance of the default random generator using `np.random.default_rng` and use `rng.permutation`

on row index).




```{code-cell} python
plt.plot(artere.age, modele.resid_dev, "+")
plt.show()
```

We get the usual shape of residuals vs $\hat p$ (or age here). This kind of graphics is not used.




```{code-cell} python
rng = np.random.default_rng(seed=1234)
indexp = rng.permutation(np.arange(artere.shape[0]))
plt.plot(indexp, modele.resid_dev, "+")
plt.show()
```

No observation have an absolute value of residual really high (in comparison to others):  the modeling fits well the data.



## Data Simulation: Variability of $\hat \beta_2$



### Simulation



1.  Generate $n=100$ values of $X$ uniformly between 0 and 1.
2.  For each value $X_i$, simulate $Y_i$ according to a logistic model with parameters $\beta_1=-5$ and $\beta_2=10$.

[  create an instance of the default random generator using `np.random.default_rng` and use `rng.binomial` ]




```{code-cell} python
rng = np.random.default_rng(seed=123)
X = rng.uniform(size=100)
Y = np.copy(X)
for i,xi in enumerate(X):
    proba = 1 / (1 + np.exp( -(-5 + 10 * xi) ))
    Y[i]=rng.binomial(1, proba, 1)[0]
df = pd.DataFrame({"X" : X, "Y" : Y})
```

### Estimation



Estimate the parameters $\beta_1$ and $\beta_2$.




```{code-cell} python
modele = smf.logit('Y~X', data=df).fit()
print(modele.params)
```

### Estimation Variability



Repeat the previous two steps 500 times and observe the variability of $\hat \beta_2$ using an appropriate graph.

We simulate 500 times the data and fit the logistic model:




```{code-cell} python
hatbeta2 = []
for it in range(500):
    X = rng.uniform(size=100)
    Y = np.copy(X)
    for i,xi in enumerate(X):
        proba = 1 / (1 + np.exp( -(-5 + 10 * xi) ))
        Y[i]=rng.binomial(1, proba, 1)[0]
    df = pd.DataFrame({"X" : X, "Y" : Y})
    modele = smf.logit('Y~X', data=df).fit()
    hatbeta2.append(modele.params[1])
```

and we can disply an histogram of $\hat \beta_2$




```{code-cell} python
plt.hist(hatbeta2, bins=30)
plt.show()
```

## Two Simple Logistic Regressions



### Importing Data



Import the data from `artere.txt` into the pandas DataFrame `artere` using `read_csv` from `numpy`. The file path on Fun Campus is `data/artere.txt`. Besides age and the presence (1) or absence (0) of cardiovascular disease (`chd`), there is a qualitative variable with 8 categories representing age groups (`agegrp`).




```{code-cell} python
artere = pd.read_csv("data/artere.txt", header=0, sep=" ")
```

### Two Logistic Regressions



1.  Perform a simple logistic regression with `age` as the explanatory variable and `chd` as the binary response variable.
2.  Repeat the same with the square root of `age` as the explanatory variable.




```{code-cell} python
modele1 = smf.logit('chd~age', data=artere).fit()
modele2 = smf.logit('chd~I(np.sqrt(age))', data=artere).fit()
```

### Comparison



Add both the linear and &ldquo;square root&rdquo; adjustments to the scatter plot and choose the best model based on a numerical criterion (using `argsort` on a DataFrame column and `plt.plot` ; using summary results).




```{code-cell} python
sel = artere['age'].argsort()
plt.scatter(x="age", y="chd", c="chd", data=artere)
plt.plot(artere.age.iloc[sel], modele1.predict()[sel], "b-", artere.age.iloc[sel], modele2.predict()[sel], "r-"  )
plt.show()
```

As the two modeling have the same number of explanatory variable (two) we can compare the log-likelihood: the higher the better. The `modele1` is better (but the log-likelihood are quite similar though).


