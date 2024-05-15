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
  title: 'Solutions to Lab Session on Model Selection for Logistic Regression'
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

# Python module

Module importation

```{code-cell} python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
```


# Data importation

```{code-cell} python
spam = pd.read_csv("data/spambase.data", header=None,  sep=",")
spam.rename(columns={spam.columns[57] : "Y"}, inplace=True)
namestr = ["X"+str(i) for i in spam.columns[0:57] ]
spam.rename(columns= dict(zip(list(spam.columns[0:57]), namestr)), inplace=True)
```


# Data splitting

We can use the function `sklearn.model_selection.GroupShuffleSplit` as follows:

```{code-cell} python
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
y = spam.iloc[:,57]
X = spam.iloc[:,0:57]
for app_index, val_index in sss.split(X,y):
    print(spam.iloc[val_index,57].mean())
    spamApp = spam.iloc[app_index,:]
    print(spam.iloc[app_index,57].mean())
    spamVal = spam.iloc[val_index,:]
```


# Modelling


## DataFrame

We assign the first column to the DataFrame which will contains the results:

```{code-cell} python
RES = pd.DataFrame(spamVal.Y)
```


## logistic modelling

We can fit the full model as follows:

```{code-cell} python
RES  = RES.assign(Logistic=0.0)
regLog = smf.logit('Y~X0+X1+X2+X3+X4+X5+X6+X7+X8+X9+X10+X11+X12+X13+X14+X15+X16+X17+X18+X19+X20+X21+X22+X23+X24+X25+X26+X27+X28+X29+X30+X31+X32+X33+X34+X35+X36+X37+X38+X39+X40+X41+X42+X43+X44+X45+X46+X47+X48+X49+X50+X51+X52+X53+X54+X55+X56', data=spamApp).fit()
RES.Logistic = regLog.predict(spamVal)>0.5
```


## Variable selection

Let us import the function for variable selection

```{code-cell} python
def logitbackward(data, start, crit="aic", verbose=False):
    """Backward selection for linear model with smf (with formula).

    Parameters:
    -----------
    data (pandas DataFrame): DataFrame with all possible predictors
            and response
    start (string): a string giving the starting model
            (ie the starting point)
    crit (string): "aic"/"AIC" or "bic"/"BIC"
    verbose (boolean): if True verbose print

    Returns:
    --------
    model: an "optimal" linear model fitted with statsmodels
           with an intercept and
           selected by forward/backward or both algorithm with crit criterion
    """
    # criterion
    if not (crit == "aic" or crit == "AIC" or crit == "bic" or crit == "BIC"):
        raise ValueError("criterion error (should be AIC/aic or BIC/bic)")
    # starting point
    formula_start = start.split("~")
    response = formula_start[0].strip()
    # explanatory variables for the 3 models
    start_explanatory = set([item.strip() for item in
                             formula_start[1].split("+")]) - set(['1'])
    # setting up the set "remove" which contains the possible
    # variable to remove
    lower_explanatory = set([])
    remove = start_explanatory - lower_explanatory
    # current point
    selected = start_explanatory
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(list(selected)))
    if crit == "aic" or crit == "AIC":
        current_score = smf.logit(formula, data).fit().aic
    elif crit == "bic" or crit == "BIC":
        current_score = smf.logit(formula, data).fit().bic
    if verbose:
        print("----------------------------------------------")
        print((current_score, "Starting", selected))
    # main loop
    while True:
        scores_with_candidates = []
        for candidate in remove:
            tobetested = selected - set([candidate])
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(list(tobetested)))
            if crit == "aic" or crit == "AIC":
                score = smf.logit(formula, data).fit().aic
            elif crit == "bic" or crit == "BIC":
                score = smf.logit(formula, data).fit().bic
            if verbose:
                print((score, "-", candidate))
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop(0)
        if current_score > best_new_score:
            remove = remove - set([best_candidate])
            selected = selected - set([best_candidate])
            current_score = best_new_score
            if verbose:
                print("----------------------------------------------")
                print((current_score, "New Current", selected))
        else:
            break
    if verbose:
        print("----------------------------------------------")
        print((current_score, "Final", selected))
    formula = "{} ~ {} + 1".format(response, ' + '.join(list(selected)))
    model = smf.logit(formula, data).fit()
    return model
```

And we use it:

```{code-cell} python
RES  = RES.assign(SelLogistic=0.0)
regSelLog = logitbackward(spamApp,"Y~X0+X1+X2+X3+X4+X5+X6+X7+X8+X9+X10+X11+X12+X13+X14+X15+X16+X17+X18+X19+X20+X21+X22+X23+X24+X25+X26+X27+X28+X29+X30+X31+X32+X33+X34+X35+X36+X37+X38+X39+X40+X41+X42+X43+X44+X45+X46+X47+X48+X49+X50+X51+X52+X53+X54+X55+X56")
RES.SelLogistic = regSelLog.predict(spamVal)>0.5
```


## Ridge, lasso, elastic net

After importing all the function from `sklearn` we build the `np.array` of explanatory variables (training and test sets) and the `np.array` of dependant variable (training and test sets)

```{code-cell} python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
RES  = RES.assign(lasso=0.0, enet=0.0, ridge=0.0)
##
Xapp = X.iloc[app_index,:].values
yapp = y.iloc[app_index, ].values
Xval = X.iloc[val_index,:].values
yval = y.iloc[val_index,].values
```

After we intantiate all the modelling as follows

```{code-cell} python
# instanciation steps
cr = StandardScaler()
lassocv = LogisticRegressionCV(cv=10, penalty="l1", n_jobs=3, solver="liblinear", max_iter=200)
ridgecv = LogisticRegressionCV(cv=10, penalty="l2", n_jobs=3, max_iter=1000)
enetcv = LogisticRegressionCV(cv=10, penalty="elasticnet", n_jobs=3, solver="saga", l1_ratios=[0.5], max_iter=10000)
# instanciation pipeline
pipe_lassocv = Pipeline(steps=[("cr", cr), ("lassocv", lassocv)])
pipe_ridgecv = Pipeline(steps=[("cr", cr), ("ridgecv", ridgecv)])
pipe_enetcv = Pipeline(steps=[("cr", cr), ("enetcv", enetcv)])
```

For some values of $\lambda$ for some folds, the algorithm (lasso) failed to converge. This usually occurs for uninteresting modelling (for example for small $\lambda$) and leads to worse modelling than if the algorithm was to converge. But as this occurs for uninteresting modelling this is harmless.

We fit/estimate all the models and use them for prediction (the `predict` method returns label not probability)

```{code-cell} python
## fit
pipe_lassocv.fit(Xapp, yapp)
pipe_ridgecv.fit(Xapp, yapp)
pipe_enetcv.fit(Xapp, yapp)
## predict
RES.lasso = pipe_lassocv.predict(Xval).ravel()
RES.ridge = pipe_ridgecv.predict(Xval).ravel()
RES.enet = pipe_enetcv.predict(Xval).ravel()
```


# Summary

An observation $i$ is missclassified if $|Y_i - \hat Y_i|=1$ and correctly classified if $|Y_i - \hat Y_i|=0$. Thus the missclassified percentage can be calculated as

```{code-cell} python
methode = []
for j in range(2,RES.shape[1]):
    methode.append((RES.Y-RES.iloc[:,j]).abs().mean())
methode = np.array(methode)
print(methode.argmin())
```

The best method is the logistic regression.
