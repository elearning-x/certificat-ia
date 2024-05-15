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
  title: 'Solutions to Lab Session on Variable Selection'
  version: '1.1'
---

```{list-table} 
:header-rows: 0
:widths: 33% 34% 33%

* - ![Logo](media/logo_IPParis.png)
  - Lisa BEDIN<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER
  - Licence CC BY-NC-ND
```

+++

# Modules

Import the modules pandas (as `pd`), numpy (as `np`), matplotlib.pyplot (as `plt`), and statsmodels.formula.api (as `smf`).

```{code-cell} python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
```


# Ridge Regression on Ozone Data


## Data Import

Import the ozone data `ozonecomplet.csv` (in Fun Campus, data is located in the `data/` directory) and remove the last two variables (categorical). Then provide a summary of numerical variables. \[Use the `astype` method on the DataFrame column and the `describe` method on the DataFrame instance. fs\]

```{code-cell} python
ozone = pd.read_csv("data/ozonecomplet.csv", header=0, sep=";")
ozone = ozone.drop(['nomligne', 'Ne', 'Dv'], axis=1)
ozone.describe()
```


## Backward Selection

Propose a function that performs backward selection. It will use the formulas from `statsmodels` and will always include the intercept. The function will take three arguments as input: the DataFrame of data, the initial formula, and the criterion (AIC or BIC). The function will return the estimated model

The function start with the full model.

-   We separe the response variable (object `response`) from the explanatory variables,
-   These variables are transformed in a set (object `start_explanatory`),
-   The lower set is the empty set (object `lower_explanatory`),
-   The potential variable to be removed are obtained as the difference (object `remove`),

We initialize the set of selected variables (object `selected`) and make our first formula using all selected variables and add the intercept in the last position. Using `smf.ols` and using `smf.ols` we get AIC or BIC of the starting model (`current_score`).

The while loop begins:

-   for every variable (for loop) to be removed we make a regression modeling with the current set of variables minus that candidate variable. We constuct a list of triplet `score` (AIC/BIC) sign (always "-" as we are doing backward selection) and candidate variable to remove from the current model.

-   at the end of the for loop , we sort all the list of triplet using score and if the best triplet have a `score` better than `current_score` we update `remove` `selected` and `current_score`, if not we break the while loop.

At the end we fit the current model and return it.

```{code-cell} python
def olsbackward(data, start, crit="aic", verbose=False):
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
        current_score = smf.ols(formula, data).fit().aic
    elif crit == "bic" or crit == "BIC":
        current_score = smf.ols(formula, data).fit().bic
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
                score = smf.ols(formula, data).fit().aic
            elif crit == "bic" or crit == "BIC":
                score = smf.ols(formula, data).fit().bic
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
    model = smf.ols(formula, data).fit()
    return model
```

Using the function:

```{code-cell} python
modelefinal = olsbackward(ozone,"O3~T9+T12+T15+Ne9+Ne12+Ne15+Vx9+Vx12+Vx15+O3v", verbose=True)
```

And the final/selected model:

```{code-cell} python
modelefinal.summary()
```