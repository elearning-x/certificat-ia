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
  title: 'Correction du TP de comparaison des méthodes de régressions'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa BEDIN<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

+++

```{code-cell} python
import scipy
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
# scikitlearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,  Ridge, ElasticNet, Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold  GridSearchCV
```

## Données




```{code-cell} python
ozone = pd.read_csv("data/ozone_transf.txt", header = 0, sep = ";", index_col=0)
ozone.shape
X = ozone.iloc[:,1:]
y = ozone.iloc[:,:1]
```

## decoupage en 10 fold et df résultats




```{code-cell} python
kf = KFold(n_splits=10, shuffle=True, random_state=0)
RES = pd.DataFrame(ozone.maxO3)
```

## Premiers modeles



### reg multiple



#### MCO complet




```{code-cell} python
RES  = RES.assign(MCO=0.0)
for app_index, val_index in kf.split(X):
    Xapp = X.iloc[app_index,:]
    yapp = y.iloc[app_index,:]
    Xval = X.iloc[val_index,:]
    yval = y.iloc[val_index,:]
    reg_lin = LinearRegression().fit(X=Xapp, y=yapp)
    RES.MCO.iloc[val_index] = reg_lin.predict(Xval).ravel()
```

#### MCO choix par BIC



La fonction de selection backward (déjà vue en TP)




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

et la modélisation




```{code-cell} python
RES = RES.assign(choix=0.0)
for app_index, val_index in kf.split(X):
    Xapp = X.iloc[app_index, :]
    yapp = y.iloc[app_index, :]
    Xval = X.iloc[val_index, :]
    yval = y.iloc[val_index, :]
    reg_bic = olsbackward(Xapp, 'maxO3v~ T6+T9+T12+T15+T18+Ne6+Ne9+Ne12+Ne15+Ne18+Vx6+Vy6+Vx9+Vy9+Vx12+Vy12+Vx15+Vy15+Vx18+Vy18')
    RES.choix.iloc[val_index] = reg_bic.predict(Xval).values
```

## Ridge, lasso, elastic net




```{code-cell} python
RES  = RES.assign(lasso=0.0, enet=0.0, ridge=0.0)
for app_index, val_index in kf.split(X):
    Xapp = X.iloc[app_index,:]
    yapp = y.iloc[app_index,:]
    Xval = X.iloc[val_index,:]
    yval = y.iloc[val_index,:]
    # instanciation steps
    cr = StandardScaler()
    lassocv = LassoCV(cv=kf, n_jobs=3)
    enetcv = ElasticNetCV(cv=kf, n_jobs=3)
    # instanciation pipeline
    pipe_lassocv = Pipeline(steps=[("cr", cr), ("lassocv", lassocv)])
    pipe_enetcv = Pipeline(steps=[("cr", cr), ("enetcv", enetcv)])
    ## fit
    pipe_lassocv.fit(Xapp, yapp.iloc[:,0].ravel())
    pipe_enetcv.fit(Xapp, yapp.iloc[:,0].ravel())
    ## ridge : path
    etape_lasso = pipe_lassocv.named_steps["lassocv"]
    path_ridge = etape_lasso.alphas_ * 100
    # intanciations
    ridge = Ridge()
    pipe_ridge = Pipeline(steps=[("cr", cr), ("ridge", ridge)])
    ## params lambda
    param_grid_ridge = {"ridge__alpha": path_ridge}
    ## GridSearchCV
    cv_ridge = GridSearchCV(pipe_ridge, param_grid_ridge, cv=kf, scoring = 'neg_mean_squared_error', n_jobs=3).fit(Xapp, yapp)
    RES.lasso.iloc[val_index] = pipe_lassocv.predict(Xval).ravel()
    RES.enet.iloc[val_index] = pipe_enetcv.predict(Xval).ravel()
    RES.ridge.iloc[val_index] = cv_ridge.predict(Xval).ravel()
```
