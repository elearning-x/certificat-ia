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
  title: 'Correction du TP comparaison de méthodes de régression logistique'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa BEDIN<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

# Modules python

Importer les modules pandas (comme `pd`) numpy (commme `np`) matplotlib.pyplot (comme `plt`) et statsmodels.formula.api (comme `smf`).

```{code-cell} python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
```


# Importation des données

```{code-cell} python
spam = pd.read_csv("data/spambase.data", header=None,  sep=",")
spam.rename(columns={spam.columns[57] : "Y"}, inplace=True)
namestr = ["X"+str(i) for i in spam.columns[0:57] ]
spam.rename(columns= dict(zip(list(spam.columns[0:57]), namestr)), inplace=True)
```


# Séparation des données en deux parties

Nous pouvons utiliser la fonction `sklearn.model_selection.GroupShuffleSplit` pour cela

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


# Modèles


## DataFrame

Constituons le DataFrame

```{code-cell} python
RES = pd.DataFrame(spamVal.Y)
```


## logistique complet

```{code-cell} python
RES  = RES.assign(Logistic=0.0)
regLog = smf.logit('Y~X0+X1+X2+X3+X4+X5+X6+X7+X8+X9+X10+X11+X12+X13+X14+X15+X16+X17+X18+X19+X20+X21+X22+X23+X24+X25+X26+X27+X28+X29+X30+X31+X32+X33+X34+X35+X36+X37+X38+X39+X40+X41+X42+X43+X44+X45+X46+X47+X48+X49+X50+X51+X52+X53+X54+X55+X56', data=spamApp).fit()
RES.Logistic = regLog.predict(spamVal)>0.5
```


## logistique selection

Importons la fonction

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

```{code-cell} python

```

Puis utilisons la:

```{code-cell} python
RES  = RES.assign(SelLogistic=0.0)
regSelLog = logitbackward(spamApp,"Y~X0+X1+X2+X3+X4+X5+X6+X7+X8+X9+X10+X11+X12+X13+X14+X15+X16+X17+X18+X19+X20+X21+X22+X23+X24+X25+X26+X27+X28+X29+X30+X31+X32+X33+X34+X35+X36+X37+X38+X39+X40+X41+X42+X43+X44+X45+X46+X47+X48+X49+X50+X51+X52+X53+X54+X55+X56")
RES.SelLogistic = regSelLog.predict(spamVal)>0.5
```


## Ridge, lasso, elastic net

Importons toutes les fonctions nécessaires provenant du module `sklearn`:

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

Instancions les modèles:

```{code-cell} python
# instanciation steps
cr = StandardScaler()
lassocv =  LogisticRegressionCV(cv=10, penalty="l1", n_jobs=3, solver="liblinear")
ridgecv = LogisticRegressionCV(cv=10, penalty="l2", n_jobs=3)
enetcv = LogisticRegressionCV(cv=10, penalty="elasticnet", n_jobs=3, solver="saga", l1_ratios=[0.5])
# instanciation pipeline
pipe_lassocv = Pipeline(steps=[("cr", cr), ("lassocv", lassocv)])
pipe_ridgecv = Pipeline(steps=[("cr", cr), ("ridgecv", ridgecv)])
pipe_enetcv = Pipeline(steps=[("cr", cr), ("enetcv", enetcv)])
```

Nous estimons tous les modèles et nous les utilisons pour prédire les labels (la fontion `predict` retourne des labels et pas la probabilité)

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


# Résumé final

Une observation $i$ est mal classée si $|Y_i - \hat Y_i|=1$ et bien classée si $|Y_i - \hat Y_i|=0$. Le taux de mal classés est donc :

```{code-cell} python
methode = []
for j in range(2,RES.shape[1]):
    methode.append((RES.Y-RES.iloc[:,j]).abs().mean())
methode = np.array(methode)
print(methode.argmin())
```

La meilleure méthode est ici la régression logistique simple