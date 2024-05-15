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
  title: 'Solutions to Lab Session on Logistic Regression, Threshold and Confusion Matrix'
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
```


# Logistic Regression


## Importing Data

Import the data from `artere.txt` into the pandas DataFrame `artere` using `read_csv` from `numpy`. The file path on Fun Campus is `data/artere.txt`. Besides age and the presence (1) or absence (0) of cardiovascular disease (`chd`), there is a qualitative variable with 8 categories representing age groups (`agegrp`).

```{code-cell} python
artere = pd.read_csv("data/artere.txt", header=0, sep=" ")
```


## Logistic Regression

Perform a logistic regression with `age` as the explanatory variable and `chd` as the binary response variable. Store the result in the `modele` object. Steps:

1.  Perform a summary of the model.
    
```{code-cell} python
modele = smf.logit('chd~age', data=artere).fit()
print(modele.summary())
```


## Confusion Matrix

Display the estimated confusion matrix for the sample data using a threshold of 0.5.

The first method is

```{code-cell} python
yhat = modele.predict()>0.5
df = pd.DataFrame({"yhat" : yhat, "chd": artere.chd})
pd.crosstab(index=df['chd'], columns=df['yhat'])
```

but a direct method can be used (only for fitted confusion matrix)

```{code-cell} python
modele.pred_table(threshold=0.5)
```


## Residuals

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

No observation have an absolute value of residual really high (in comparison to others): the modeling fits well the data.


# Variable selection


## data importation

As usually data in Fun Campus are inside `data/` directory. We rename all variable as `X0, X1, ..., X57` and `Y`

```{code-cell} python
spam = pd.read_csv("data/spambase.data", header=None,  sep=",")
spam.rename(columns={spam.columns[57] : "Y"}, inplace=True)
namestr = ["X"+str(i) for i in spam.columns[0:57] ]
spam.rename(columns= dict(zip(list(spam.columns[0:57]), namestr)), inplace=True)
```


## Sélection descendante/backward

We just replace `ols` by `logit` in the code of variable selection (Part 1.7)

The function start with the full model.

-   We separe the response variable (object `response`) from the explanatory variables,
-   These variables are transformed in a set (object `start_explanatory`),
-   The lower set is the empty set (object `lower_explanatory`),
-   The potential variable to be removed are obtained as the difference (object `remove`),

We initialize the set of selected variables (object `selected`) and make our first formula using all selected variables and add the intercept in the last position. Using `smf.logit` and using `smf.logit` we get AIC or BIC of the starting model (`current_score`).

The while loop begins:

-   for every variable (for loop) to be removed we make a regression modeling with the current set of variables minus that candidate variable. We constuct a list of triplet `score` (AIC/BIC) sign (always "-" as we are doing backward selection) and candidate variable to remove from the current model.

-   at the end of the for loop , we sort all the list of triplet using score and if the best triplet have a `score` better than `current_score` we update `remove` `selected` and `current_score`, if not we break the while loop.

At the end we fit the current model and return it. La fonction commence avec le modèle complet.

-   Nous séparons la variable réponse (objet `response`) des variables explicatives,
-   Ces dernières sont transformées en un ensemble (objet `start_explanatory`),
-   L'ensemble le plus petit est l'ensemble vide (objet `lower_explanatory`),
-   Les variables potentielles à supprimer sont obtenues par différence (objet `remove`),

Nous initialisons l'ensemble des variables sélectionnées (objet `selected`) et réalisons notre première formule en utilisant toutes les variables sélectionnées. En utilisant `smf.logit` nous obtenons l'AIC ou le BIC du modèle de départ (`current_score`).

La boucle while commence :

-   pour chaque variable (boucle for) à supprimer, nous effectuons une régression avec l'ensemble actuel des variables moins cette variable candidate. Nous construisons une liste de triplets `score` (AIC/BIC), signe (toujours "-" car nous effectuons une sélection backward) et la variable candidate à supprimer du modèle actuel.

-   A la fin de la boucle for, nous trions toute la liste des triplets en utilisant le score et si le meilleur triplet a un `score` meilleur que `current_score` nous mettons à jour `remove`, `selected` et `current_score`, si ce n'est pas le cas, nous interrompons la boucle while.

A la fin, nous ajustons le modèle actuel et le renvoyons comme résultat.

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

Using the function:

```{code-cell} python
modelefinal = logitbackward(spam,"Y~X0+X1+X2+X3+X4+X5+X6+X7+X8+X9+X10+X11+X12+X13+X14+X15+X16+X17+X18+X19+X20+X21+X22+X23+X24+X25+X26+X27+X28+X29+X30+X31+X32+X33+X34+X35+X36+X37+X38+X39+X40+X41+X42+X43+X44+X45+X46+X47+X48+X49+X50+X51+X52+X53+X54+X55+X56", verbose=True)
```

And the final/selected model:

```{code-cell} python
print(modelefinal.summary())
```