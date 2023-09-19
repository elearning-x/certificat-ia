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
  title: Correction du TP choix de variables
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa BEDIN<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

# Modules

Importer les modules pandas (comme `pd`) numpy (commme `np`) matplotlib.pyplot (comme `plt`) et statsmodels.formula.api (comme `smf`).

```{code-cell} python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
```


# Régression ridge sur les données d'ozone


## Importation des données

Importer les données d'ozone `ozonecomplet.csv` (dans Fun Campus les données sont dans le répertoire `data/`) et éliminer les deux dernières variables (qualitatives) et faites un résumé numérique par variable \[méthode `astype` sur la colonne du DataFrame et méthode `describe` sur l'instance DataFrame\]

```{code-cell} python
ozone = pd.read_csv("data/ozonecomplet.csv", header=0, sep=";")
ozone = ozone.drop(['nomligne', 'Ne', 'Dv'], axis=1)
ozone.describe()
```


## Sélection descendante/backward

Proposer une fonction qui permet la sélection descendante/backward. Elle utilisera les formules de `statsmodels` et incluera toujours la constante. En entrée serviront trois arguments: le DataFrame des données, la formule de départ et le critère (AIC ou BIC). La fonction retournera le modèle estimé via `smf.ols`

La fonction commence avec le modèle complet.

-   Nous séparons la variable réponse (objet `response`) des variables explicatives,
-   Ces dernières sont transformées en un ensemble (objet `start_explanatory`),
-   L'ensemble le plus petit est l'ensemble vide (objet `lower_explanatory`),
-   Les variables potentielles à supprimer sont obtenues par différence (objet `remove`),

Nous initialisons l'ensemble des variables sélectionnées (objet `selected`) et réalisons notre première formule en utilisant toutes les variables sélectionnées. En utilisant `smf.ols` nous obtenons l'AIC ou le BIC du modèle de départ (`current_score`).

La boucle while commence :

-   pour chaque variable (boucle for) à supprimer, nous effectuons une régression avec l'ensemble actuel des variables moins cette variable candidate. Nous construisons une liste de triplets `score` (AIC/BIC), signe (toujours "-" car nous effectuons une sélection backward) et la variable candidate à supprimer du modèle actuel.

-   A la fin de la boucle for, nous trions toute la liste des triplets en utilisant le score et si le meilleur triplet a un `score` meilleur que `current_score` nous mettons à jour `remove`, `selected` et `current_score`, si ce n'est pas le cas, nous interrompons la boucle while.

A la fin, nous ajustons le modèle actuel et le renvoyons comme résultat.

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

La mise en oeuvre

```{code-cell} python
modelefinal = olsbackward(ozone,"O3~T9+T12+T15+Ne9+Ne12+Ne15+Vx9+Vx12+Vx15+O3v", verbose=True)
```

Le modèle sélectionné

```{code-cell} python
modelefinal.summary()
```