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
  title: 'Correction du TP régression Logistique: seuil et matrice de confusion'
  version: '1.0'
---

```{list-table} 
:header-rows: 0
:widths: 33% 34% 33%

* - ![Logo](media/logo_IPParis.png)
  - Lisa BEDIN<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER
  - Licence CC BY-NC-ND
```

# Modules python

Importer les modules pandas (comme `pd`) numpy (commme `np`) matplotlib.pyplot (comme `plt`) et statsmodels.formula.api (comme `smf`).

```{code-cell} python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
```


# Régression logistique (suite TP précédent)


## Importation des données

Importer les données `artere.txt` dans le DataFrame pandas `artere` \[`read_csv` de `numpy` \]. Sur Fun Campus le chemin est `data/artere.txt`. Outre l'age et la présence=1/absence=0 de la maladie cardio-vasculaire (`chd`) une variable qualitative à 8 modalités donne les classes d'age (`agegrp`)

```{code-cell} python
artere = pd.read_csv("data/artere.txt", header=0, sep=" ")
```


## Régression logistique

Effectuer une régression logistique où `age` est la variable explicative et `chd` la variable binaire à expliquer. Stocker le résultat dans l'objet `modele`

\[`logit` de `smf`, méthode `fit` \]

```{code-cell} python
modele = smf.logit('chd~age', data=artere).fit()
```


## Matrice de confusion (ajustée)

Afficher la matrice de confusion estimée sur les données de l'échantillon pour un seuil choisi à 0.5.

Une méthode manuelle est la suivante

```{code-cell} python
yhat = modele.predict()>0.5
df = pd.DataFrame({"yhat" : yhat, "chd": artere.chd})
pd.crosstab(index=df['chd'], columns=df['yhat'])
```

mais il existe aussi une fonction adptée uniquement à l'estimation de la matrice de confusion en ajustement:

```{code-cell} python
modele.pred_table(threshold=0.5)
```

Attention cette matrice de confusion est ajustée et reste donc très optimiste. Une matrice de confusion calculée par validation croisée ou apprentissage/validation est plus que conseillée !


## Résidus

Représenter graphiquement les résidus de déviance avec

1.  en abscisse la variable `age` et en ordonnée les résidus \[attribut `resid_dev` du modèle\];
2.  en abscisse le numéro de ligne du tableau (index) après permutation aléatoire et en ordonnées les résidus.

\[`plt.plot`, méthode `predict` pour l'instance/modèle ajusté et `np.arange` pour générer les numéros de ligne avec l'attribut `shape` du DataFrame ; créer une instance de générateur aléatoire `np.random.default_rng` et utiliser `rng.permutation` sur les numéros de ligne\]

```{code-cell} python
plt.plot(artere.age, modele.resid_dev, "+")
plt.show()
```

Nous retrouvons l'allure caractéristique du graphique résidus fonction de $\hat p$ (ou de l'age ici). Ce type de graphique n'est pas utilisé en pratique.

```{code-cell} python
rng = np.random.default_rng(seed=1234)
indexp = rng.permutation(np.arange(artere.shape[0]))
plt.plot(indexp, modele.resid_dev, "+")
plt.show()
```

Aucune observation avec des valeurs extrêmes, le modèle ajuste bien les données.


## Matrice de confusion (en prévision)

Ayant peu de données ici plutôt que d'évaluer la matrice de confusion en apprentissage/validation nous choisissons (contraints et forcés) d'évaluer celle-ci en validation croisée.


### Séparation en 10 blocs

Nous allons séparer le jeu de données en 10 blocs grâce à la fonction [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold) (du module `sklearn` sous module `model_selection`) créer une instance de `StratifiedKFold` nommée `skf`.

```{code-cell} python
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
```

Cela permet d'obtenir une répartition 0/1 quasiment identique de blocs en blocs. En effet la proportion de 1 dans chaque bloc de validation est ici:

```{code-cell} python
y = artere.chd.values
X = artere.values
for app_index, val_index in skf.split(X,y):
    print(val_index)
    print(artere.chd.iloc[val_index].mean())
```


### DataFrame prevision et $chd$

Créer un DataFrame `res` avec deux colonnes: la variable $chd$ et une second colonne remplie de 0 qui contiendra les prévisions. Cette colonne pourra être nommée `yhat`

```{code-cell} python
res = pd.DataFrame(artere.chd)
res["yhat"] = 0
```

Ajouter au fur et à mesure les prévisions en validation croisée dans la deuxième colonne: Pour chaque «tour» de bloc faire

1.  estimer sur les 9 blocs en apprentissage le modèle de régression logistique
2.  prévoir les données du bloc en validation (seuil $s=1/2$)
3.  ranger dans les lignes correspondantes de `res` les prévisions (dans la colonne `yhat`)

```{code-cell} python
y = artere.chd.values
X = artere.iloc[:,0:2].values
s = 0.5
for app_index, val_index in skf.split(X,y):
    artereapp=artere.iloc[app_index,:]
    artereval=artere.iloc[val_index,:]
    modele = smf.logit('chd~age', data=artereapp).fit()
    res.iloc[val_index,1] = modele.predict(exog=artereval)>s
```


### Calculer la matrice de confusion

Avec la fonction `crosstab` du module `pd` proposer la matrice de confusion estimée par validation croisée. En déduire la spécifité et la sensibilité.

```{code-cell} python
print(pd.crosstab(index=res['chd'], columns=res['yhat']))
```


## Choix d'un seuil

Un test physique réalise une sensibilité de 50% et pour cette sensibilité une spécifité de $90\%$. Choisir le seuil pour une sensibilité de 50% (en validation croisée 10 blocs) et donner la spécifité correspondante.

```{code-cell} python
sall = np.linspace(0, 1 , 100)
res = np.zeros((artere.shape[0], len(sall)+1))
res[:,0] = artere.chd.values
res = pd.DataFrame(res)
res.rename(columns={res.columns[0] : "chd"}, inplace=True)
for i,s in enumerate(sall):
    for app_index, val_index in skf.split(X,y):
        artereapp=artere.iloc[app_index,:]
        artereval=artere.iloc[val_index,:]
        modele = smf.logit('chd~age', data=artereapp).fit()
        res.iloc[val_index,i+1] = modele.predict(exog=artereval)>s
```

On calcule pour chaque seuil les matrices de confusion (attention au cas où le modèle ne prévoit que des "0" ou que des "1").

De tous les seuils qui dépassent 0.5 de sensibilité prenons le plus grand:

```{code-cell} python
confusion = pd.DataFrame(np.zeros((len(sall),2)))
for i,s in enumerate(sall):
    tab = pd.crosstab(index=res['chd'], columns=res.iloc[:,i+1])
    if tab.shape[1]==1:
        if tab.columns[0]:
            confusion.iloc[i, 0]=0
            confusion.iloc[i, 1]=1
        else:
            confusion.iloc[i, 0]=1
            confusion.iloc[i, 1]=0
    else:
        confusion.iloc[i, 0] = tab.iloc[0,0]/ tab.iloc[0,:].sum()
        confusion.iloc[i, 1] = tab.iloc[1,1]/ tab.iloc[1,:].sum()

print(sall[confusion.loc[confusion.iloc[:,1]>=0.5].shape[0]])
```

Ce modèle d'age permet ici un sensibilité de 0.5 et presque 0.9 de spécifité.


# Choix de variables


## Importation des données

Importation des données d'ozone `spambase.data` (dans Fun Campus les données sont dans le répertoire `data/`)

```{code-cell} python
spam = pd.read_csv("data/spambase.data", header=None,  sep=",")
spam.rename(columns={spam.columns[57] : "Y"}, inplace=True)
namestr = ["X"+str(i) for i in spam.columns[0:57] ]
spam.rename(columns= dict(zip(list(spam.columns[0:57]), namestr)), inplace=True)
```


## Sélection descendante/backward

Nous reprenons le code de sélection descendante sur la régression et nous remplaçons simplement le mot `ols` par `logit`

La fonction commence avec le modèle complet.

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

La mise en oeuvre

```{code-cell} python
modelefinal = logitbackward(spam,"Y~X0+X1+X2+X3+X4+X5+X6+X7+X8+X9+X10+X11+X12+X13+X14+X15+X16+X17+X18+X19+X20+X21+X22+X23+X24+X25+X26+X27+X28+X29+X30+X31+X32+X33+X34+X35+X36+X37+X38+X39+X40+X41+X42+X43+X44+X45+X46+X47+X48+X49+X50+X51+X52+X53+X54+X55+X56", verbose=True)
```

Le modèle sélectionné

```{code-cell} python
print(modelefinal.summary())
```