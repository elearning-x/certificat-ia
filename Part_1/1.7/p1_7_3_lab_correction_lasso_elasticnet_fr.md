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
  title: 'Correction du TP Lasso et Elastic-Net'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa BEDIN<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

+++

Correction_lasso_elasticnet
===========================



## Modules



Importer les modules pandas (comme `pd`) numpy (commme `np`)
le sous module `pyplot` de `matplotlib` comme `plt`
les fonctions `StandardScaler` de `sklearn.preprocessing`,
`Lasso` de  `sklearn.linear_model`,
`LassoCV` de  `sklearn.linear_model`,
`ElasticNet` de  `sklearn.linear_model`,
`ElasticNetCV` de  `sklearn.linear_model`,
`cross_val_predict` de `sklearn.model_selection`,
`KFold` de `sklearn.model_selection`


```{code-cell} python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
```

## Régression lasso sur les données d&rsquo;ozone



#### Importation des données



Importer les données d&rsquo;ozone `ozonecomplet.csv` et éliminer les deux dernières
variables (qualitatives) et faites un résumé numérique par variable [méthode
`astype` sur la colonne du DataFrame et méthode `describe` sur l&rsquo;instance
DataFrame]




```{code-cell} python
ozone = pd.read_csv("data/ozonecomplet.csv", header=0, sep=";")
ozone = ozone.drop(['nomligne', 'Ne', 'Dv'], axis=1)
ozone.describe()
```

#### Création des tableaux `numpy`



avec l&rsquo;aide des méthodes d&rsquo;instance `iloc` ou `loc` créer les tableaux `numpy`
`y` et `X` (on se servira de l&rsquo;attribut `values` qui donne le tableau `numpy` sous-jascent)




```{code-cell} python
y = ozone.O3.values
X = ozone.iloc[:,1:].values
```

#### Centrage et réduction



Centrer et réduire les variable avec `StandardScaler` selon le schéma
suivant

1.  créer une instance avec la fonction `StandardScaler`. On notera
    `scalerX` l&rsquo;instance créée.
2.  l&rsquo;ajuster via la méthode d&rsquo;instance `fit` (calcul des moyennes et écart-types) et avec le tableau `numpy` des $X$
3.  Transformer le tableau $X$ en tableau centré réduit via la méthode d&rsquo;instance `transform` et avec le tableau `numpy` des $X$.




```{code-cell} python
scalerX = StandardScaler().fit(X)
Xcr= scalerX.transform(X)
```

#### Evolution des coefficients selon $\lambda$



La fonction `LassoCV` va donner directement la grille de $\lambda$
(contrairement à ridge). Utiliser cette fonction sur les données centrées
réduites pour récupérer la grille (attribut `alphas_`). Avec cette grille faire
un boucle pour estimer les coefficients $\hat\beta(\lambda)$ pour chaque valeur
de $\lambda$

Ajustons le modèle pour chaque valeur de $\lambda$:




```{code-cell} python
rl = LassoCV().fit(Xcr,y)
alphas_lasso = rl.alphas_
lcoef = []
for ll in alphas_lasso:
    rl = Lasso(alpha=ll).fit(Xcr,y)
    lcoef.append(rl.coef_)
```

et traçons les coefficients:




```{code-cell} python
plt.plot(np.log(alphas_lasso), lcoef)
plt.show()
```

On voit que pour une certaine valeur de $\lambda$ (ici 22) tous les
coefficients sont nuls.



#### Choix du $\hat \lambda$ optimal (par validation croisée 10 blocs/fold)



En séparant le jeu de données en 10 Blocs  grâce
à la fonction [KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold) (l&rsquo;instance de `KFold` sera nommée `kf`)
trouver le $\hat \lambda$ optimal avec un score  &ldquo;somme des erreurs quadratiques par bloc&rdquo; ; utiliser
 `cross_val_predict` (la grille devra être fournie à `Lasso`)




```{code-cell} python
kf = KFold(n_splits=10, shuffle=True, random_state=0)
res = pd.DataFrame(np.zeros((X.shape[0], len(alphas_lasso))))
for j, ll in enumerate(alphas_lasso):
    res.iloc[:,j] = cross_val_predict(Lasso(alpha=ll),Xcr,y,cv=kf)
sse = res.apply(lambda x: ((x-y)**2).sum(), axis=0)
print(alphas_lasso[sse.argmin()])
```

    0.7727174033372736

#### Retrouver les résultats de la question précédente



Avec la fonction `LassoCV` et l&rsquo;objet `kf` retrouver
le $\hat \lambda$ optimal (par validation croisée 10 blocs/fold)




```{code-cell} python
rl = LassoCV(cv=kf).fit(Xcr, y)
print(rl.alpha_)
```

    0.7727174033372736

Ici la fonction objectif est le $\mathrm{R}^2$ par bloc (et pas la somme des écarts quadratiques) et on retrouve le même $\hat \lambda$
(ce qui n&rsquo;est pas garanti dans tous les cas…)



#### Prévision



Utiliser la régression ridge avec $\hat \lambda$ optimal pour prévoir
la concentration d&rsquo;ozone pour
$x^*=(18, 18, 18 ,5 ,5 , 6, 5 ,-4 ,-3, 90)'$




```{code-cell} python
xet = np.array([[18, 18, 18 ,5 ,5 , 6, 5 ,-4 ,-3, 90]])
xetcr = scalerX.transform(xet)
print(rl.predict(xetcr))
```

    [85.28390512]

## Elastic-Net



refaire avec les mêmes données les questions de l&rsquo;exercice précédent avec une balance entre norme 1 et norme 2 de 1/2 (`l1_ratio`).



#### Importation




```{code-cell} python
ozone = pd.read_csv("data/ozonecomplet.csv", header=0, sep=";")
ozone = ozone.drop(['nomligne', 'Ne', 'Dv'], axis=1)
ozone.describe()
```

#### Création des tableaux `numpy`



avec l&rsquo;aide des méthodes d&rsquo;instance `iloc` ou `loc` créer les tableaux `numpy`
`y` et `X` (on se servira de l&rsquo;attribut `values` qui donne le tableau `numpy` sous-jascent)




```{code-cell} python
y = ozone.O3.values
X = ozone.iloc[:,1:].values
```

#### Centrage et réduction




```{code-cell} python
scalerX = StandardScaler().fit(X)
Xcr= scalerX.transform(X)
```

#### Evolution des coefficients selon $\lambda$



Ajustons le modèle pour chaque valeur de $\lambda$:




```{code-cell} python
ren = ElasticNetCV().fit(Xcr,y)
alphas_elasticnet = ren.alphas_
lcoef = []
for ll in alphas_elasticnet:
    ren = ElasticNet(alpha=ll).fit(Xcr,y)
    lcoef.append(ren.coef_)
```

et traçons les coefficients:




```{code-cell} python
plt.plot(np.log(alphas_elasticnet), lcoef)
plt.show()
```

On voit que les coefficients en général décroissent (en valeur absolue)
quand la pénalité augmente.



#### Choix du $\hat \lambda$ optimal (par validation croisée 10 blocs/fold)




```{code-cell} python
kf = KFold(n_splits=10, shuffle=True, random_state=0)
res = pd.DataFrame(np.zeros((X.shape[0], len(alphas_elasticnet))))
for j, ll in enumerate(alphas_elasticnet):
    res.iloc[:,j] = cross_val_predict(ElasticNet(alpha=ll),Xcr,y,cv=kf)
sse = res.apply(lambda x: ((x-y)**2).sum(), axis=0)
print(alphas_elasticnet[sse.argmin()])
```

    0.41048105093488396

#### Retrouver les résultats de la question précédente



Avec la fonction `ElasticNetCV` et l&rsquo;objet `kf` retrouver
le $\hat \lambda$ optimal (par validation croisée 10 blocs/fold)




```{code-cell} python
ren = ElasticNetCV(cv=kf).fit(Xcr, y)
print(ren.alpha_)
```

    0.41048105093488396

Ici la fonction objectif est le $\mathrm{R}^2$ par bloc (et pas la somme des écarts quadratiques) et on retrouve le même $\hat \lambda$
(ce qui n&rsquo;est pas garanti dans tous les cas…)



#### Prévision



Utiliser la régression ridge avec $\hat \lambda$ optimal pour prévoir
la concentration d&rsquo;ozone pour
$x^*=(18, 18, 18 ,5 ,5 , 6, 5 ,-4 ,-3, 90)'$




```{code-cell} python
xet = np.array([[18, 18, 18 ,5 ,5 , 6, 5 ,-4 ,-3, 90]])
xetcr = scalerX.transform(xet)
print(ren.predict(xetcr))
```

    [87.15292087]

Pas le même modèle ici donc pas la même prévision.


