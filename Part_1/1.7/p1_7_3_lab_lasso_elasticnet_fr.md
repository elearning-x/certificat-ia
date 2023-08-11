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
  title: 'TP Lasso et Elastic-Net'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa Bedin<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

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

```

## Régression lasso sur les données d&rsquo;ozone



#### Importation des données



Importer les données d&rsquo;ozone `ozonecomplet.csv` (dans FunStudio les données sont dans `data/`) et éliminer les deux dernières
variables (qualitatives) et faites un résumé numérique par variable [méthode
`astype` sur la colonne du DataFrame et méthode `describe` sur l&rsquo;instance
DataFrame]




```{code-cell} python

```

#### Création des tableaux `numpy`



avec l&rsquo;aide des méthodes d&rsquo;instance `iloc` ou `loc` créer les tableaux `numpy`
`y` et `X` (on se servira de l&rsquo;attribut `values` qui donne le tableau `numpy` sous-jascent)




```{code-cell} python

```

#### Centrage et réduction



Centrer et réduire les variable avec `StandardScaler` selon le schéma
suivant

1.  créer une instance avec la fonction `StandardScaler`. On notera
    `scalerX` l&rsquo;instance créée.
2.  l&rsquo;ajuster via la méthode d&rsquo;instance `fit` (calcul des moyennes et écart-types) et avec le tableau `numpy` des $X$
3.  Transformer le tableau $X$ en tableau centré réduit via la méthode d&rsquo;instance `transform` et avec le tableau `numpy` des $X$.




```{code-cell} python

```

#### Evolution des coefficients selon $\lambda$



La fonction `LassoCV` va donner directement la grille de $\lambda$
(contrairement à ridge). Utiliser cette fonction sur les données centrées
réduites pour récupérer la grille (attribut `alphas_`). Avec cette grille faire
un boucle pour estimer les coefficients $\hat\beta(\lambda)$ pour chaque valeur
de $\lambda$




```{code-cell} python

```

#### Choix du $\hat \lambda$ optimal (par validation croisée 10 blocs/fold)



En séparant le jeu de données en 10 Blocs  grâce
à la fonction [KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold) (l&rsquo;instance de `KFold` sera nommée `kf`)
trouver le $\hat \lambda$ optimal avec un score  &ldquo;somme des erreurs quadratiques par bloc&rdquo; ; utiliser
 `cross_val_predict` (la grille devra être fournie à `Lasso`)




```{code-cell} python

```

#### Retrouver les résultats de la question précédente



Avec la fonction `LassoCV` et l&rsquo;objet `kf` retrouver
le $\hat \lambda$ optimal (par validation croisée 10 blocs/fold)




```{code-cell} python

```

#### Prévision



Utiliser la régression ridge avec $\hat \lambda$ optimal pour prévoir
la concentration d&rsquo;ozone pour
$x^*=(18, 18, 18 ,5 ,5 , 6, 5 ,-4 ,-3, 90)'$




```{code-cell} python

```

## Elastic-Net



refaire avec les mêmes données les questions de l&rsquo;exercice précédent avec une balance entre norme 1 et norme 2 de 1/2 (`l1_ratio`).




```{code-cell} python

```
