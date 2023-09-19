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
  title: 'TP régression Ridge'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa BEDIN<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

# Modules

Importer les modules pandas (comme `pd`) numpy (commme `np`) le sous module `pyplot` de `matplotlib` comme `plt` les fonctions `StandardScaler` de `sklearn.preprocessing`, `Ridge` de `sklearn.linear_model`, `RidgeCV` de `sklearn.linear_model`, `Pipeline` de `sklearn.pipeline`, `cross_val_predict` de `sklearn.model_selection`, `KFold` de `sklearn.model_selection`

```{code-cell} python

```


# Régression ridge sur les données d'ozone


## Importation des données

Importer les données d'ozone `ozonecomplet.csv` (dans Fun Campus les données sont dans `data/`) et éliminer les deux dernières variables (qualitatives) et faites un résumé numérique par variable [méthode `astype` sur la colonne du DataFrame et méthode `describe` sur l'instance DataFrame\]

```{code-cell} python

```


## Création des tableaux `numpy`

avec l'aide des méthodes d'instance `iloc` ou `loc` créer les tableaux `numpy` `y` et `X` (on se servira de l'attribut `values` qui donne le tableau `numpy` sous-jascent)

```{code-cell} python

```


## Centrage et réduction

Centrer et réduire les variable avec `StandardScaler` selon le schéma suivant

1.  créer une instance avec la fonction `StandardScaler`. On notera `scalerX` l'instance créée.
2.  l'ajuster via la méthode d'instance `fit` (calcul des moyennes et écart-types) et avec le tableau `numpy` des $X$
3.  Transformer le tableau $X$ en tableau centré réduit via la méthode d'instance `transform` et avec le tableau `numpy` des $X$.

```{code-cell} python

```


## Calcul de la régression Ridge pour $\lambda=0.00485$

1.  Estimation/ajustement: en utilisant les données centrées réduites pour $X$ et le vecteur `y` estimer le modèle de régression Ridge:
    -   Instancier un modèle `Ridge` avec la fonction éponyme Attention dans `scikitlearn` le paramètre $\lambda$ de la ridge (et lasso et elastic-net) s'appelle $\alpha$.
    -   Estimer le modèle avec $\lambda=0.00485$ et la méthode d'instance `fit`
2.  Afficher $\hat\beta(\lambda)$
3.  Prévoir une valeur en $x^*=(17, 18.4, 5, 5, 7, -4.3301, -4, -3, 87)'$ (on pourra constater qu'il s'agit de la seconde ligne du tableau initial).

```{code-cell} python

```


## Pipeline

On voit bien que si l'on nous donne des valeurs nouvelles il faut enlever la moyenne et diviser par l'écart-type ce qui n'est pas très pratique.

-   Vérifier que `scalerX.transform(X[1,:].reshape(1, 10))` donne bien `Xcr[1,:]`. Cependant l'enchainement «transformation des X» puis «modélisation» peut être automatisé grâce au [Pipeline](https://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html)
-   Créer une instance de pipeline:
    1.  Créer une instance de `StandardScaler`
    2.  Créer une instance de Régression Ridge
    3.  Créer une instance de `Pipeline` avec l'argument `steps` qui sera une liste de tuple dont le premier élément est le nom de l'étape (par exemple `"cr"` ou `"ridge"`) et dont la seconde valeur sera l'instance de l'étape à faire (instances créées aux étapes précédentes.)
-   ajuster cette instance de pipeline avec la méthode d'instance `fit` avec les données `X` et `y`.
-   Retrouver les paramètres $\hat\beta(\lambda)$ en affectant la coordonnée `"ridge"` (nom de l'étape choisi ici) de l'attribut named<sub>steps</sub> dans un objet. Les attributs et méthodes de cet objets seront ensuite les mêmes que ceux la régression `Ridge` après ajustement.
-   Retrouver l'ajustement pour $x^*$

```{code-cell} python

```


## Evolution des coefficients selon $\lambda$


### Calcul d'une grille de $\lambda$

La grille classique pour ridge est constituée sur la même idée que celle pour le lasso:

1.  Calcul de la valeur maximale $\lambda_0 = \arg\max_{i} |[X'y]_i|/n$ Pour le lasso au delà de cette contrainte tous les coefficients sont nuls.
2.  On prend une grille en puissance de 10, avec les exposants qui varient entre 0 et -4 (en général on prend 100 valeurs régulièrement espacées)
3.  Cette grille est multipliée par $\lambda_0$
4.  Pour la régression ridge la grille précédente (qui est celle pour le lasso) est multipliée par $100$ ou $1000$.

On a donc la grille $\{\lambda_0 10^{k+2}\}_{k=0}^{-4}$.

Créer cette grille avec `np.linspace`, méthode d'instance `transpose`, `dot` et `max` (sans oublier l'attribut `shape` pour $n$).

```{code-cell} python

```


### Tracer l'évolution des $\hat\beta(\lambda)$

Tracer en fonction du logarithme des valeurs de $\lambda$ de la grille les coefficients $\hat\beta(\lambda)$

```{code-cell} python

```


## $\hat \lambda$ optimal (par validation croisée 10 blocs/fold)


### Séparation en 10 blocs

Nous allons séparer le jeu de données en 10 blocs grâce à la fonction [KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold): créer une instance de `KFold` nommée `kf`.

```{code-cell} python

```


### Sélection du $\hat \lambda$ optimal

1.  Créer un DataFrame `res` avec 100 colonnes de 0
2.  Faire une boucle sur tous les blocs ; utiliser la méthode d'instance `split` sur `kf` avec les données `X`
3.  Pour chaque «tour» de bloc faire a. estimer sur les 9 blocs en apprentissage les modèles ridge pour chaque $\lambda$ de la grille. b. prévoir les données du bloc en validation c. ranger dans les lignes correspondantes de `res` pour les 100 colonnes correspondantes aux 100 modèles ridge.

```{code-cell} python

```


## Sélection du $\hat \lambda$ optimal

En prenant l'erreur quadratique $\|Y - \hat Y(\lambda)\|^2$ donner le meilleur modèle (et donc le $\hat \lambda$ optimal ) \[méthode d'instance `apply` sur `res` et `argmin` \]

```{code-cell} python

```


## Représentation graphique

Représenter en abscisse les logarithmes des valeurs de $\lambda$ sur la grille et en ordonnée l'erreur quadratique calculée en question précédente.

```{code-cell} python

```


## Modéliser rapidement

1.  Les questions précédentes peuvent être enchainées plus rapidement grâce à \[`cross_val_predict` \] (la grille devra être calculée à la main)
2.  Presque la même chose peut être obtenue avec `RidgeCV` et la perte `'neg_mean_squared_error'` dans l'argument `scoring` (la grille devra être calculée à la main)
3.  Construire un score "somme des erreurs quadratiques par bloc" en utilisant `make_scorer` (voir un des exemples de [scores scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)) et l'utiliser dans `RidgeCV` pour obtenir le résultat du 1. (la grille devra être calculée à la main)

```{code-cell} python

```