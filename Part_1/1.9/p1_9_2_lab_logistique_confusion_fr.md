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
  title: 'TP régression Logistique: seuil et matrice de confusion'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa BEDIN<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

- [Modules python](#org6d4caa4)
- [Régression logistique (suite TP précédent)](#orga6dadbd)
  - [Importation des données](#org32960d1)
  - [Régression logistique](#org370de89)
  - [Matrice de confusion (ajustée)](#org05b48d7)
  - [Résidus](#org665ac81)
  - [Matrice de confusion (en prévision)](#org85c4403)
    - [Séparation en 10 blocs](#org52f38d1)
    - [DataFrame prevision et $chd$](#org2608d6f)
    - [Calculer la matrice de confusion](#orgd254402)
  - [Choix d'un seuil](#orgf895ea4)
- [Choix de variables](#org31d618e)


<a id="org6d4caa4"></a>

# Modules python

Importer les modules pandas (comme `pd`) numpy (commme `np`) matplotlib.pyplot (comme `plt`) et statsmodels.formula.api (comme `smf`).

```{code-cell} python

```


<a id="orga6dadbd"></a>

# Régression logistique (suite TP précédent)


<a id="org32960d1"></a>

## Importation des données

Importer les données `artere.txt` dans le DataFrame pandas `artere` \[`read_csv` de `numpy` \]. Sur Fun Campus le chemin est `data/artere.txt`. Outre l'age et la présence=1/absence=0 de la maladie cardio-vasculaire (`chd`) une variable qualitative à 8 modalités donne les classes d'age (`agegrp`)

```{code-cell} python

```


<a id="org370de89"></a>

## Régression logistique

Effectuer une régression logistique où `age` est la variable explicative et `chd` la variable binaire à expliquer. Stocker le résultat dans l'objet `modele`

\[`logit` de `smf`, méthode `fit` \]

```{code-cell} python

```


<a id="org05b48d7"></a>

## Matrice de confusion (ajustée)

Afficher la matrice de confusion estimée sur les données de l'échantillon pour un seuil choisi à 0.5.

```{code-cell} python

```

\[méthode de classe `pred_table` sur le modèle et/ou méthode de classe `predict` sur le modèle et `pd.crosstab` sur un DataFrame de 2 colonnes à constituer\]

Attention cette matrice de confusion est ajustée et reste donc très optimiste. Une matrice de confusion calculée par validation croisée ou apprentissage/validation est plus que conseillée !


<a id="org665ac81"></a>

## Résidus

Représenter graphiquement les résidus de déviance avec

1.  en abscisse la variable `age` et en ordonnée les résidus \[attribut `resid_dev` du modèle\];
2.  en abscisse le numéro de ligne du tableau (index) après permutation aléatoire et en ordonnées les résidus.

\[`plt.plot`, méthode `predict` pour l'instance/modèle ajusté et `np.arange` pour générer iles numéros de ligne avec l'attribut `shape` du DataFrame ; créer une instance de générateur aléatoire `np.random.default_rng` et utiliser `rng.permutation` sur les numéros de ligne\]

```{code-cell} python

```


<a id="org85c4403"></a>

## Matrice de confusion (en prévision)

Ayant peu de données ici plutôt que d'évaluer la matrice de confusion en apprentissage/validation nous choisissons (contraints et forcés) d'évaluer celle-ci en validation croisée.


<a id="org52f38d1"></a>

### Séparation en 10 blocs

Nous allons séparer le jeu de données en 10 blocs grâce à la fonction [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold) (du module `sklearn` sous module `model_selection`) créer une instance de `StratifiedKFold` nommée `skf`.

```{code-cell} python

```

Pourquoi utiliser `StratifiedKFold` plutôt que `KFold` ?


<a id="org2608d6f"></a>

### DataFrame prevision et $chd$

Créer un DataFrame `res` avec deux colonnes: la variable $chd$ et une second colonne remplie de 0 qui contiendra les prévisions. Cette colonne pourra être nommée `yhat`.

```{code-cell} python

```

Ajouter au fur et à mesure les prévisions en validation croisée dans la deuxième colonne: Pour chaque «tour» de bloc faire

1.  estimer sur les 9 blocs en apprentissage le modèle de régression logistique
2.  prévoir les données du bloc en validation (seuil $s=1/2$)
3.  ranger dans les lignes correspondantes de `res` les prévisions (dans la colonne `yhat`)

```{code-cell} python

```


<a id="orgd254402"></a>

### Calculer la matrice de confusion

Avec la fonction `crosstab` du module `pd` proposer la matrice de confusion estimée par validation croisée. En déduire la spécifité et la sensibilité.


<a id="orgf895ea4"></a>

## Choix d'un seuil

Un test physique réalise une sensibilité de 50% et pour cette sensibilité une spécifité de $90\%$. Choisir le seuil pour une sensibilité de 50% (en validation croisée 10 blocs) et donner la spécifité correspondante. On pourra prendre une grille de seuils régulièrement espacés entre 0 et 1 de 100 valeurs.

```{code-cell} python

```


<a id="org31d618e"></a>

# Choix de variables

Le jeu de données que nous voulons traiter est `spambase.data`. Ce fichier contient 4601 observations et 58 variables ont été mesurées. Il correspond à l'analyse d'emails.

La dernière colonne de `spambase.data` indique si l'e-mail a été considéré comme du spam (1) ou non (0), c'est-à-dire comme un e-mail commercial non sollicité. C'est la variable à expliquer. La plupart des attributs indiquent si un mot ou un caractère particulier apparaît fréquemment dans l'e-mail. Les attributs de longueur (55-57) mesurent la longueur des séquences de lettres majuscules consécutives. Voici les définitions des attributs (voir aussi le site de l'[UCI](http://archive.ics.uci.edu/dataset/94/spambase)) :

-   48 attributs réels continus qui représentent le pourcentage de mots de l'e-mail qui correspondent au MOT, c'est-à-dire 100 \* (nombre de fois où le MOT apparaît dans l'e-mail) / nombre total de mots dans l'e-mail.
-   6 attributs réels continus qui représentent le pourcentage de caractères de l'e-mail qui correspondent à CHAR, c'est-à-dire 100 \* (nombre d'occurrences de CHAR) / nombre total de caractères dans l'e-mail.
-   1 continu réel qui représente la longueur moyenne des séquences ininterrompues de lettres majuscules
-   1 entier qui représente la longueur de la plus longue séquence ininterrompue de lettres capitales
-   1 entier qui représente la longueur des séquences ininterrompues de lettres capitales ou du nombre total de majuscules dans l'e-mail
-   1 nominal {0,1} attribut de classe de type spam ie indique si l'e-mail est considéré comme du spam (1) ou non (0), c'est-à-dire un courriel commercial non sollicité.

En se basant sur le TP de choix de variable en régression proposer une procédure de choix de modèle basée sur le BIC.