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
  title: 'TP régression Logistique'
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

```


# Régression logistique


## Importation des données

Importer les données `artere.txt` dans le DataFrame pandas `artere` \[`read_csv` de `numpy` \]. Sur Fun Campus le chemin est `data/artere.txt`. Outre l'age et la présence=1/absence=0 de la maladie cardio-vasculaire (`chd`) une variable qualitative à 8 modalités donne les classes d'age (`agegrp`)

```{code-cell} python

```


## Nuage de points

Tracer le nuage de points avec `age` en abscisses et `chd` en ordonnées \[`plt.plot` \]

```{code-cell} python

```


## Régression logistique

Effectuer une régression logistique où `age` est la variable explicative et `chd` la variable binaire à expliquer. Stocker le résultat dans l'objet `modele` et

1.  effectuer le résumé de cette modélisation;
2.  afficher l'attribut contenant les paramètres estimés par régression logistique;

\[`logit` de `smf`, méthode `fit`, méthode `summary` pour l'instance/modèle ajusté, attributs `params`.\]

```{code-cell} python

```


## Prévision et probabilités estimées

Afficher l'ajustement/prévision pour les données de l'échantillon via la méthode `predict` (sans arguments) sur le modèle `reg`. Que représente ce vecteur:

-   une probabilité d'être malade pour chaque valeur de l'age de l'échantillon
-   une probabilité d'être non-malade pour chaque valeur de l'age de l'échantillon
-   une prévision de l'état malade/non-malade pour chaque valeur de l'age de l'échantillon.

```{code-cell} python

```

Donner la prévision de l'état malade/non-malade avec l'indicatrice que $\hat p(x)>s$ où $s$ est le seuil classique de 0.5.

```{code-cell} python

```


# Simulation de données  Variabilité de $\hat \beta_2$


## Simulation

1.  Générer $n=100$ valeurs de $X$ uniformément entre 0 et 1.
2.  Pour chaque valeur $X_i$ simuler $Y_i$ selon un modèle logistique de paramètres $\beta_1=-5$ et $\beta_2=10$

\[créer une instance de générateur aléatoire `np.random.default_rng` et utiliser `rng.uniform` et `rng.binomial` \]

```{code-cell} python

```


## Estimation

Estimer les paramètres $\beta_1$ et $\beta_2$

```{code-cell} python

```


## Variabilité de l'estimation

Refaire les deux questions ci-dessus 500 fois et constater par un graphique adapté la variabilité de $\hat \beta_2$.

```{code-cell} python

```


# Deux régressions logistiques simples


## Importation des données

Importer les données `artere.txt` dans le DataFrame pandas `artere` \[`read_csv` de `numpy` \]. Sur Fun Campus le chemin est `data/artere.txt`. Outre l'age et la présence=1/absence=0 de la maladie cardio-vasculaire (`chd`) une variable qualitative à 8 modalités donne les classes d'age (`agegrp`)

```{code-cell} python

```


## Deux régressions logistiques

1.  Effectuer une régression logistique simple où `age` est la variable explicative et `chd` la variable binaire à expliquer;
2.  Refaire la même chose avec la racine carrée de `age` comme variable explicative;

```{code-cell} python

```


## Comparaison

Ajouter au nuage de points les 2 ajustements (la droite et la "racine carrée") et choisir le meilleur modèle via un critère numérique. \[méthode `argsort` sur une colonne du DataFrame et `plt.plot` ; utiliser le résumé des modèles\]

```{code-cell} python

```