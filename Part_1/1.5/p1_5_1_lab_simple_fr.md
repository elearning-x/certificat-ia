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
  title: 'TP régression simple'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa Bedin &amp;<br />Pierre André CORNILLON &amp;<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

# Modules python
Importer les modules pandas (comme `pd`) numpy (commme `np`)
matplotlib.pyplot (comme  `plt`) et statsmodels.formula.api (comme `smf`). 


```{code-cell} python

```

# Régression simple

### Importation des données
Importer les données d'eucalytus dans le DataFrame pandas `eucalypt`
\[`read_csv` de `numpy`\]


```{code-cell} python

```

### Nuage de points
Tracer le nuage de points avec `circ` en  abscisses et `ht` en ordonnées
\[`plt.plot`\]


```{code-cell} python

```

### Régression simple
Effectuer une régression linéaire simple où `circ` est  la variable
explicative et `ht` la variable à expliquer. Stocker le résultat
dans l'objet `reg` et 
1. effectuer le résumé de cette modélisation;
2. afficher l'attribut contenant les paramètres estimés par MCO de la droite;
3. afficher l'attribut contenant l'estimation de l'écart-type de l'erreur.
\[`ols` de `smf`, méthode `fit` de la classe `OLS`, 
méthode `summary` pour l'instance/modèle ajusté,
attributs `params` et `scale`.\]


```{code-cell} python

```

### Résidus
Représenter graphiquement les résidus avec
1. en abscisse la variable `circ` et en ordonnée les résidus;
2. en abscisse l'ajustement \$\hat y\$ et en ordonnée les résidus;
3. en abscisse le numéro de ligne du tableau (index) et en ordonnées les résidus.
\[`plt.plot`, méthode `predict` pour l'instance/modèle ajusté et
`np.arange` pour générer les numéros de ligne avec l'attribut `shape`
du DataFrame\]


```{code-cell} python

```

# Variabilité de l'estimation

### Importation des données
Importer les données d'eucalytus dans le DataFrame pandas `eucalypt`
\[`read_csv` de `numpy`\]


```{code-cell} python

```

Créer deux listes vides `beta1` et `beta2`
Faire 500 fois les étapes suivantes
1. Tirer au hasard sans remise 100 lignes dans le tableau `eucalypt`
2. Sur ce tirage effectuer une régression linéaire simple
   où `circ` est la variable explicative et `ht` la variable 
   à expliquer. Stocker les paramètres estimés dans `beta1` et `beta2`
\[créer une instance de générateur aléatoire `np.random.default_rng`\]


```{code-cell} python

```

### Variabilité de \$\hat \beta_2\$
Représenter la variabilité de la variable aléatoire  \$\hat \beta_2\$.
\[une fonction de `plt`...\]


```{code-cell} python

```

Tracer les couples \$\hat \beta_1\$ et \$\hat \beta_2\$ et
constater la variabilité de l'estimation et la corrélation
entre les deux paramètres.
\[une fonction de `plt`...\]


```{code-cell} python

```

# Deux modèles

### Importation des données
Importer les données d'eucalytus dans le DataFrame pandas `eucalypt`
\[`read_csv` de `numpy`\]


```{code-cell} python

```

### Nuage de points
Tracer le nuage de points avec `circ` en  abscisses et `ht` en ordonnées
et constater que les points ne sont pas exactement autour
d'une droite mais plutôt une courbe qui est de type "racine carrée"
\[`plt.plot`\]


```{code-cell} python

```

### Deux régressions simples
1. Effectuer une régression linéaire simple où `circ` est
   la variable explicative et `ht` la variable à expliquer.
   Stocker le résultat dans l'objet `reg`
2. Effectuer une régression linéaire simple où la racine carrée de `circ`
   est  la variable explicative et `ht` la variable à expliquer.
   Stocker le résultat dans l'objet `regsqrt`. On pourra utiliser les
   opérations et fonctions dans les formules
   (voir https://www.statsmodels.org/stable/example_formulas.html)
\[`ols` de `smf`, méthode `fit` de la classe `OLS`, 
méthode `summary` pour l'instance/modèle ajusté\]


```{code-cell} python

```

### Comparaison
Ajouter au nuage de points les 2 ajustements (la droite et la "racine carrée")
et choisir le meilleur modèle.
\[méthode `argsort` sur une colonne du DataFrame et `plt.plot`\]


```{code-cell} python

```
