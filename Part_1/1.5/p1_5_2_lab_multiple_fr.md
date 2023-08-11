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
  title: 'TP régression multiple'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa Bedin<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

# Modules python
Importer les modules pandas (comme `pd`) numpy (commme `np`)
matplotlib.pyplot (comme  `plt`) et statsmodels.formula.api (comme `smf`). 
Importer aussi `Axes3D` de `mpl_toolkits.mplot3d`.


```{code-cell} python

```

# Régression multiple ozone (2 variables)

### Importation des données
Importer les données d'ozone dans le DataFrame pandas `ozone`
\[`read_csv` de `numpy`\]. Sur FunStudio le chemin est `data/ozone.txt`.


```{code-cell} python

```

### Représention en 3D
Nous sommes intéressé par batir un modèle de prévision de l'ozone par 
une régression multiple. Ce modèle de régression expliquera
le maximum de la concentration en ozone du jour (variable `O3`) par 
- la température à midi notée `T12`
- la vitesse du vent sur l'axe Est-Ouest notée `Vx`
Représentons graphiquement les données avec `O3` sur l'axe z, 
`T12` sur l'axe x et `Vx` sur l'axe y.
\[`figure` et sa méthode `add_subplot` méthode `scatter` de la classe `Axes`\]


```{code-cell} python

```

### Modèle de prévision
Écrire le modèle évoqué ci-dessus.



### Estimation du modèle
Estimer par MCO les paramètres du modèle décrit ci-dessus et faites en le résumé.
\[`ols` de `smf`, méthode `fit` de la classe `OLS` et 
méthode `summary` pour l'instance/modèle ajusté\]


```{code-cell} python

```

# Régression multiple ozone (modèle du cours)

### Importation des données
Importer les données d'ozone dans le DataFrame pandas `ozone`
\[`read_csv` de `numpy`\]. Sur FunStudio le chemin est `data/ozone.txt`.


```{code-cell} python

```

### Estimation du modèle du cours
Nous sommes intéressé par batir un modèle de prévision de l'ozone par 
une régression multiple. Ce régression expliquera
le maximum de la concentration en ozone du jour (variable `O3`) par 
- la température à midi notée `T12`
- la nébulosité à midi notée `Ne12`
- la vitesse du vent sur l'axe Est-Ouest notée `Vx`
Traditionnellement on introduit toujours la constante (le faire ici aussi).
Estimer le modèle par MCO et faire le résumé.
\[`ols` de `smf`, méthode `fit` de la classe `OLS` et 
méthode `summary` pour l'instance\]


```{code-cell} python

```

### Variabilité 
- Trouver parmi les estimateurs des coefficients des effets des variables
(hors constante) celui qui est le plus variable.
- La variabilité est indiquée par
  - la variance du paramètre
  - l'écart-type du paramètre
  - la variance estimée du paramètre
  - l'écart-type estimé du paramètre
- Afficher l'estimation de \$\sigma^2\$
\[attribut `scale` du modèle ajusté/instance\]


```{code-cell} python

```

# Régression multiple eucalytus

### Importation des données
Importer les données d'eucalytus dans le DataFrame pandas `eucalypt`
\[`read_csv` de `numpy`\]. Sur FunStudio le chemin est `data/eucalyptus.txt`.


```{code-cell} python

```

### représentation des données
Représenter le nuage de points
\[`plot` de plt et `xlabel` et `ylabel` de `plt`\]


```{code-cell} python

```

### Modèle de prévision
Estimer (par MCO) le modèle linéaire expliquant la hauteur (`ht`) 
par la variable circonférence (`circ`) et la racine carrée de la
circonférence.  On pourra utiliser les
opérations et fonctions dans les formules
(voir https://www.statsmodels.org/stable/example_formulas.html)


```{code-cell} python

```

### Répresentation graphique du modèle
Réprésenter sur un graphique les données, la prévision par le modèle ci-dessus et
la prévision par les modèles de régression simple vus dans l'exercice « deux modèles »
dans le TP de régression simple.
\[`argsort` méthode d'instance pour les colonnes des DataFrames, 
`plot` de plt et `xlabel` et `ylabel` de `plt`\]


```{code-cell} python

```
