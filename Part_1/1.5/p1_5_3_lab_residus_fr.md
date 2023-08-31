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
  title: 'TP résidus'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa Bedin<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

+++

# Modules python
Importer les modules pandas (comme `pd`) numpy (commme `np`)
matplotlib.pyplot (comme  `plt`), statsmodels.formula.api (comme `smf`)
et statsmodels.api (comme `sm`)


```{code-cell} python

```

# Régression multiple (modèle du cours)

### Importation des données
Importer les données d'ozone dans le DataFrame pandas `ozone`
\[`read_csv` de `numpy`\]. Dans Fun Campus les jeux de données sont dans le répertoire `data/`.


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
méthode `summary` pour l'instance/modèle ajusté\]


```{code-cell} python

```

### Résidus \$\varepsilon\$
Afficher le graphique des résidus (attribut `resid` du modèle estimé)
(avec \$\hat y\$ en abscisse et \$\varepsilon\$ en ordonnée).
\[`plot` de plt\]


```{code-cell} python

```

### Résidus \$\varepsilon\$
Afficher le graphique des résidus studentisés par validation croisée (avec \$\hat y\$ en abscisse et 
\$\varepsilon\$ en ordonnée). Pour cela utiliser la fonction/méthode `get_influence` 
qui renverra un objet (que l'on nommera `infl`) avec un attribut `resid_studentized_external`
contenant les résidus souhaités.


```{code-cell} python

```

### Points leviers
Représenter les \$h_{ii}\$ grâce à `plt.stem` en fonction du numéro de ligne
\[`np.arange`, attribut `shape` d'un DataFrame, attribut d'instance 
`hat_matrix_diag` pour `infl`\]


```{code-cell} python

```

# R²
Nous sommes intéressé par batir un modèle de prévision de l'ozone par 
une régression multiple. Cependant nous ne savons pas trop a priori
quelles sont les variables utiles. Batissons plusieurs modèles.

### Estimation du modèle du cours
Ce régression expliquera
le maximum de la concentration en ozone du jour (variable `O3`) par 
- la température à midi notée `T12`
- la nébulosité à midi notée `Ne12`
- la vitesse du vent sur l'axe Est-Ouest notée `Vx`
Traditionnellement on introduit toujours la constante (le faire ici aussi).
Estimer le modèle par MCO et faire le résumé.
\[`ols` de `smf`, méthode `fit` de la classe `OLS` et 
méthode `summary` pour l'instance/modèle ajusté\]


```{code-cell} python

```

### Estimation d'un modèle complémentaire
Ce régression expliquera
le maximum de la concentration en ozone du jour (variable `O3`) par 
- la température à midi notée `T12`
- la température à six heures notée `T15`
- la nébulosité à midi notée `Ne12`
- la vitesse du vent sur l'axe Est-Ouest notée `Vx`
- le maximum du jour d'avant/la veille `O3v`
Traditionnellement on introduit toujours la constante (le faire ici aussi).
Estimer le modèle par MCO et faire le résumé.


```{code-cell} python

```

### Comparer les R2
Comparer les R2 des modèles à 3 et 5 variables 
et expliquer pourquoi cela était attendu.


```{code-cell} python

```

# Résidus partiels (pour aller plus loin)
Cet exercice montre l'utilité pratique des résidus partiels envisagés en TD.
Les données se trouvent dans le fichier `tprespartiel.dta` et
`tpbisrespartiel.dta`, l'objectif de ce TP est de montrer que l'analyse
des résidus partiels peut améliorer la modélisation.

### Importer les données
Vous avez une variable à expliquer \$Y\$
et quatre variables explicatives dans le fichier `tprespartiel.dta`. Dans Fun Campus les jeux de données sont dans le répertoire `data/`.


```{code-cell} python

```

### Estimation
Estimer par MCO les paramètres du modèle \$Y_i=\beta_0 + \beta_1 X_{i,1}+\cdots+
\beta_4 X_{i,4} + \varepsilon_i.\$
[`ols` de `smf`, méthode `fit` de la classe `OLS` et 
méthode `summary` pour l'instance/modèle ajusté]


```{code-cell} python

```

### Analyser les résidus partiels
Que pensez-vous des résultats ?
\[`plot_ccpr_grid` du sous module `sm.graphics`\], les résidus partiels sont
appelés "Component-Component plus Residual"
(CCPR) dans le module statsmodels…


```{code-cell} python

```

### Amélioration du modèle 
Remplacer $X_4$ par $X_5=X_4^2$ dans le modèle précédent. Que pensez-vous de
  la nouvelle modélisation ? On pourra comparer ce modèle à celui de la
  question précédente.
\[`ols` de `smf`, méthode `fit` de la classe `OLS` et 
 attribut d'instance `rsquared`\]
On pourra utiliser les
opérations et fonctions dans les formules
(voir https://www.statsmodels.org/stable/example_formulas.html)


```{code-cell} python

```

### Analyser les résidus partiels
Analyser les résidus partiels du nouveau modèle et constater
qu'ils semblent corrects.
\[`plot_ccpr_grid` du sous module `sm.graphics`\], les résidus partiels sont
appelés "Component-Component plus Residual"
(CCPR) dans le module statsmodels…


```{code-cell} python

```

Faire le même travail pour `tp2bisrespartiel`.


```{code-cell} python

```
