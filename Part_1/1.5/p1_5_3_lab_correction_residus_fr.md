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
  title: 'Correction du TP résidus'
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
Importer les modules pandas (comme `pd`) numpy (commme `np`)
matplotlib.pyplot (comme  `plt`), statsmodels.formula.api (comme `smf`)
et statsmodels.api (comme `sm`)


```{code-cell} python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
```

# Régression multiple (modèle du cours)

## Importation des données
Importer les données d'ozone dans le DataFrame pandas `ozone`


```{code-cell} python
ozone = pd.read_csv("data/ozone.txt", header=0, sep=";")
```

## Estimation du modèle du cours
Nous sommes intéressé par batir un modèle de prévision de l'ozone par 
une régression multiple. Ce régression expliquera
le maximum de la concentration en ozone du jour (variable `O3`) par 
- la température à midi notée `T12`
- la nébulosité à midi notée `Ne12`
- la vitesse du vent sur l'axe Est-Ouest notée `Vx`
Traditionnellement on introduit toujours la constante (le faire ici aussi).
Estimer le modèle par MCO et faire le résumé.


```{code-cell} python
reg = smf.ols('O3~T12+Ne12+Vx', data=ozone).fit()
reg.summary()
```

## Résidus \$\varepsilon\$
Afficher le graphique des résidus (attribut `resid` du modèle estimé)
(avec \$\hat y\$ en abscisse et \$\varepsilon\$ en ordonnée).


```{code-cell} python
plt.plot(reg.predict(), reg.resid ,"+")
```

Aucune structuration des résidus visible. L'épaisseur (écart-type) des points semble 
un peu toujours la même mais ces résidus par construction n'ont pas la même variance, 
donc il est délicat de conclure sur l'hypothèse \$\mathrm{V}(\varepsilon_i)=\sigma^2\$.
De plus l'échelle des ordonnées dépend du problème, donc ces résidus sont peu 
praticables.

## Résidus \$\varepsilon\$
Afficher le graphique des résidus studentisés par validation croisée (avec \$\hat y\$ en abscisse et 
\$\varepsilon\$ en ordonnée). Pour cela utiliser la fonction/méthode `get_influence` 
qui renverra un objet (que l'on nommera `infl`) avec un attribut `resid_studentized_external`
contenant les résidus souhaités.


```{code-cell} python
infl = reg.get_influence()
plt.plot(reg.predict(), infl.resid_studentized_external,"+")
```

Aucune structuration des résidus visible. L'épaisseur (écart-type) des points semble 
un peu toujours la même donc l'hypothèse \$\mathrm{V}(\varepsilon_i)=\sigma^2\$ semble
correcte. Aucun point en dehors de -2,2 donc pas d'individus aberrant.

## Points leviers
Représenter les \$h_{ii}\$ grâce à `plt.stem` en fonction du numéro de ligne


```{code-cell} python
index=np.arange(1, ozone.shape[0]+1)
plt.stem(index, infl.hat_matrix_diag)
```

Aucun  \$h_{ii}\$ notablement plus grand 
que les autres donc le plan d'expérience est correct.

# R²
Nous sommes intéressé par batir un modèle de prévision de l'ozone par 
une régression multiple. Cependant nous ne savons pas trop a priori
quelles sont les variables utiles. Batissons plusieurs modèles.

## Estimation du modèle du cours
Ce régression expliquera
le maximum de la concentration en ozone du jour (variable `O3`) par 
- la température à midi notée `T12`
- la nébulosité à midi notée `Ne12`
- la vitesse du vent sur l'axe Est-Ouest notée `Vx`
Traditionnellement on introduit toujours la constante (le faire ici aussi).
Estimer le modèle par MCO et faire le résumé.


```{code-cell} python
reg3 = smf.ols('O3~T12+Ne12+Vx', data=ozone).fit()
reg3.summary()
```

## Estimation du modèle du cours
Ce régression expliquera
le maximum de la concentration en ozone du jour (variable `O3`) par 
- la température à six heures notée `T12`
- la température à midi notée `T15`
- la nébulosité à midi notée `Ne12`
- la vitesse du vent sur l'axe Est-Ouest notée `Vx`
- le maximum du jour d'avant/la veille `O3v`
Traditionnellement on introduit toujours la constante (le faire ici aussi).
Estimer le modèle par MCO et faire le résumé.


```{code-cell} python
ozone.head()
```


```{code-cell} python
reg6 = smf.ols('O3~T12+T15+Ne12+Vx+O3v', data=ozone).fit()
reg6.summary()
```

## Comparer les R2
Comparer les R2 des modèles à 3 et 5 variables 
et expliquer pourquoi cela était attendu.

Le R2 augmente avec le nombre de variables ajoutées. Le modèle à 6 variables 
consiste à ajouter les variables `T15` et `O3v` au modèle à 3 variables
et donc il est normal qu'il augmente. Il ne peut pas servir à comparer
des ajustements pour des modèles ayant des nombres de variables différents.

# Résidus partiels (pour aller plus loin)
Cet exercice montre l'utilité pratique des résidus partiels envisagés en TD.
Les données se trouvent dans le fichier `tprespartiel.dta` et
`tpbisrespartiel.dta`, l'objectif de ce TP est de montrer que l'analyse
des résidus partiels peut améliorer la modélisation.

## Importer les données
Vous avez une variable à expliquer \$Y\$
et quatre variables explicatives dans le fichier `tprespartiel.dta`


```{code-cell} python
tp = pd.read_csv("tprespartiel.dta", header=0, sep=";")
tp.head()
```

## Estimation
Estimer par MCO les paramètres du modèle \$Y_i=\beta_0 + \beta_1 X_{i,1}+\cdots+
\beta_4 X_{i,4} + \varepsilon_i.\$
[`ols` de `smf`, méthode `fit` de la classe `OLS` et 
méthode `summary` pour l'instance/modèle ajusté]


```{code-cell} python
reg = smf.ols("Y~X1+X2+X3+X4", data=tp).fit()
```

## Analyser les résidus partiels
Que pensez-vous des résultats ?
\[`plot_ccpr_grid` du sous module `sm.graphics`\], les résidus partiels sont
appelés "Component-Component plus Residual"
(CCPR) dans le module statsmodels…


```{code-cell} python
sm.graphics.plot_ccpr_grid(reg)
```

De manière évidente le graphique pour la variable `X4` ne montre pas
des points disposés le long d'une droite ou un nuage sans structure. 
Il montre une structuration de type \$x\mapsto x^2\$

## Amélioration du modèle 
Remplacer $X_4$ par $X_5=X_4^2$ dans le modèle précédent. Que pensez-vous de
  la nouvelle modélisation ? On pourra comparer ce modèle à celui de la
  question précédente.
\[`ols` de `smf`, méthode `fit` de la classe `OLS` et 
méthode `summary` pour l'instance/modèle ajusté\]
On pourra utiliser les
opérations et fonctions dans les formules
(voir https://www.statsmodels.org/stable/example_formulas.html)


```{code-cell} python
reg2 = smf.ols("Y~X1+X2+X3+I(X4**2)", data=tp).fit()
```

## Analyser les résidus partiels
Analyser les résidus partiels du nouveau modèle et constater
qu'ils semblent corrects.
\[`plot_ccpr_grid` du sous module `sm.graphics`\], les résidus partiels sont
appelés "Component-Component plus Residual"
(CCPR) dans le module statsmodels…


```{code-cell} python
sm.graphics.plot_ccpr_grid(reg2)
```

The graphs show points with no obvious structure
or arranged along straight lines. The model would appear to be correct. We can compare 
compare them (same number of variables) by R2


```{code-cell} python
reg.rsquared, reg2.rsquared
```


```{code-cell} python

```

et le R2 de la seconde modélisation apparait meilleur.

Faire le même travail pour `tp2bisrespartiel`.


```{code-cell} python
tp = pd.read_csv("tprespartiel.dta", header=0, sep=";")
tp.head()
reg = smf.ols("Y~X1+X2+X3+X4", data=tp).fit()
sm.graphics.plot_ccpr_grid(reg)
```

Nous voyons clairement une sinusoïde de type \$\sin(-2\pi X_4)\$ 
sur le dernier graphique. Changeons \$X_4\$


```{code-cell} python
reg2 = smf.ols("Y~X1+X2+X3+I(np.sin(-2*np.pi*X4))", data=tp).fit()
sm.graphics.plot_ccpr_grid(reg2)
```

Là encore les graphiques deviennent corrects et nous pouvons comparer 
les R2 et constater que la seconde modélisation améliore le R2.


```{code-cell} python
reg.rsqared
```


```{code-cell} python
reg2.rsquared
```
