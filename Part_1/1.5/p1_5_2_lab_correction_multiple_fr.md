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
  title: 'Correction de la régression multiple'
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
matplotlib.pyplot (comme  `plt`) et statsmodels.formula.api (comme `smf`). 
Importer aussi `Axes3D` de `mpl_toolkits.mplot3d`.


```{code-cell} python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.formula.api as smf
```

# Régression multiple ozone (2 variables)

## Importation des données
Importer les données d'ozone dans le DataFrame pandas `ozone`


```{code-cell} python
ozone = pd.read_csv("data/ozone.txt", header=0, sep=";")
```

## Représention en 3D
Nous sommes intéressé par batir un modèle de prévision de l'ozone par 
une régression multiple. Cette régression expliquera
le maximum de la concentration en ozone du jour (variable `O3`) par 
- la température à midi notée `T12`
- la vitesse du vent sur l'axe Est-Ouest notée `Vx`
Représentons graphiquement les données avec `O3` sur l'axe z, 
`T12` sur l'axe x et `Vx` sur l'axe y.


```{code-cell} python
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax = Axes3D(fig)
ax.scatter(ozone["T12"], ozone["Vx"],ozone["O3"])
ax.set_xlabel('T12')
ax.set_ylabel('Vx')
ax.set_zlabel('O3')
```

## Modèle de prévision
Écrire le modèle évoqué ci-dessus.

\$y_i = \beta_1 + \beta_2 X_i + \beta_3 Z_i + \varepsilon_i\$
où 
- \$X_i\$ est la \$i^e\$ observation de la variable explicative `T12` et
- \$Z_i\$ est la \$i^e\$ observation de la variable explicative `Vx`
- \$X_i\$ est la \$i^e\$ observation de la variable à expliquer `O3`
- \$\varepsilon_i\$ est la \$i^e\$ coordonnée du vecteur d'erreur
  \$\varepsilon\$
Traditionnellement on introduit toujours comme c'est le cas ici la constante 
(variable associée à \$\beta_1\$).

## Estimation du modèle
Estimer par MCO les paramètres du modèle décrit ci-dessus et faites en le résumé.


```{code-cell} python
reg = smf.ols('O3~T12+Vx', data=ozone).fit()
reg.summary()
```

# Régression multiple ozone (modèle du cours)

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

## Variabilité 
- Trouver parmi les estimateurs des coefficients des effets des variables
(hors constante) celui qui est le plus variable.
- La variabilité est indiquée par
  - la variance du paramètre
  - l'écart-type du paramètre
  - la variance estimée du paramètre
  - l'écart-type estimé du paramètre
- Afficher l'estimation de \$\sigma^2\$

Par lecture du résumé la colonne `std err` donne les
écart-types estimés des coordonnées de \$\hat \beta\$ et le plus grand 
est celui associé à la variable `Ne12`.


```{code-cell} python
reg.scale
```

# Régression multiple eucalytus

## Importation des données
Importer les données d'eucalytus dans le DataFrame pandas `eucalypt`


```{code-cell} python
eucalypt = pd.read_csv("data/eucalyptus.txt", header=0, sep=";")
```

## représentation des données
Représenter le nuage de points


```{code-cell} python
plt.plot(eucalypt["circ"],eucalypt["ht"],'o')
plt.xlabel("circ")
plt.ylabel("ht")
```

## Modèle de prévision
Estimer (par MCO) le modèle linéaire expliquant la hauteur (`ht`) 
par la variable circonférence (`circ`) et la racine carrée de la
circonférence.  On pourra utiliser les
opérations et fonctions dans les formules
(voir https://www.statsmodels.org/stable/example_formulas.html)


```{code-cell} python
regmult = smf.ols("ht ~ circ +  np.sqrt(circ)", data = eucalypt).fit()
regmult.summary()
```

## Répresentation graphique du modèle
Réprésenter sur un graphique les données, la prévision par le modèle ci-dessus et
la prévision par les modèles de régression simple vus dans l'exercice « deux modèles »
dans le TP de régression simple.


```{code-cell} python
reg = smf.ols('ht~circ', data=eucalypt).fit()
regsqrt = smf.ols('ht~I(np.sqrt(circ))', data=eucalypt).fit()
```


```{code-cell} python
sel = eucalypt['circ'].argsort()
xs = eucalypt.circ.iloc[sel]
ys1 = regmult.predict()[sel]
ys2 = reg.predict()[sel]
ys3 = regsqrt.predict()[sel]
plt.plot(eucalypt['circ'], eucalypt['ht'], "o", xs, ys1, "-", xs, ys2, "--", xs, ys3, "-.")
```
