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
  title: 'Correction du TP régression simple'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa Bedin<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

# Modules python
Importer les modules pandas (comme `pd`) numpy (commme `np`)
matplotlib.pyplot (comme  `plt`) et statsmodels.formula.api (comme `smf`)


```{code-cell} python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
```

# Régression simple

## Importation des données
Importer les données d'eucalytus dans le DataFrame pandas `eucalypt`


```{code-cell} python
eucalypt = pd.read_csv("eucalyptus.txt", header=0, sep=";")
```

## Nuage de points
Tracer le nuage de points avec `circ` en  abscisses et `ht` en ordonnées


```{code-cell} python
plt.plot(eucalypt['circ'], eucalypt['ht'], "o")
```

On observe que les points sont grossièrement autour d'une droite,
nous pouvons donc proposer une régression linéaire.
Pour celles et ceux qui pense que c'est trop courbé pour être une droite,
l'exercice « deux modèles » permet de voir comment y remédier simplement.

## Régression simple
Effectuer une régression linéaire simple où `circ` est  la variable
explicative et `ht` la variable à expliquer. Stocker le résultat
dans l'objet `reg` et 
1. effectuer le résumé de cette modélisation;
2. afficher l'attribut contenant les paramètres estimés par MCO de la droite;
3. afficher l'attribut contenant l'estimation de l'écart-type de l'erreur.


```{code-cell} python
reg = smf.ols('ht~circ', data=eucalypt).fit()
reg.summary()
```


```{code-cell} python
reg.params
```


```{code-cell} python
reg.scale
```

## Résidus
Représenter graphiquement les résidus avec
1. en abscisse la variable `circ` et en ordonnée les résidus;
2. en abscisse l'ajustement \$\hat y\$ et en ordonnée les résidus;
3. en abscisse le numéro de ligne du tableau (index) et en ordonnées les résidus.


```{code-cell} python
plt.plot(eucalypt['circ'], reg.resid, "o")
```

Les erreurs d'après le modèle sont indépendantes et ont toutes
la même espérance de 0 et la même variance \$\sigma^2\$.
Les résidus sont une prédiction des erreurs et devrait avoir
les même propriétés. La variance ici n'est pas la même
(épaisseur de la bande de points plus fine par endroits)
et la moyenne semble fluctuer. Cependant, ces résidus
sont plutôt satisfaisant.


```{code-cell} python
plt.plot(reg.predict(), reg.resid, "o")
```

Comme nous interprétons uniquement la forme/l'aspect visuel 
et que seule l'échelle des abscisses a changée et nous obtenons donc
la même interprétation qu'au graphique précédent.


```{code-cell} python
plt.plot(np.arange(1,eucalypt.shape[0]+1), reg.resid , "o")
```

On retrouve les fluctuations de la moyenne des résidus, mais ce
graphique est moins adapté pour la variance dans ce problème.

# Variabilité de l'estimation

## Importation des données
Importer les données d'eucalytus dans le DataFrame pandas `eucalypt`


```{code-cell} python
eucalypt = pd.read_csv("eucalyptus.txt", header=0, sep=";")
```

## estimation sur \$n=100\$ données
Créer deux listes vides `beta1` et `beta2`
Faire 500 fois les étapes suivantes
1. Tirer au hasard sans remise 100 lignes dans le tableau `eucalypt`
2. Sur ce tirage effectuer une régression linéaire simple
   où `circ` est la variable explicative et `ht` la variable à expliquer. Stocker les paramètres estimés dans `beta1` et `beta2`


```{code-cell} python
beta1 = []
beta2 = []
rng = np.random.default_rng(seed=123) # fixe la graine du générateur, les tirages seront les mêmes
for k in range(500):
    lines = rng.choice(eucalypt.shape[0], size=10, replace=False)
    euca100 = eucalypt.iloc[lines]
    reg100 = smf.ols('ht~circ', data=euca100).fit()
    beta1.append(reg100.params[0])
    beta2.append(reg100.params[1])
    
```

## Variabilité de \$\hat \beta_2\$
Représenter la variabilité de la variable aléatoire  \$\hat \beta_2\$.


```{code-cell} python
plt.hist(beta2, bins=30)
```

Cet histogramme est plutôt symétrique, une valeurs aberrante autour de 0.5 ; Mise à part cette valeur, il ressemble à celui que l'ont obtiendrait avec des tirages d'une loi normale.

## Dépendance de \$\hat \beta_1\$ et \$\hat \beta_2\$
Tracer les couples \$\hat \beta_1\$ et \$\hat \beta_2\$ et
constater la variabilité de l'estimation et la corrélation
entre les deux paramètres.


```{code-cell} python
plt.plot(beta1, beta2, "o")
```

On constate ici la très forte corrélation (le nuage est le long d'une droite)
négative (la pente est négative). Ce résultat illustre les résultats
théorique de la corrélation négative entre les deux estimateurs.
Nous retrouvons notre valeur aberrante indiquant qu'un échantillon
tiré est différents des autres. Parmi tous les arbres du champ
certains (un nombre non négligeable) sont différents, ce qui est
normal dans ce type d'essai. 

# Deux modèles

## Importation des données
Importer les données d'eucalytus dans le DataFrame pandas `eucalypt`


```{code-cell} python
eucalypt = pd.read_csv("eucalyptus.txt", header=0, sep=";")
```

## Nuage de points
Tracer le nuage de points avec `circ` en  abscisses et `ht` en ordonnées
et constater que les points ne sont pas exactement autour
d'une droite mais plutôt une courbe qui est de type "racine carrée"


```{code-cell} python
plt.plot(eucalypt['circ'], eucalypt['ht'], "o")
```

## Deux régressions simples
1. Effectuer une régression linéaire simple où `circ` est
   la variable explicative et `ht` la variable à expliquer.
   Stocker le résultat dans l'objet `reg`
2. Effectuer une régression linéaire simple où la racine carrée de `circ`
   est  la variable explicative et `ht` la variable à expliquer.
   Stocker le résultat dans l'objet `regsqrt`. On pourra utiliser les
   opérations et fonctions dans les formules
   (voir https://www.statsmodels.org/stable/example_formulas.html)


```{code-cell} python
reg = smf.ols('ht~circ', data=eucalypt).fit()
regsqrt = smf.ols('ht~I(np.sqrt(circ))', data=eucalypt).fit()
```

## Comparaison
Ajouter au nuage de points les 2 ajustements (la droite et la "racine carrée")
et choisir le meilleur modèle.


```{code-cell} python
sel = eucalypt['circ'].argsort()
plt.plot(eucalypt['circ'], eucalypt['ht'], "o", eucalypt['circ'], reg.predict(), "-", eucalypt.circ.iloc[sel], regsqrt.predict()[sel], "-"  )
```

Graphiquement le modèle « racine carré » semble mieux passer dans les points.
Ces deux modèles pourront être comparé via le R²


```{code-cell} python
reg.rsquared
```


```{code-cell} python
regsqrt.rsquared
```

Le R² le plus élevé permet de sélectionner la meilleure régression simple,
ce qui indique ici que le modèle « racine carré » est meilleur ;
il s'agit d'un résumé numérique de notre constatation graphique…
