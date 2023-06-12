```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
```

# Régression simple


```python
### Importation des données
Importer les données d'eucalytus dans le DataFrame pandas `eucalypt`
```

### Nuage de points
Tracer le nuage de points avec `circ` en  abscisses et `ht` en ordonnées

### Régression simple
Effectuer une régression linéaire simple où `circ` est  la variable
explicative et `ht` la variable à expliquer. Stocker le résultat
dans l'objet `reg` et 
1. effectuer le résumé de cette modélisation;
2. afficher l'attribut contenant les paramètres estimés par MCO de la droite;
3. afficher l'attribut contenant l'estimation de l'écart-type de l'erreur.

### Résidus
Représenter graphiquement les résidus avec
1. en abscisse la variable `circ` et en ordonnée les résidus;
2. en abscisse l'ajustement \$\hat y\$ et en ordonnée les résidus;
3. en abscisse le numéro de ligne du tableau (index) et en ordonnées les résidus.

# Variabilité de l'estimation

### Importation des données
Importer les données d'eucalytus dans le DataFrame pandas `eucalypt`

Créer deux listes vides `beta1` et `beta2`
Faire 500 fois les étapes suivantes
1. Tirer au hasard sans remise 100 lignes dans le tableau `eucalypt`
2. Sur ce tirage effectuer une régression linéaire simple
   où `circ` est la variable explicative et `ht` la variable à expliquer. Stocker les paramètres estimés dans `beta1` et `beta2`

### Variabilité de \$\hat \beta_2\$
Représenter la variabilité de la variable aléatoire  \$\hat \beta_2\$.

Tracer les couples \$\hat \beta_1\$ et \$\hat \beta_2\$ et
constater la variabilité de l'estimation et la corrélation
entre les deux paramètres.

# Deux modèles

### Importation des données
Importer les données d'eucalytus dans le DataFrame pandas `eucalypt`

### Nuage de points
Tracer le nuage de points avec `circ` en  abscisses et `ht` en ordonnées
et constater que les points ne sont pas exactement autour
d'une droite mais plutôt une courbe qui est de type "racine carrée"

### Deux régressions simples
1. Effectuer une régression linéaire simple où `circ` est
   la variable explicative et `ht` la variable à expliquer.
   Stocker le résultat dans l'objet `reg`
2. Effectuer une régression linéaire simple où la racine carrée de `circ`
   est  la variable explicative et `ht` la variable à expliquer.
   Stocker le résultat dans l'objet `regsqrt`. On pourra utiliser les
   opérations et fonctions dans les formules
   (voir https://www.statsmodels.org/stable/example_formulas.html)

### Comparaison
Ajouter au nuage de points les 2 ajustements (la droite et la "racine carrée")
et choisir le meilleur modèle.
