---
jupytext:
  cell_metadata_filter: all, -hidden, -heading_collapsed, -run_control, -trusted
  notebook_metadata_filter: all, -jupytext.text_representation.jupytext_version, -jupytext.text_representation.format_version,
    -language_info.version, -language_info.codemirror_mode.version, -language_info.codemirror_mode,
    -language_info.file_extension, -language_info.mimetype, -toc
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
  title: Simple regression
  version: '1.0'
---

<div class="licence">
<span>Licence CC BY-NC-ND</span>
<span>Eric MATZNER-LOBER &amp; Pierre Antoine Cornillon</span>
<span><img src="media/logo_IPParis.png" /></span>
</div>

```{code-cell} ipython3
---
vscode:
  languageId: python
---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
```

# Régression simple

```{code-cell} ipython3
---
vscode:
  languageId: python
---
### Importation des données
# Importer les données d'eucalytus dans le DataFrame pandas `eucalypt`
# source : data/eucalyptus.txt
```

### Nuage de points
Tracer le nuage de points avec `circ` en  abscisses et `ht` en ordonnées

```{code-cell} ipython3
---
vscode:
  languageId: python
---

```

### Régression simple
Effectuer une régression linéaire simple où `circ` est  la variable
explicative et `ht` la variable à expliquer. Stocker le résultat
dans l'objet `reg` et 
1. effectuer le résumé de cette modélisation;
2. afficher l'attribut contenant les paramètres estimés par MCO de la droite;
3. afficher l'attribut contenant l'estimation de l'écart-type de l'erreur.

```{code-cell} ipython3
---
vscode:
  languageId: python
---

```

### Résidus
Représenter graphiquement les résidus avec
1. en abscisse la variable `circ` et en ordonnée les résidus;
2. en abscisse l'ajustement \$\hat y\$ et en ordonnée les résidus;
3. en abscisse le numéro de ligne du tableau (index) et en ordonnées les résidus.

```{code-cell} ipython3
---
vscode:
  languageId: python
---

```

# Variabilité de l'estimation

### Importation des données
Importer les données d'eucalytus dans le DataFrame pandas `eucalypt`

```{code-cell} ipython3
---
vscode:
  languageId: python
---

```

Créer deux listes vides `beta1` et `beta2`
Faire 500 fois les étapes suivantes
1. Tirer au hasard sans remise 100 lignes dans le tableau `eucalypt`
2. Sur ce tirage effectuer une régression linéaire simple
   où `circ` est la variable explicative et `ht` la variable à expliquer. Stocker les paramètres estimés dans `beta1` et `beta2`

```{code-cell} ipython3
---
vscode:
  languageId: python
---

```

### Variabilité de \$\hat \beta_2\$
Représenter la variabilité de la variable aléatoire  \$\hat \beta_2\$.

Tracer les couples \$\hat \beta_1\$ et \$\hat \beta_2\$ et
constater la variabilité de l'estimation et la corrélation
entre les deux paramètres.

```{code-cell} ipython3
---
vscode:
  languageId: python
---

```

# Deux modèles

### Importation des données
Importer les données d'eucalytus dans le DataFrame pandas `eucalypt`

```{code-cell} ipython3
---
vscode:
  languageId: python
---

```

### Nuage de points
Tracer le nuage de points avec `circ` en  abscisses et `ht` en ordonnées
et constater que les points ne sont pas exactement autour
d'une droite mais plutôt une courbe qui est de type "racine carrée"

```{code-cell} ipython3
---
vscode:
  languageId: python
---

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

```{code-cell} ipython3
---
vscode:
  languageId: python
---

```

### Comparaison
Ajouter au nuage de points les 2 ajustements (la droite et la "racine carrée")
et choisir le meilleur modèle.

```{code-cell} ipython3
---
vscode:
  languageId: python
---

```
