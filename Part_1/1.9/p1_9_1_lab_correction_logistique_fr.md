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
  title: 'Correction du TP régression Logistique'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa Bedin<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

## Modules python



Importer les modules pandas (comme `pd`) numpy (commme `np`)
matplotlib.pyplot (comme  `plt`) et statsmodels.formula.api (comme `smf`).




```{code-cell} python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
```

## Régression logistique



### Importation des données



Importer les données `artere.txt` dans le DataFrame pandas `artere`
[ `read_csv` de `numpy` ]. Sur Fun Campus le chemin est `data/artere.txt`. Outre l&rsquo;age et la présence=1/absence=0 de la maladie cardio-vasculaire (`chd`) une variable qualitative à 8 modalités donne
les classes d&rsquo;age (`agegrp`)




```{code-cell} python
artere = pd.read_csv("data/artere.txt", header=0, sep=" ")
```

### Nuage de points



Tracer le nuage de points avec `age` en  abscisses et `chd` en ordonnées
[ `plt.plot` ]




```{code-cell} python
plt.scatter(x="age", y="chd", c="chd", data=artere)
plt.show()
```

### Régression logistique



Effectuer une régression logistique où `age` est  la variable
explicative et `chd` la variable binaire à expliquer. Stocker le résultat
dans l&rsquo;objet `reg` et

1.  effectuer le résumé de cette modélisation;




```{code-cell} python
modele = smf.logit('chd~age', data=artere).fit()
print(modele.summary())
```

Le modèle possède une log-vraisemblance de -53.677, le modèle avec uniquement la constante (appelé ici Null) à de son coté une log-vraisemblance de -68.331. L&rsquo;ajout de l&rsquo;âge permet un gain de vraisemblance assez conséquent.

1.  afficher l&rsquo;attribut contenant les paramètres estimés par régression logistique.




```{code-cell} python
print(modele.params)
```

### Prévision et probabilités estimées



Afficher l&rsquo;ajustement/prévision pour les données de l&rsquo;échantillon via la méthode `predict` (sans arguments) sur le modèle `reg`. Que représente ce vecteur:

-   une probabilité d&rsquo;être malade pour chaque valeur de l&rsquo;age de
    l&rsquo;échantillon (OUI on modélise la probabilité que $Y=1$ et dans l&rsquo;échantillon $Y_i=1$ équivaut à $Y_i$ malade)
-   une probabilité d&rsquo;être non-malade pour chaque valeur de l&rsquo;age de l&rsquo;échantillon (NON)
-   une prévision de l&rsquo;état malade/non-malade pour chaque valeur de l&rsquo;age de l&rsquo;échantillon (NON, par défaut la fonction renvoie la probabilité)




```{code-cell} python
modele.predict()
```

Donner la prévision de l&rsquo;état malade/non-malade avec l&rsquo;indicatrice que $\hat p(x)>s$ où $s$ est le seuil classique de 0.5.




```{code-cell} python
print(modele.predict()>0.5)
```

### Matrice de confusion



Afficher la matrice de confusion estimée sur les données de
l&rsquo;échantillon pour un seuil choisi à 0.5.

Une méthode manuelle est la suivante




```{code-cell} python
yhat = modele.predict()>0.5
pd.crosstab(index=df['Age'], columns=df['Grade'])
```

mais il existe aussi une fonction adptée uniquement à l&rsquo;estimation de la matrice de confusion en ajustement:




```{code-cell} python
modele.pred_table(threshold=0.5)
```

### Résidus



Représenter graphiquement les résidus de déviance avec

1.  en abscisse la variable `age` et en ordonnée les résidus
    [ attribut `resid_dev` du modèle ];
2.  en abscisse le numéro de ligne du tableau (index) après permutation aléatoire et en ordonnées les résidus.

[ `plt.plot`, méthode `predict` pour l&rsquo;instance/modèle ajusté et
`np.arange` pour générer les numéros de ligne avec l&rsquo;attribut `shape`
du DataFrame ; créer une instance de générateur aléatoire `np.random.default_rng` et utiliser `rng.permutation`
sur les numéros de ligne ]




```{code-cell} python
plt.plot(artere.age, modele.resid_dev, "+")
plt.show()
```

Nous retrouvons l&rsquo;allure   caractéristique du graphique résidus fonction
de $\hat p$ (ou de l&rsquo;age ici). Ce type de graphique n&rsquo;est pas utilisé en pratique.




```{code-cell} python
rng = np.random.default_rng(seed=1234)
indexp = rng.permutation(np.arange(artere.shape[0]))
plt.plot(indexp, modele.resid_dev, "+")
plt.show()
```

Aucune observation avec des valeurs extrèmes, le modèle ajuste bien les
données.



## Simulation de données  Variabilité de $\hat \beta_2$



### Simulation



1.  Générer $n=100$ valeurs de $X$ uniformément entre 0 et 1.
2.  Pour chaque valeur $X_i$ simuler $Y_i$ selon un modèle logistique
    de paramètres $\beta_1=-5$ et $\beta_2=10$

[ créer une instance de générateur aléatoire `np.random.default_rng` et utiliser `rng.uniform` et `rng.binomial` ]




```{code-cell} python
rng = np.random.default_rng(seed=123)
X = rng.uniform(size=100)
Y = np.copy(X)
for i,xi in enumerate(X):
    proba = 1 / (1 + np.exp( -(-5 + 10 * xi) ))
    Y[i]=rng.binomial(1, proba, 1)[0]
df = pd.DataFrame({"X" : X, "Y" : Y})
```

### Estimation



Estimer les paramètres $\beta_1$ et $\beta_2$




```{code-cell} python
modele = smf.logit('Y~X', data=df).fit()
print(modele.params)
```

### Variabilité de l&rsquo;estimation



Refaire les deux questions ci-dessus 500 fois et constater par un graphique adapté la variabilité de $\hat \beta_2$.

Simulons 500 fois les données




```{code-cell} python
hatbeta2 = []
for it in range(500):
    X = rng.uniform(size=100)
    Y = np.copy(X)
    for i,xi in enumerate(X):
        proba = 1 / (1 + np.exp( -(-5 + 10 * xi) ))
        Y[i]=rng.binomial(1, proba, 1)[0]
    df = pd.DataFrame({"X" : X, "Y" : Y})
    modele = smf.logit('Y~X', data=df).fit()
    hatbeta2.append(modele.params[1])
```

Et construisons un histogramme




```{code-cell} python
plt.hist(hatbeta2, bins=30)
plt.show()
```

## Deux régressions logistiques simples



### Importation des données



Importer les données `artere.txt` dans le DataFrame pandas `artere`
[ `read_csv` de `numpy` ]. Sur Fun Campus le chemin est `data/artere.txt`. Outre l&rsquo;age et la présence=1/absence=0 de la maladie cardio-vasculaire (`chd`) une variable qualitative à 8 modalités donne
les classes d&rsquo;age (`agegrp`)




```{code-cell} python
artere = pd.read_csv("data/artere.txt", header=0, sep=" ")
```

### Deux régressions logistiques



1.  Effectuer une régression logistique simple où `age` est la
    variable explicative et `chd` la variable binaire à expliquer;
2.  Refaire la même chose avec la racine carrée de `age`
    comme variable explicative;




```{code-cell} python
modele1 = smf.logit('chd~age', data=artere).fit()
modele2 = smf.logit('chd~I(np.sqrt(age))', data=artere).fit()
```

### Comparaison



Ajouter au nuage de points les 2 ajustements (la droite et la &ldquo;racine carrée&rdquo;)
et choisir le meilleur modèle via un critère numérique.
[ méthode `argsort` sur une colonne du DataFrame et `plt.plot` ; utiliser le résumé des modèles ]




```{code-cell} python
sel = artere['age'].argsort()
plt.scatter(x="age", y="chd", c="chd", data=artere)
plt.plot(artere.age.iloc[sel], modele1.predict()[sel], "b-", artere.age.iloc[sel], modele2.predict()[sel], "r-"  )
plt.show()
```

Comme les deux modèles ont le même nombre de variables explicatives nous pouvons comparer celles-ci et la plus élevée donne le meilleur modèle. C&rsquo;est le modèle 1 qui l&rsquo;emporte mais les log-vraisemblances sont assez comparables.


