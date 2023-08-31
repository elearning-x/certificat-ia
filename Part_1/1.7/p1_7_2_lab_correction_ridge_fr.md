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
  title: 'Correction du TP régression ridge'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa Bedin<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

+++

## Modules



Importer les modules pandas (comme `pd`) numpy (commme `np`)
le sous module `pyplot` de `matplotlib` comme `plt`
les fonctions `StandardScaler` de `sklearn.preprocessing`,
`Ridge` de  `sklearn.linear_model`,
`RidgeCV` de  `sklearn.linear_model`,
`Pipeline` de `sklearn.pipeline`,
`cross_val_predict` de `sklearn.model_selection`,
`KFold` de `sklearn.model_selection`
`make_scorer` de `sklearn.metrics`


```{code-cell} python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
```

## Régression ridge sur les données d&rsquo;ozone



#### Importation des données



Importer les données d&rsquo;ozone `ozonecomplet.csv` et éliminer les deux dernières
variables (qualitatives) et faites un résumé numérique par variable [méthode
`astype` sur la colonne du DataFrame et méthode `describe` sur l&rsquo;instance
DataFrame]




```{code-cell} python
ozone = pd.read_csv("data/ozonecomplet.csv", header=0, sep=";")
ozone = ozone.drop(['nomligne', 'Ne', 'Dv'], axis=1)
ozone.describe()
```

#### Création des tableaux `numpy`



avec l&rsquo;aide des méthodes d&rsquo;instance `iloc` ou `loc` créer les tableaux `numpy`
`y` et `X` (on se servira de l&rsquo;attribut `values` qui donne le tableau `numpy` sous-jascent)




```{code-cell} python
y = ozone.O3.values
X = ozone.iloc[:,1:].values
```

#### Centrage et réduction



Centrer et réduire les variable avec `StandardScaler` selon le schéma
suivant

1.  créer une instance avec la fonction `StandardScaler`. On notera
    `scalerX` l&rsquo;instance créée.
2.  l&rsquo;ajuster via la méthode d&rsquo;instance `fit` (calcul des moyennes et écart-types) et avec le tableau `numpy` des $X$
3.  Transformer le tableau $X$ en tableau centré réduit via la méthode d&rsquo;instance `transform` et avec le tableau `numpy` des $X$.




```{code-cell} python
scalerX = StandardScaler().fit(X)
Xcr= scalerX.transform(X)
```

#### Calcul de la régression Ridge pour $\lambda=0.00485$



1.  Estimation/ajustement: en utilisant les données centrées réduites pour $X$ et
    le vecteur `y` estimer le modèle de régression Ridge:
    -   Instancier un modèle `Ridge` avec la fonction éponyme




```{code-cell} python
ridge = Ridge(alpha=0.00485)
```

Attention dans `scikitlearn` le paramètre $\lambda$ de la ridge (et lasso
     et elastic-net) s&rsquo;appelle $\alpha$.

-   Estimer le modèle avec $\lambda=0.00485$ et la méthode d&rsquo;instance `fit`




```{code-cell} python
ridge.fit(Xcr, y)
```

1.  Afficher $\hat\beta(\lambda)$




```{code-cell} python
print(ridge.coef_)
```

    : [-0.05647245  8.93275737  2.52242717 -5.65394122 -0.95834909  0.42724318
       :   2.4850383   0.0863599   1.17132698  9.90737987]

1.  Prévoir une valeur en $x^*=(17, 18.4, 5, 5, 7, -4.3301, -4, -3, 87)'$ (on pourra constater qu&rsquo;il s&rsquo;agit de la seconde ligne du tableau initial).




```{code-cell} python
print(ridge.predict(Xcr[1,:].reshape(1, 10)))
```

    : [76.0489625]

#### Pipeline



On voit bien que si l&rsquo;on nous donne des valeurs nouvelles il faut enlever
la moyenne et diviser par l&rsquo;écart-type ce qui n&rsquo;est pas très pratique.

-   Vérifier que `scalerX.transform(X[1,:].reshape(1, 10))` donne bien `Xcr[1,:]`. Cependant
    l&rsquo;enchainement «transformation des X» puis «modélisation» peut être automatisé
    grâce au [Pipeline](https://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html)




```{code-cell} python
np.all(np.abs( scalerX.transform(X[1,:].reshape(1, 10))[0,:] - Xcr[1,:])<1e-10)
```

    : True

-   Créer une instance de pipeline:
    1.  Créer une instance de `StandardScaler`




```{code-cell} python
cr = StandardScaler()
```

1.  Créer une instance de Régression Ridge




```{code-cell} python
ridge = Ridge(alpha=0.00485)
```

1.  Créer une instance de `Pipeline` avec l&rsquo;argument `steps` qui
    sera une liste de tuple dont le premier élément est le nom de l&rsquo;étape
    (par exemple `"cr"` ou `"ridge"`) et dont la seconde valeur sera l&rsquo;instance de l&rsquo;étape à faire (instances créées aux étapes précédentes.)




```{code-cell} python
pipe = Pipeline(steps=[("cr", cr) , ("ridge",  ridge)])
```

-   ajuster cette instance de pipeline avec la méthode d&rsquo;instance `fit` avec les
    données `X` et `y`.




```{code-cell} python
pipe.fit(X,y)
```

-   Retrouver les paramètres $\hat\beta(\lambda)$ en affectant la coordonnée
    `"ridge"` (nom de l&rsquo;étape choisi ici) de l&rsquo;attribut named<sub>steps</sub> dans un objet.
    Les attributs et méthodes de cet objets seront ensuite les mêmes que ceux
    la régression `Ridge` après ajustement.




```{code-cell} python
er=pipe.named_steps["ridge"]
print(er.coef_)
```

    : [-0.05647245  8.93275737  2.52242717 -5.65394122 -0.95834909  0.42724318
      :   2.4850383   0.0863599   1.17132698  9.90737987]

-   Retrouver l&rsquo;ajustement pour $x^*$




```{code-cell} python
print(pipe.predict(X[1,:].reshape(1,10)))
```

    : [76.0489625]

#### Evolution des coefficients selon $\lambda$



##### Calcul d&rsquo;une grille de $\lambda$



La grille classique pour ridge est constituée sur la même idée
que celle pour le lasso:

1.  Calcul de la valeur maximale $\lambda_0 = \arg\max_{i} |[X'y]_i|/n$
    Pour le lasso au delà de cette contrainte tous les coefficients sont nuls.
2.  On prend une grille en puissance de 10, avec les exposants
    qui varient entre 0 et -4 (en général on prend 100 valeurs régulièrement
    espacées)
3.  Cette grille est multipliée par $\lambda_0$
4.  Pour la régression ridge la grille précédente
    (qui est celle pour le lasso)
    est multipliée par $100$ ou $1000$.

On a donc la grille $\{\lambda_0 10^{k+2}\}_{k=0}^{-4}$.

Créer cette grille avec `np.linspace`, méthode d&rsquo;instance `transpose`,
`dot` et `max` (sans oublier l&rsquo;attribut `shape` pour $n$).




```{code-cell} python
llc = np.linspace(0, -4, 100)
l0 = np.abs(Xcr.transpose().dot(y)).max()/X.shape[0]
alphas_ridge = l0*100*10**(llc)
```

##### Tracer l&rsquo;évolution des $\hat\beta(\lambda)$



Tracer en fonction du logarithme des valeurs de $\lambda$
de la grille les coefficients $\hat\beta(\lambda)$

D&rsquo;abord la liste des coefficients:




```{code-cell} python
lcoef = []
for ll in alphas_ridge:
    pipe = Pipeline(steps=[("cr", StandardScaler()) , ("ridge",  Ridge(alpha=ll))]).fit(X,y)
    er = pipe.named_steps["ridge"]
    lcoef.append(er.coef_)
```

ou sans le pipeline




```{code-cell} python
lcoef = []
for ll in alphas_ridge:
    rr = Ridge(alpha=ll).fit(Xcr,y)
    lcoef.append(rr.coef_)
```

Puis le tracé




```{code-cell} python
plt.plot(np.log(alphas_ridge), lcoef)
plt.show()
```

Nous voyons clairement que les valeurs sont retrécies vers 0 quand
la valeur de $\lambda$ augmente.



#### $\hat \lambda$ optimal (par validation croisée 10 blocs/fold)



##### Séparation en 10 blocs



Nous allons séparer le jeu de données en 10 blocs grâce
à la fonction [KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold): créer une instance de `KFold` nommée `kf`.




```{code-cell} python
kf = KFold(n_splits = 10, shuffle=True, random_state=0)
```

##### Sélection du $\hat \lambda$ optimal



1.  Créer un DataFrame `res` avec 100 colonnes de 0
2.  Faire une boucle sur tous les blocs ; utiliser la méthode d&rsquo;instance
    `split` sur `kf` avec les données `X`
3.  Pour chaque «tour» de bloc faire
    1.  estimer sur les 9 blocs en apprentissage
        les modèles ridge pour chaque $\lambda$ de la grille.
    2.  prévoir les données du bloc en validation
    3.  ranger dans les lignes correspondantes de `res` pour les
        100 colonnes correspondantes aux 100 modèles ridge.




```{code-cell} python
kf = KFold(n_splits=10, shuffle=True, random_state=0)
res = pd.DataFrame(np.zeros((X.shape[0], len(alphas_ridge))))
for app_index, val_index in kf.split(X):
    Xapp = Xcr[app_index,:]
    yapp = y[app_index]
    Xval = Xcr[val_index,:]
    yval = y[val_index]
    for j, ll in enumerate(alphas_ridge):
        rr = Ridge(alpha=ll).fit(Xapp, yapp)
        res.iloc[val_index,j] = rr.predict(Xval)
```

#### Sélection du $\hat \lambda$ optimal



En prenant l&rsquo;erreur quadratique $\|Y - \hat Y(\lambda)\|^2$
donner le meilleur modèle (et donc le $\hat \lambda$ optimal )
[ méthode d&rsquo;instance `apply` sur `res` et `argmin` ]




```{code-cell} python
sse = res.apply(lambda x: ((x-y)**2).sum(), axis=0)
print(alphas_ridge[sse.argmin()])
```

    23.05516147986161

#### Représentation graphique



Représenter en abscisse les logarithmes des valeurs de $\lambda$
sur la grille et en ordonnée l&rsquo;erreur quadratique calculée
en question précédente.




```{code-cell} python
plt.plot(np.log(alphas_ridge), sse, "-")
```

#### Modéliser rapidement



Les questions précédentes peuvent être enchainées plus rapidement
grâce à [=cross<sub>val</sub><sub>predict</sub>=]

1.  Les questions précédentes peuvent être enchainées plus rapidement
    grâce à $$ =cross_val_predict= $$ (la grille devra être calculée à la main)
    Calcul de la grille (il n&rsquo;existe pas encore de calcul de grille)




```{code-cell} python
scalerX = StandardScaler().fit(X)
Xcr= scalerX.transform(X)
llc = np.linspace(0, -4, 100)
l0 = np.abs(Xcr.transpose().dot(y)).max()/X.shape[0]
alphas_ridge = l0*100*10**(llc)
```

Validation croisée 10 blocs




```{code-cell} python
kf = KFold(n_splits=10, shuffle=True, random_state=0)
resbis = pd.DataFrame(np.zeros((X.shape[0], len(alphas_ridge))))
for j, ll in enumerate(alphas_ridge):
    resbis.iloc[:,j] = cross_val_predict(Ridge(alpha=ll),Xcr,y,cv=kf)
```

et résultat comme dans les questions précédentes.

1.  Il est nécessaire de contruire la grille:




```{code-cell} python
scalerX = StandardScaler().fit(X)
Xcr= scalerX.transform(X)
llc = np.linspace(0, -4, 100)
l0 = np.abs(Xcr.transpose().dot(y)).max()/X.shape[0]
alphas_ridge = l0*100*10**(llc)
```

Ici on va utiliser la fonction RidgeCV avec toujours `kf`




```{code-cell} python
kf = KFold(n_splits=10, shuffle=True, random_state=0)
modele_ridge = RidgeCV(alphas=alphas_ridge, cv=kf, scoring = 'neg_mean_squared_error').fit(Xcr, y)
```

Le résultat est un modèle ridge ajusté qui
   contient $\hat\lambda$ qui vaut ici




```{code-cell} python
print(modele_ridge.alpha_)
```

    : 27.770023623731202

ainsi que $\hat\beta(\hat\lambda)$




```{code-cell} python
print(modele_ridge.coef_)
```

    : [ 3.04681722  4.7852898   3.82669397 -3.74215398 -2.42237328 -0.16252902
       :   2.7504887   0.50796561  1.12687075  7.4831738 ]

1.  Si nous préférons revenir à la somme par bloc il faut construire une
    fonction de perte et un (objet) score associé




```{code-cell} python
def my_custom_loss_func(y_true, y_pred):
    sse = np.sum((y_true - y_pred)**2)
    return sse
monscore = make_scorer(my_custom_loss_func, greater_is_better=False)
```

que l&rsquo;on peut utiliser ensuite




```{code-cell} python
kf = KFold(n_splits=10, shuffle=True, random_state=0)
modele_ridge = RidgeCV(alphas=alphas_ridge, cv=kf, scoring = monscore).fit(Xcr, y)
```

Le résultat est un modèle ridge ajusté qui
   contient $\hat\lambda$ qui vaut ici




```{code-cell} python
print(modele_ridge.alpha_)
```

    : 23.05516147986161
