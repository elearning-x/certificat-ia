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
  title: 'Correction du TP inférence'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa Bedin<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

## Modules python



Importer les modules pandas (comme `pd`) numpy (commme `np`)
matplotlib.pyplot (comme `plt`) et statsmodels.formula.api (comme `smf`)




```{code-cell} python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
```

## Intervalles de confiance (IC)



### Importation des données



Importer les données d&rsquo;eucalytus dans le DataFrame pandas `eucalypt`
[ `read_csv` de `numpy` ]




```{code-cell} python
eucalypt = pd.read_csv("data/eucalyptus.txt", header=0, sep=";")
```

### Régression simple



Effectuer une régression linéaire simple où `circ` est la variable
explicative et `ht` la variable à expliquer. Stocker le résultat dans
l&rsquo;objet `reg` [ols=de=smf=, méthode=fit=de la classe=OLS\`]




```{code-cell} python
reg = smf.ols("ht ~ 1+ circ",data=eucalypt).fit()
```

### IC des coefficients



Obtenir les IC des coefficients au niveau de 95% [=méthode=conf<sub>int</sub>\`
pour l&rsquo;instance/modèle ajusté]




```{code-cell} python
reg.conf_int(alpha=0.05)
```

### IC des prévisions



Créer une grille de 100 nouvelles observations régulièrement espacées
entre le minimum et le maximum de `circ`. Calculer un IC à 95% pour ces
100 nouvelles observations $y^*$ (prévoir les valeurs grâce à la méthode
`get_prediction` sur l&rsquo;instance/modèle estimé et utiliser la méthode
`conf_int` sur le résultat de la prévision).




```{code-cell} python
grille = pd.DataFrame({"circ" : np.linspace(eucalypt["circ"].min(),eucalypt["circ"].max(), 100)})
calculprev = reg.get_prediction(grille)
ICobs = calculprev.conf_int(obs=True, alpha=0.05)
```

### IC de l&rsquo;espérance



Pour la même grille de valeurs de `circ` que celle de la question
précédente proposer l&rsquo;IC à 95% sur les espérances $X^*\beta$




```{code-cell} python
ICdte = calculprev.conf_int(obs=False, alpha=0.05)
```

### Représentation des IC



En utilisant les 100 observations prévues ci-dessus et leurs IC
(observations et espérance) représenter sur un même graphique

-   les observations
-   l&rsquo;IC pour les prévisions
-   l&rsquo;IC pour l&rsquo;espérance.

[ `plt.plot`, `plt.legend` ]




```{code-cell} python
prev = calculprev.predicted_mean
plt.plot(eucalypt["circ"], eucalypt["ht"], 'o', color='xkcd:light grey')
plt.plot(grille['circ'], prev, 'k-', lw=2, label="E(Y)")
lesic, = plt.plot(grille['circ'], ICdte[:,0], linestyle='--', color='xkcd:cerulean', label=r"$\mathbb{E}(Y)$")
plt.plot(grille['circ'], ICdte[:, 1], linestyle='--', color='xkcd:cerulean')
lesic2, = plt.plot(grille['circ'], ICobs[:,0], linestyle='-.', color='xkcd:grass', label=r"$Y$")
plt.plot(grille['circ'], ICobs[:, 1], linestyle='-.', color='xkcd:grass')
plt.legend(handles=[lesic, lesic2], loc='upper left')
```

## IC pour deux coefficients



L&rsquo;objectif de ce TP est de tracer la région de confiance pour paramètres
et constater la différence avec 2 IC univariés. Pour ce TP nous aurons
besoin en plus des modules classiques de des modules suivants




```{code-cell} python
import math
from scipy.stats import f
```

### Importation des données



Importer les données d&rsquo;ozone dans le DataFrame pandas `ozone`
[ `read_csv` de `numpy` ]




```{code-cell} python
ozone = pd.read_csv("data/ozone.txt", header=0, sep=";")
```

### Modèle à 3 variables



Estimer un modèle de régression expliquant le maximum de la
concentration en ozone du jour (variable `O3`) par

-   la température à midi notée `T12`
-   la vitesse du vent sur l&rsquo;axe Est-Ouest notée `Vx`
-   la nébulosité à midi `Ne12` avec comme toujours la constante.

[ `ols` de `smf`, méthode `fit` de la classe `OLS` et méthode `summary` pour l&rsquo;instance/modèle ajusté ]




```{code-cell} python
modele3 = smf.ols("O3 ~ T12 + Vx + Ne12",data=ozone).fit()
```

### Région de confiance pour toutes les variables



Intéressons nous aux deux premières variables `T12` et `Vx` de
coefficients notés ici $\beta_2$ et $\beta_3$ (le coefficient $\beta_1$
est celui pour la variable constante/intercept).

Notons $F_{2:3}= \|\hat\beta/{2:3} -
\beta/{2:3}\|^2_{\hat V_{\hat\beta/{2:3}}^{-1}}$ et introduisons la
notation suivante:
$\hat V/{\hat\beta/{2:3}}=\hat\sigma [(X'X)^{-1}]/{2:3,2:3} =
\hat\sigma \Sigma$. On notera aussi que $\Sigma=U\Lambda U'$ et
$\Sigma^{{1/2}=U\Delta}{1/2} U'$ ($U$ matrice orthogonale des vecteurs
propres de $\Sigma$ et $\Delta$ matrice diagonale des valeurs propres
positives ou nulles).

1.  Montrer que $F_{2:3,2:3}$ suit une loi de
    Fisher $\mathcal{F}(2,n-4)$. Calculer son quantile à 95% avec la
    fonction `f` du sous module `scipy.stats` (méthode `isf`).




```{code-cell} python
f.isf(0.05, 2, modele3.nobs - 2)
```

1.  Déduire que la région de confiance pour $\beta_{1:2}$ est l&rsquo;image
    d&rsquo;un disque par une matrice à déterminer. Calculer cette matrice en
    python [méthode `cov_params` pour l&rsquo;instance `modele3`, fonctions
    `eigh` du sous module `np.linalg`, `np.matmul`, `np.diag`, =np.sqrt=]




```{code-cell} python
hatSigma = modele3.cov_params().iloc[1:3,1:3]
   valpr,vectpr = np.linalg.eigh(hatSigma)
   hatSigmademi = np.matmul(vectpr, np.diag(np.sqrt(valpr)))
```

1.  Construire 500 points sur le cercle [=cos= et `sin` de =np=]




```{code-cell} python
theta = np.linspace(0, 2 * math.pi, 500)
   rho = (2 * f.isf(0.05, 2, modele3.nobs - 2))**0.5
   x = rho * np.cos(theta)
   y = rho * np.sin(theta)
   XX = np.array([x, y])
```

1.  Transformer ces points via la matrice donnant ainsi l&rsquo;ellipse de
    confiance.




```{code-cell} python
ZZ = np.add(np.matmul(hatSigmademi, XX).transpose(), np.array(modele3.params[1:3]))
```

1.  Tracer l&rsquo;ellipse [=plt.fill= (pour l&rsquo;ellipse), `plt.plot` (pour le
    centre)]




```{code-cell} python
plt.fill(ZZ[:, 0], ZZ[:, 1], facecolor='yellow', edgecolor='black', linewidth=1)
   plt.plot(modele3.params[1], modele3.params[2], "+")
```

### IC univariés



Ajouter le « rectangle de confiance » issu des 2 IC univariés à
l&rsquo;ellipes en récupérant l&rsquo;`Axe` via `plt.gca()`, en créant le `patch`
rectangle avec `matplotlib.patches.Rectangle` et en l&rsquo;ajoutant avec
`ax.add_artist`.




```{code-cell} python
ICparams = modele3.conf_int(alpha=0.025)
from matplotlib.patches import Rectangle
plt.fill(ZZ[:, 0], ZZ[:, 1], facecolor='yellow', edgecolor='black', linewidth=1)
plt.plot(modele3.params[1], modele3.params[2], "+")
ax = plt.gca()
r = Rectangle(ICparams.iloc[1:3, 0],
              ICparams.diff(axis=1).iloc[1, 1],
              ICparams.diff(axis=1).iloc[2, 1],
              fill=False)
ax.add_artist(r)
```

On voit que 2 IC univariés (et donc considérer que les variables sont
indépendantes ne convient pas et des points qui sont dans la région de
confiance rectangulaire et ne sont pas dans l&rsquo;ellipse de confiance et
vice-versa&#x2026;



## IC et bootstrap



L&rsquo;objectif de ce TD est de construire un IC grâce au Bootstrap.



### Importation des données



Importer les données d&rsquo;ozone dans le DataFrame pandas `ozone`
[ `read_csv` de `numpy` ]




```{code-cell} python
ozone = pd.read_csv("data/ozone.txt", header=0, sep=";")
```

### Modèle à 3 variables



Estimer un modèle de régression expliquant le maximum de la
concentration en ozone du jour (variable `O3`) par - la température à
midi notée `T12` - la vitesse du vent sur l&rsquo;axe Est-Ouest notée `Vx` -
la nébulosité à midi `Ne12` avec comme toujours la constante. [=ols= de
`smf`, méthode `fit` de la classe `OLS` et méthode `summary` pour
l&rsquo;instance/modèle ajusté]




```{code-cell} python
modele3 = smf.ols("O3 ~ T12 + Vx + Ne12",data=ozone).fit()
```

### Bootstrap et IC



#### Calcul du modèle empirique: $\hat Y$ et $\hat\varepsilon$



Stocker les résidus dans l&rsquo;objet `residus=et les ajustememt dans =ychap`




```{code-cell} python
ychap = modele3.fittedvalues
residus = modele3.resid
```

#### Géneration d&rsquo;échantillon bootstrap



Le modèle de régression générant les $Y_i$ ($i\in\{1,\cdots,n\}$) est le
suivant
$$
Y_i = \beta_1 +  \beta_2 X_{i2} +   \beta_3 X_{i3} +   \beta_4 X_{i4} +  \varepsilon_i$
$$
où la loi de $\varepsilon_i$ (notée $F$) est inconnue.

Si on avait par exemple $B=1000$ échantillons alors on pourrait estimer
$B$ fois $\beta$ et voir à partir de ces $B$ estimations la variabilité
des $\hat\beta$ et en tirer des quantiles empiriques de niveau
$\alpha/2$ et $1-\alpha/2$ et donc un intervalle de confiance.

Bien entendu nous n&rsquo;avons qu&rsquo;un seul $n$-échantillon et si nous
souhaitons générer $B$ échantillon il faudrait connaitre $\beta$ et $F$.
L&rsquo;idée du bootstrap est de remplacer $\beta$ et $F$ inconnus par
$\hat\beta$ (l&rsquo;estimateur des MCO) et $\hat F$ (un estimateur de $F$),
de générer $B$ échantillons puis de calculer les $B$ estimations
$\hat\beta^*$ la variabilité des $\hat\beta^*$ et en tirer des quantiles
empiriques de niveau $\alpha/2$ et $1-\alpha/2$ et donc un intervalle de
confiance.

Générons $B=1000$ échantillons bootstrap. 1. Pour chaque valeur de
$b\in\{1,\cdots,B\}$ tirer indépendamment avec remise parmi les résidus
de la régression $n$ valeurs. Notons $\hat\varepsilon^{(b)}$ le vecteur
résultant de ces tirages; 2. Ajouter ces résidus à l&rsquo;ajustement $\hat Y$
pour obtenir un nouvel échantillon $Y^*$. Avec les données $X, Y^*$
obtenir l&rsquo;estimation par MCO $\hat\beta^{(b)}$; 3. Stocker la valeur
$\hat\beta^{(b)}$ dans la ligne $b$ de l&rsquo;array numpy `COEFF`. [créer une
instance de générateur aléatoire `np.random.default_rng`, et utiliser la
méthode `randint` sur cette instance ; créer une copie des colonnes
adéquates de `ozone` via la méthode `copy` afin d&rsquo;utiliser `smf.ols` et
remplir ce DataFrame avec l&rsquo;échantillon.]




```{code-cell} python
B =1000
COEFF = np.zeros((B, 4))
n = ozone.shape[0]
rng = np.random.default_rng(seed=1234)
ozoneetoile = ozone[["O3", "T12" , "Vx",  "Ne12"]].copy()
for  b in range(B):
    resetoile = residus[rng.integers(n, size=n)]
    O3etoile = np.add(ychap.values ,resetoile.values)
    ozoneetoile.loc[:,"O3"] = O3etoile
    regboot = smf.ols("O3 ~ 1+ T12 + Vx + Ne12", data=ozoneetoile).fit()
    COEFF[b] = regboot.params.values

COEFF.shape
```

### IC bootstrap



A partir des $B=1000$ valeurs $\hat\beta^{(b)}$ proposer un IC à 95%.
[ `np.quantile` ]




```{code-cell} python
pd.DataFrame(np.quantile(COEFF, [0.025, 0.975], axis=0).T)
```

## Modélisation de la hauteur d&rsquo;eucalyptus



### Importation



Importer les données d&rsquo;eucalytus dans le DataFrame pandas `eucalypt`
[=read<sub>csv</sub>= de =numpy=]




```{code-cell} python
eucalypt = pd.read_csv("data/eucalyptus.txt", header=0, sep=";")
```

### Deux régressions



Nous avons déjà lors des précédents TP effectué plusieurs modélisations.
Pour les modélisations à une variable, celle choisie était celle avec la
racine carrée (voir TP régression simple). Ensuite nous avons introduit
la régression multiple et nous allons maintenant comparer ces deux
modèles.

1.  Effectuer une régression linéaire simple où la racine carrée de
    `circ` est la variable explicative et `ht` la variable à expliquer.
    Stocker le résultat dans l&rsquo;objet `regsqrt`.
2.  Effectuer une régression linéaire multiple où la racine carrée de
    `circ` et la variable `circ` sont les variables explicatives et `ht`
    la variable à expliquer. Stocker le résultat dans l&rsquo;objet `reg`.
    [=ols= de `smf`, méthode `fit` de la classe =OLS=]




```{code-cell} python
regsqrt = smf.ols('ht~I(np.sqrt(circ))', data=eucalypt).fit()
reg = smf.ols('ht~I(np.sqrt(circ)) + circ', data=eucalypt).fit()
```

### Comparaison



1.  Comparer ces deux modèles via un test $T$ [méthode =summary=]




```{code-cell} python
reg.summary()
```

La ligne `circ` du tableau donne l&rsquo;estimation du coefficient
$\hat\beta_3$, l&rsquo;écart-type estimé du coefficient, la valeur de la
statistique $t$ du test $\mathrm{H}_0: \beta_3=0$ contre
$\mathrm{H}_1: \beta_3\neq 0$ qui vaut ici $-8.336$ et sa probabilité
critique quasi nulle. Nous repoussons donc $\mathrm{H}_0$ et le modèle
`reg` semble meilleur.

1.  Comparer ces deux modèles via un test $F$ [=stats.anova<sub>lm</sub>= du sous
    module =statsmodels.api=]




```{code-cell} python
import statsmodels.api as sm
   sm.stats.anova_lm(regsqrt,reg)
```

Nous retrouvons les mêmes résultats que précédemment (car $F=t^2$)



## L&rsquo;âge a t-il une influence sur le temps libre ?



Une enquête a été conduite sur 40 individus afin d&rsquo;étudier le lien entre
le temps libre (estimé par l&rsquo;enquêté comme le temps, en nombre d&rsquo;heures
par jour, disponible pour soi) et l&rsquo;âge. Les résultats de cette enquête
sont contenus dans le fichier `temps_libre.csv`. Nous nous proposons de
savoir si ces deux variables sont liées.

1.  Quel est le type des variables ?




```{code-cell} python
tpslibre = pd.read_csv("data/temps_libre.csv", header=0, sep=";")
   tpslibre.columns = [ "age", "tempslibre" ]
   tpslibre.describe()
```

Les deux variables sont quantitatives continues. Nous changeons les noms
   car le &rsquo;.&rsquo; est mal géré dans les formules&#x2026;

1.  Comment calcule t-on le lien (le plus commun) entre ces deux
    variables ?




```{code-cell} python
tpslibre.corr()
```

La mesure du lien est la corrélation linéaire (dont le carré vaut le
   R2), ici elle est très faible et tend à montrer qu&rsquo;il n&rsquo;y a pas de lien
   linéaire entre les 2 variables.

1.  Comment teste-t-on si l&rsquo;âge à une influence sur le temps libre à
    l&rsquo;aide de la régression ? Effectuer ce test et conclure.




```{code-cell} python
reg = smf.ols("tempslibre~1+age", data=tpslibre).fit()
   reg.summary()
```

La ligne `age` du tableau donne l&rsquo;estimation du coefficient
   $\hat\beta_2$, l&rsquo;écart-type estimé du coefficient, la valeur de la
   statistique $t$ du test $\mathrm{H}_0: \beta_2=0$ contre
   $\mathrm{H}_1: \beta_2\neq 0$ qui vaut ici $0.285$ et sa probabilité
   critique qui est de 0.777. Nous conservons donc $\mathrm{H}_0$ et il ne
   semble pas y avoir de lien linéaire.

1.  Représentez les données et discuter du bien fondé du test précédent.




```{code-cell} python
plt.plot(tpslibre.age, tpslibre.tempslibre, "*")
```

Clairement on observe 2 régimes: entre 30 et 60 ans, peu de temps libre
   et avant 30 ou après 60 plus de temps libre. Il y a une influence de
   l&rsquo;âge mais pas linéaire (plutôt constante par morceaux). Le test
   précédent est inadapté.



## L&rsquo;obésité a t-elle une influence sur la pression sanguine ?



Une enquête a été conduite sur 102 individus afin d&rsquo;étudier le lien
entre l&rsquo;obésité (estimée par le ratio du poids de la personne sur le
poids idéal obtenu dans la \og New York Metropolitan Life Tables\fg{})
et la pression sanguine en mm de mercure. Les résultats de cette enquête
sont contenus dans le fichier `obesite.csv` (dans Fun Campus les données
sont dans le répertoire `data/`). Nous nous proposons de savoir si ces
deux variables sont liées.

1.  Quel est le type des variables ?




```{code-cell} python
obesite = pd.read_csv("data/obesite.csv", header=0, sep=";")
   obesite.describe()
```

Les variables sont quantitatives.

1.  Comment calcule t-on le lien (le plus commun) entre ces deux
    variables ?
    
    Il s&rsquo;agit de la corrélation linéaire qui vaut ici




```{code-cell} python
obesite.corr()
```

La corrélation semble modérée.

1.  Comment teste-t-on si l&rsquo;obésite à une influence sur la pression à
    l&rsquo;aide de la régression ? Effectuer ce test et conclure.
    
    Nous avons vu à l&rsquo;exercice précédent qu&rsquo;il semble raisonnable de faire
    d&rsquo;abord le graphique pour voir si le modèle de régression linéaire est
    adapté:




```{code-cell} python
plt.plot(obesite.obesite, obesite.pression, "o")
```

Même si les points sont pas vraiment sur une droite nous pouvons quand
   même nous dire que ce modèle peut grossièrement convenir. Effectuons la
   régression simple et un test $t$ de nullité de pente




```{code-cell} python
reg = smf.ols("pression~1+obesite", data=obesite).fit()
   reg.summary()
```

La ligne `obesite` du tableau donne l&rsquo;estimation du coefficient
   $\hat\beta_2$, l&rsquo;écart-type estimé du coefficient, la valeur de la
   statistique $t$ du test $\mathrm{H}_0: \beta_2=0$ contre
   $\mathrm{H}_1: \beta_2\neq 0$ qui vaut ici $3.45$ et sa probabilité
   critique qui est de 0.001. Nous repoussons donc $\mathrm{H}_0$ et il
   semble y avoir un lien linéaire (assez grossier ; ce n&rsquo;est pas avec ce
   modèle que l&rsquo;on va prédire la pression sanguine).


