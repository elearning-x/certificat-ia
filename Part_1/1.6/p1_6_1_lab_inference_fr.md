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
  title: TP Inférence
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

```

## Intervalles de confiance (IC)



### Importation des données



Importer les données d&rsquo;eucalytus dans le DataFrame pandas `eucalypt`
[ `read_csv` de `numpy` ]. Dans Fun Campus les jeux de données sont dans le répertoire `data/`.




```{code-cell} python

```

### Régression simple



Effectuer une régression linéaire simple où `circ` est la variable
explicative et `ht` la variable à expliquer. Stocker le résultat dans
l&rsquo;objet `reg` [ `ols` de `smf`, méthode `fit` de la classe `OLS` ]




```{code-cell} python

```

### IC des coefficients



Obtenir les IC des coefficients au niveau de 95%
[ `méthode =conf_int` pour l&rsquo;instance/modèle ajusté ]




```{code-cell} python

```

### IC des prévisions



Créer une grille de 100 nouvelles observations régulièrement espacées
entre le minimum et le maximum de `circ`. Calculer un IC à 95% pour ces
100 nouvelles observations $y^*$ (prévoir les valeurs grâce à la méthode
`get_prediction` sur l&rsquo;instance/modèle estimé et utiliser la méthode
`conf_int` sur le résultat de la prévision).




```{code-cell} python

```

### IC de l&rsquo;espérance



Pour la même grille de valeurs de `circ` que celle de la question précédente
proposer l&rsquo;IC à 95% sur les espérances $X^*\beta$




```{code-cell} python

```

### Représentation des IC



En utilisant les 100 observations prévues ci-dessus et leurs IC
(observations et espérance) représenter sur un même graphique

-   les observations
-   l&rsquo;IC pour les prévisions
-   l&rsquo;IC pour l&rsquo;espérance.

[ `plt.plot`, `plt.legend` ]




```{code-cell} python

```

## IC pour deux coefficients



L&rsquo;objectif de ce TP est de tracer la région de confiance pour
  paramètres et constater la différence avec 2 IC univariés. 
  Pour ce TP nous aurons besoin en plus des modules classiques
  de des modules suivants




```{code-cell} python
import math
  from scipy.stats import f
```

### Importation des données



Importer les données d&rsquo;ozone dans le DataFrame pandas `ozone`
[ `read_csv` de `numpy` ]. Dans Fun Campus les jeux de données sont dans le
répertoire `data/`.




```{code-cell} python

```

### Modèle à 3 variables



Estimer un modèle de régression expliquant
  le maximum de la concentration en ozone du jour (variable `O3`) par

-   la température à midi notée `T12`
-   la vitesse du vent sur l&rsquo;axe Est-Ouest notée `Vx`
-   la nébulosité à midi `Ne12`

avec comme toujours la constante.
[ `ols` de `smf`, méthode `fit` de la classe `OLS` et
méthode `summary` pour l&rsquo;instance/modèle ajusté ]




```{code-cell} python

```

### Région de confiance pour toutes les variables



Intéressons nous aux deux premières variables `T12` et `Vx` de
coefficients notés ici $\beta_2$ et $\beta_3$ (le coefficient $\beta_1$
est celui pour la variable constante/intercept).

Notons
$F_{2:3}= \|\hat\beta_{2:3} - \beta_{2:3}\|^2_{\hat V_{\hat\beta_{2:3}}^{-1}}$
et introduisons la notation suivante:
$\hat V_{\hat\beta_{2:3}}=\hat\sigma [(X'X)^{-1}]_{2:3,2:3} = \hat\sigma \Sigma$.
On notera aussi que $\Sigma=U\Lambda U'$ et
$\Sigma^{1/2}=U\Delta^{1/2} U'$ ($U$ matrice orthogonale des vecteurs
propres de $\Sigma$ et $\Delta$ matrice diagonale des valeurs propres
positives ou nulles).

1.  Montrer que $F_{2:3,2:3}$ suit une loi de
    Fisher $\mathcal{F}(2,n-4)$. Calculer son quantile à 95% avec la
    fonction `f` du sous module `scipy.stats` (méthode `isf`).




```{code-cell} python

```

1.  Déduire que la région de confiance pour $\beta_{1:2}$ est l&rsquo;image
    d&rsquo;un disque par une matrice à déterminer. Calculer cette matrice
    en python [ méthode `cov_params` pour l&rsquo;instance `modele3`, fonctions `eigh` du sous module `np.linalg`,
    `np.matmul`, `np.diag`, `np.sqrt` ]




```{code-cell} python

```

1.  Construire 500 points sur le cercle [ `cos` et `sin` de `np` ]




```{code-cell} python

```

1.  Transformer ces points via la matrice donnant ainsi l&rsquo;ellipse de confiance.




```{code-cell} python

```

1.  Tracer l&rsquo;ellipse [ `plt.fill` (pour l&rsquo;ellipse), `plt.plot` (pour le
    centre) ]




```{code-cell} python

```

### IC univariés



Ajouter le « rectangle de confiance » issu des 2 IC univariés à l&rsquo;ellipes en
récupérant l&rsquo;`Axe` via `plt.gca()`, en créant le `patch` rectangle avec
`matplotlib.patches.Rectangle` et en l&rsquo;ajoutant avec `ax.add_artist`.




```{code-cell} python

```

## IC et bootstrap



L&rsquo;objectif de ce TD est de construire un IC grâce au Bootstrap.



### Importation des données



Importer les données d&rsquo;ozone dans le DataFrame pandas `ozone`
[ `read_csv` de `numpy` ]. Dans Fun Campus les jeux de données sont dans le
répertoire `data/`.




```{code-cell} python

```

### Modèle à 3 variables



Estimer un modèle de régression expliquant
le maximum de la concentration en ozone du jour (variable `O3`) par

-   la température à midi notée `T12`
-   la vitesse du vent sur l&rsquo;axe Est-Ouest notée `Vx`
-   la nébulosité à midi `Ne12`

avec comme toujours la constante.
[ `ols` de `smf`, méthode `fit` de la classe `OLS` et
méthode `summary` pour l&rsquo;instance/modèle ajusté ]



### Bootstrap et IC



#### Calcul du modèle empirique: $\hat Y$ et $\hat\varepsilon$



Stocker les résidus dans l&rsquo;objet `residus=et les ajustememt dans =ychap`



#### Géneration d&rsquo;échantillon bootstrap



Le modèle de régression générant les $Y_i$ ($i\in\{1,\cdots,n\}$) est le suivant
$$
Y_i = \beta_1 +  \beta_2 X_{i2} +   \beta_3 X_{i3} +   \beta_4 X_{i4} +  \varepsilon_i$
$$
où la loi de $\varepsilon_i$ (notée $F$) est inconnue.

Si on avait par exemple  $B=1000$ échantillons alors on pourrait
estimer  $B$ fois $\beta$ et voir à partir de ces $B$ estimations
la variabilité des $\hat\beta$ et en tirer des quantiles empiriques de
niveau $\alpha/2$ et $1-\alpha/2$ et donc un intervalle de confiance.

Bien entendu nous n&rsquo;avons qu&rsquo;un seul $n$-échantillon et si nous souhaitons
générer $B$ échantillons il faudrait connaitre $\beta$ et $F$. L&rsquo;idée du bootstrap
est de remplacer $\beta$ et $F$ inconnus par $\hat\beta$ (l&rsquo;estimateur des MCO)
et $\hat F$ (un estimateur de $F$), de générer $B$ échantillons puis de calculer
les  $B$ estimations $\hat\beta^*$
la variabilité des $\hat\beta^*$ et en tirer des quantiles empiriques de
niveau $\alpha/2$ et $1-\alpha/2$ et donc un intervalle de confiance.

Générons  $B=1000$ échantillons bootstrap.

1.  Pour chaque valeur de $b\in\{1,\cdots,B\}$ tirer indépendamment avec remise
    parmi les résidus de la régression $n$ valeurs.
    Notons $\hat\varepsilon^{(b)}$ le vecteur résultant de ces tirages;
2.  Ajouter ces résidus à l&rsquo;ajustement $\hat Y$ pour obtenir un nouvel
    échantillon $Y^*$. Avec les données $X, Y^*$ obtenir l&rsquo;estimation par
    MCO $\hat\beta^{(b)}$;
3.  Stocker la valeur $\hat\beta^{(b)}$ dans la ligne $b$ de
    l&rsquo;array numpy `COEFF`.
    [ créer une instance de générateur aléatoire `np.random.default_rng`, et
    utiliser la méthode `randint` sur cette instance ; créer une copie des
    colonnes adéquates de `ozone` via la méthode `copy` afin d&rsquo;utiliser
    `smf.ols` et remplir ce DataFrame avec l&rsquo;échantillon. ]




```{code-cell} python

```

### IC bootstrap



A partir des $B=1000$ valeurs $\hat\beta^{(b)}$ proposer un IC à 95%.
[ `np.quantile` ]



## Modélisation de la hauteur d&rsquo;eucalyptus



### Importation



Importer les données d&rsquo;eucalytus dans le DataFrame pandas `eucalypt`
[ `read_csv` de `numpy` ]. Dans Fun Campus les jeux de données sont dans le répertoire `data/`.




```{code-cell} python

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
    [ `ols` de `smf`, méthode `fit` de la classe `OLS` ]




```{code-cell} python

```

### Comparaison



1.  Comparer ces deux modèles via un test $T$ [ méthode `summary` ]




```{code-cell} python

```

1.  Comparer ces deux modèles via un test $F$ [ `stats.anova_lm` du sous
    module `statsmodels.api` ]




```{code-cell} python

```

## L&rsquo;âge a t-il une influence sur le temps libre ?



Une enquête a été conduite sur 40 individus afin d&rsquo;étudier le lien
entre le temps libre (estimé par l&rsquo;enquêté comme le temps, en nombre
d&rsquo;heures par jour, disponible pour soi) et l&rsquo;âge. Les résultats de
cette enquête sont contenus dans le fichier
`temps_libre.csv` (dans Fun Campus les jeux de données sont dans le répertoire `data/`).
Nous nous proposons de savoir si ces deux variables sont liées.

1.  Quel est le type des variables ?




```{code-cell} python

```

1.  Comment calcule t-on le lien (le plus commun) entre ces deux
    variables ?




```{code-cell} python

```

1.  Comment teste-t-on si l&rsquo;âge à une influence sur le temps libre
    à l&rsquo;aide de la régression ? Effectuer ce test et conclure.




```{code-cell} python

```

1.  Représentez les données et discuter du bien fondé du test précédent.




```{code-cell} python

```

## L&rsquo;obésité a t-elle une influence sur la pression sanguine ?



Une enquête a été conduite sur 102 individus afin d&rsquo;étudier le lien
entre l&rsquo;obésité (estimée par le ratio du poids de la personne sur le
poids idéal obtenu dans la &ldquo;New York Metropolitan Life Tables&rdquo;)
et la pression sanguine en mm de mercure. Les résultats de
cette enquête sont contenus dans le fichier  `obesite.csv`
(dans Fun Campus les données sont dans le répertoire `data/`).
Nous nous proposons de savoir si ces deux variables sont liées.

1.  Quel est le type des variables ?




```{code-cell} python

```

1.  Comment calcule t-on le lien (le plus commun) entre ces deux
    variables ?
2.  Comment teste-t-on si l&rsquo;obésite à une influence sur la
    pression à l&rsquo;aide de la régression ? Effectuer ce test
    et conclure.




```{code-cell} python

```
