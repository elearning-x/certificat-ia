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

+++

# Modules python
Importer les modules pandas (comme `pd`) numpy (commme `np`)
matplotlib.pyplot (comme  `plt`) et statsmodels.formula.api (comme `smf`)


```{code-cell} python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
```

# Intervalles de confiance (IC)

### Importation des données
Importer les données d'eucalytus dans le DataFrame pandas `eucalypt`
\[`read_csv` de `numpy`\]


```{code-cell} python
eucalypt = pd.read_csv("data/eucalyptus.txt", header=0, sep=";")
```

### Régression simple
Effectuer une régression linéaire simple où `circ` est  la variable
explicative et `ht` la variable à expliquer. Stocker le résultat
dans l'objet `reg` 
\[ols` de `smf`, méthode `fit` de la classe `OLS`\]


```{code-cell} python
reg = smf.ols("ht ~ 1+ circ",data=eucalypt).fit()
```

### IC des coefficients
Obtenir les IC des coefficients au niveau de 95%
\[`méthode `conf_int` pour l'instance/modèle ajusté\]


```{code-cell} python
reg.conf_int(alpha=0.05)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Intercept</th>
      <td>8.684772</td>
      <td>9.390179</td>
    </tr>
    <tr>
      <th>circ</th>
      <td>0.249805</td>
      <td>0.264470</td>
    </tr>
  </tbody>
</table>
</div>



### IC des prévisions
Créer une grille de 100 nouvelles observations régulièrement espacées 
entre le minimum et le maximum de `circ`.
Calculer un IC à 95% pour ces 100 nouvelles observations \$y^*\$ (prévoir les valeurs 
grâce à la méthode `get_prediction` sur l'instance/modèle estimé et utiliser
la méthode `conf_int` sur le résultat de la prévision).


```{code-cell} python
grille = pd.DataFrame({"circ" : np.linspace(eucalypt["circ"].min(),eucalypt["circ"].max(), 100)})
calculprev = reg.get_prediction(grille)
ICobs = calculprev.conf_int(obs=True, alpha=0.05)
```

### IC de l'espérance
Pour la même grille de valeurs de `circ` que celle de la question précédente
proposer l'IC à 95% sur les espérances \$X^*\beta\$


```{code-cell} python
ICdte = calculprev.conf_int(obs=False, alpha=0.05)
```

### Représentation des IC 
En utilisant les 100 observations prévues ci-dessus et leurs IC
(observations et espérance) représenter sur un même graphique
- les observations
- l'IC pour les prévisions
- l'IC pour l'espérance.
\[`plt.plot`, `plt.legend`\]


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




    <matplotlib.legend.Legend at 0x7f8682a68df0>




    
![png](p1_6_1_lab_correction_inference_fr_files/p1_6_1_lab_correction_inference_fr_14_1.png)
    


# IC pour deux coefficients
L'objectif de ce TP est de tracer la région de confiance pour
paramètres et constater la différence avec 2 IC univariés. 
Pour ce TP nous aurons besoin en plus des modules classiques
de des modules suivants


```{code-cell} python
import math
from scipy.stats import f
```

### Importation des données
Importer les données d'ozone dans le DataFrame pandas `ozone`
\[`read_csv` de `numpy`\]


```{code-cell} python
ozone = pd.read_csv("data/ozone.txt", header=0, sep=";")
```

### Modèle à 3 variables
Estimer un modèle de régression expliquant
le maximum de la concentration en ozone du jour (variable `O3`) par 
- la température à midi notée `T12`
- la vitesse du vent sur l'axe Est-Ouest notée `Vx`
- la nébulosité à midi `Ne12`
avec comme toujours la constante.
\[`ols` de `smf`, méthode `fit` de la classe `OLS` et 
méthode `summary` pour l'instance/modèle ajusté\]


```{code-cell} python
modele3 = smf.ols("O3 ~ T12 + Vx + Ne12",data=ozone).fit()
```

###  Région de confiance pour toutes les variables
Intéressons nous aux deux premières variables `T12` et `Vx` de coefficients notés ici
\$\beta_2\$ et \$\beta_3\$ (le coefficient \$\beta_1\$ est celui pour la variable 
constante/intercept).

Notons \$F_{2:3}= \\|\hat\beta_{2:3} - \beta_{2:3}\\|^2_{\hat V_{\hat\beta_{2:3}}^{-1}}\$
et introduisons la notation suivante: 
\$\hat V_{\hat\beta_{2:3}}=\hat\sigma [(X'X)^{-1}]_{2:3,2:3} = \hat\sigma \Sigma\$.
On notera aussi que \$\Sigma=U\Lambda U'\$ et \$\Sigma^{1/2}=U\Delta^{1/2} U'\$
(\$U\$ matrice orthogonale des vecteurs 
   propres de \$\Sigma\$ et \$\Delta\$ matrice diagonale des valeurs 
   propres positives ou nulles).
1. Montrer que \$F_{2:3,2:3}\$ suit une loi de Fisher \$\mathcal{F}(2,n-4)\$.
   Calculer son quantile à 95% avec la fonction `f` du sous module `scipy.stats` 
   (méthode `isf`).


```{code-cell} python
f.isf(0.05, 2, modele3.nobs - 2)
```




    3.1907273359284987



2. Déduire que la région de confiance pour \$\beta_{1:2}\$ est l'image d'un 
   disque par une matrice à déterminer. Calculer cette matrice
   en python \[méthode `cov_params` pour l'instance `modele3`, fonctions `eigh` du sous module `np.linalg`, 
   `np.matmul`, `np.diag`, `np.sqrt`\]


```{code-cell} python
hatSigma = modele3.cov_params().iloc[1:3,1:3]
valpr,vectpr = np.linalg.eigh(hatSigma)
hatSigmademi = np.matmul(vectpr, np.diag(np.sqrt(valpr)))
```

3. Construire 500 points sur le cercle
   \[`cos` et `sin` de `np`\]


```{code-cell} python
theta = np.linspace(0, 2 * math.pi, 500)
rho = (2 * f.isf(0.05, 2, modele3.nobs - 2))**0.5
x = rho * np.cos(theta)
y = rho * np.sin(theta)
XX = np.array([x, y])
```

4. Transformer ces points via la matrice donnant ainsi l'ellipse de confiance.


```{code-cell} python
ZZ = np.add(np.matmul(hatSigmademi, XX).transpose(), np.array(modele3.params[1:3]))
```

5. Tracer l'ellipse 
   \[`plt.fill` (pour l'ellipse), `plt.plot` (pour le centre)\]


```{code-cell} python
plt.fill(ZZ[:, 0], ZZ[:, 1], facecolor='yellow', edgecolor='black', linewidth=1)
plt.plot(modele3.params[1], modele3.params[2], "+")
```




    [<matplotlib.lines.Line2D at 0x7f86828a24c0>]




    
![png](p1_6_1_lab_correction_inference_fr_files/p1_6_1_lab_correction_inference_fr_30_1.png)
    


### IC univariés
Ajouter le « rectangle de confiance » issu des 2 IC univariés à l'ellipes en
récupérant l'`Axe` via `plt.gca()`, en créant le `patch` rectangle avec
`matplotlib.patches.Rectangle` et en l'ajoutant avec `ax.add_artist`.


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




    <matplotlib.patches.Rectangle at 0x7f86828ea340>




    
![png](p1_6_1_lab_correction_inference_fr_files/p1_6_1_lab_correction_inference_fr_32_1.png)
    


On voit que 2 IC univariés (et donc considérer que les variables sont indépendantes ne convient pas et des points qui sont dans la région de confiance rectangulaire et ne sont pas dans l'ellipse de confiance et vice-versa…

# IC et bootstrap
L'objectif de ce TD est de construire un IC grâce au Bootstrap.

### Importation des données
Importer les données d'ozone dans le DataFrame pandas `ozone`
\[`read_csv` de `numpy`\]


```{code-cell} python
ozone = pd.read_csv("data/ozone.txt", header=0, sep=";")
```

### Modèle à 3 variables
Estimer un modèle de régression expliquant
le maximum de la concentration en ozone du jour (variable `O3`) par 
- la température à midi notée `T12`
- la vitesse du vent sur l'axe Est-Ouest notée `Vx`
- la nébulosité à midi `Ne12`
avec comme toujours la constante.
\[`ols` de `smf`, méthode `fit` de la classe `OLS` et 
méthode `summary` pour l'instance/modèle ajusté\]


```{code-cell} python
modele3 = smf.ols("O3 ~ T12 + Vx + Ne12",data=ozone).fit()
```

### Bootstrap et IC 

#### Calcul du modèle empirique:  $\hat Y$ et $\hat\varepsilon$
Stocker les résidus dans l'objet `residus`et les ajustememt dans `ychap`


```{code-cell} python
ychap = modele3.fittedvalues
residus = modele3.resid
```

#### Géneration d'échantillon bootstrap
Le modèle de régression générant les $Y_i$ ($i\in\{1,\cdots,n\}$) est le suivant
$$
Y_i = \beta_1 +  \beta_2 X_{i2} +   \beta_3 X_{i3} +   \beta_4 X_{i4} +  \varepsilon_i$
$$
où la loi de $\varepsilon_i$ (notée $F$) est inconnue. 

Si on avait par exemple  $B=1000$ échantillons alors on pourrait
estimer  $B$ fois $\beta$ et voir à partir de ces $B$ estimations
la variabilité des $\hat\beta$ et en tirer des quantiles empiriques de
niveau $\alpha/2$ et $1-\alpha/2$ et donc un intervalle de confiance.


Bien entendu nous n'avons qu'un seul $n$-échantillon et si nous souhaitons
générer $B$ échantillon il faudrait connaitre $\beta$ et $F$. L'idée du bootstrap
est de remplacer $\beta$ et $F$ inconnus par $\hat\beta$ (l'estimateur des MCO)
et $\hat F$ (un estimateur de $F$), de générer $B$ échantillons puis de calculer
les  $B$ estimations $\hat\beta^*$
la variabilité des $\hat\beta^*$ et en tirer des quantiles empiriques de
niveau $\alpha/2$ et $1-\alpha/2$ et donc un intervalle de confiance.

Générons  $B=1000$ échantillons bootstrap. 
1. Pour chaque valeur de $b\in\{1,\cdots,B\}$ tirer indépendamment avec remise 
   parmi les résidus de la régression $n$ valeurs. 
   Notons $\hat\varepsilon^{(b)}$ le vecteur résultant de ces tirages;
2. Ajouter ces résidus à l'ajustement $\hat Y$ pour obtenir un nouvel
   échantillon $Y^*$. Avec les données $X, Y^*$ obtenir l'estimation par
   MCO $\hat\beta^{(b)}$;
3. Stocker la valeur $\hat\beta^{(b)}$ dans la ligne $b$ de 
   l'array numpy `COEFF`.
   \[créer une instance de générateur aléatoire `np.random.default_rng`, et
   utiliser la méthode `randint` sur cette instance ; créer une copie des 
   colonnes adéquates de `ozone` via la méthode `copy` afin d'utiliser
   `smf.ols` et remplir ce DataFrame avec l'échantillon.\]



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




    (1000, 4)



### IC bootstrap
A partir des $B=1000$ valeurs $\hat\beta^{(b)}$ proposer un IC à 95%.
\[`np.quantile`\]


```{code-cell} python
pd.DataFrame(np.quantile(COEFF, [0.025, 0.975], axis=0).T)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>58.651154</td>
      <td>108.561817</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.442724</td>
      <td>2.315243</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.189338</td>
      <td>0.808281</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-6.847673</td>
      <td>-2.795357</td>
    </tr>
  </tbody>
</table>
</div>



# Modélisation de la hauteur d'eucalyptus

### Importation
Importer les données d'eucalytus dans le DataFrame pandas `eucalypt`
\[`read_csv` de `numpy`\]



```{code-cell} python
eucalypt = pd.read_csv("data/eucalyptus.txt", header=0, sep=";")
```

### Deux régressions
Nous avons déjà lors des précédents TP effectué plusieurs modélisations.
Pour les modélisations à une variable, celle choisie était celle avec la 
racine carrée (voir TP régression simple). Ensuite nous avons introduit la
régression multiple et nous allons maintenant comparer ces deux modèles.

1. Effectuer une régression linéaire simple où la racine carrée de `circ`
   est  la variable explicative et `ht` la variable à expliquer.
   Stocker le résultat dans l'objet `regsqrt`.
2. Effectuer une régression linéaire multiple où la racine carrée de `circ`
   et la variable `circ` sont  les variables explicatives et `ht` la variable à expliquer.
   Stocker le résultat dans l'objet `reg`.
\[`ols` de `smf`, méthode `fit` de la classe `OLS`\]


```{code-cell} python
regsqrt = smf.ols('ht~I(np.sqrt(circ))', data=eucalypt).fit()
reg = smf.ols('ht~I(np.sqrt(circ)) + circ', data=eucalypt).fit()
```

### Comparaison

1. Comparer ces deux modèles via un test $T$ \[méthode `summary`\]


```{code-cell} python
reg.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>ht</td>        <th>  R-squared:         </th> <td>   0.792</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.792</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   2718.</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 12 Jul 2023</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>16:06:37</td>     <th>  Log-Likelihood:    </th> <td> -2208.5</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1429</td>      <th>  AIC:               </th> <td>   4423.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  1426</td>      <th>  BIC:               </th> <td>   4439.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>            <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>        <td>  -24.3520</td> <td>    2.614</td> <td>   -9.314</td> <td> 0.000</td> <td>  -29.481</td> <td>  -19.223</td>
</tr>
<tr>
  <th>I(np.sqrt(circ))</th> <td>    9.9869</td> <td>    0.780</td> <td>   12.798</td> <td> 0.000</td> <td>    8.456</td> <td>   11.518</td>
</tr>
<tr>
  <th>circ</th>             <td>   -0.4829</td> <td>    0.058</td> <td>   -8.336</td> <td> 0.000</td> <td>   -0.597</td> <td>   -0.369</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 3.015</td> <th>  Durbin-Watson:     </th> <td>   0.947</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.221</td> <th>  Jarque-Bera (JB):  </th> <td>   2.897</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.097</td> <th>  Prob(JB):          </th> <td>   0.235</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.103</td> <th>  Cond. No.          </th> <td>4.41e+03</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 4.41e+03. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



La ligne `circ` du tableau donne l'estimation du coefficient $\hat\beta_3$, 
l'écart-type estimé du coefficient, la valeur de la statistique 
$t$ du test $\mathrm{H}_0: \beta_3=0$ contre $\mathrm{H}_1: \beta_3\neq 0$
qui vaut ici $-8.336$ et sa probabilité critique quasi nulle. Nous repoussons donc
$\mathrm{H}_0$ et le modèle `reg` semble meilleur.

2. Comparer ces deux modèles via un test $F$ \[`stats.anova_lm` du sous module
`statsmodels.api`\]


```{code-cell} python
import statsmodels.api as sm
sm.stats.anova_lm(regsqrt,reg)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>df_resid</th>
      <th>ssr</th>
      <th>df_diff</th>
      <th>ss_diff</th>
      <th>F</th>
      <th>Pr(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1427.0</td>
      <td>1930.351780</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1426.0</td>
      <td>1840.656131</td>
      <td>1.0</td>
      <td>89.69565</td>
      <td>69.489349</td>
      <td>1.786144e-16</td>
    </tr>
  </tbody>
</table>
</div>



Nous retrouvons les mêmes résultats que précédemment (car $F=t^2$)

# L'âge a t-il une influence sur le temps libre ?

Une enquête a été conduite sur 40 individus afin d'étudier le lien
entre le temps libre (estimé par l'enquêté comme le temps, en nombre
d'heures par jour, disponible pour soi) et l'âge. Les résultats de
cette enquête sont contenus dans le fichier
`temps_libre.csv`. Nous nous proposons de savoir si ces
deux variables sont liées.


1. Quel est le type des variables ?


```{code-cell} python
tpslibre = pd.read_csv("data/temps_libre.csv", header=0, sep=";")
tpslibre.columns = [ "age", "tempslibre" ]
tpslibre.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>tempslibre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>40.0000</td>
      <td>40.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>46.7000</td>
      <td>4.265000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>21.3628</td>
      <td>2.106102</td>
    </tr>
    <tr>
      <th>min</th>
      <td>20.0000</td>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>29.0000</td>
      <td>2.300000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>40.5000</td>
      <td>4.800000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>68.0000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>83.0000</td>
      <td>7.700000</td>
    </tr>
  </tbody>
</table>
</div>



Les deux variables sont quantitatives continues. Nous changeons les noms car le '.' est mal géré dans les formules…

2. Comment calcule t-on le lien (le plus commun) entre ces deux variables ?


```{code-cell} python
tpslibre.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>tempslibre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>1.00000</td>
      <td>0.04615</td>
    </tr>
    <tr>
      <th>tempslibre</th>
      <td>0.04615</td>
      <td>1.00000</td>
    </tr>
  </tbody>
</table>
</div>



La mesure du lien est la corrélation linéaire (dont le carré vaut le R2), ici elle est très faible et tend à montrer qu'il n'y a pas de lien linéaire entre les 2 variables.

3. Comment teste-t-on si l'âge à une influence sur le temps libre 
   à l'aide de la régression ? Effectuer ce test et conclure.


```{code-cell} python
reg = smf.ols("tempslibre~1+age", data=tpslibre).fit()
reg.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>tempslibre</td>    <th>  R-squared:         </th> <td>   0.002</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>  -0.024</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td> 0.08111</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 12 Jul 2023</td> <th>  Prob (F-statistic):</th>  <td> 0.777</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>16:06:37</td>     <th>  Log-Likelihood:    </th> <td> -86.002</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    40</td>      <th>  AIC:               </th> <td>   176.0</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    38</td>      <th>  BIC:               </th> <td>   179.4</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>    4.0525</td> <td>    0.819</td> <td>    4.950</td> <td> 0.000</td> <td>    2.395</td> <td>    5.710</td>
</tr>
<tr>
  <th>age</th>       <td>    0.0045</td> <td>    0.016</td> <td>    0.285</td> <td> 0.777</td> <td>   -0.028</td> <td>    0.037</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 8.585</td> <th>  Durbin-Watson:     </th> <td>   0.569</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.014</td> <th>  Jarque-Bera (JB):  </th> <td>   2.949</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.305</td> <th>  Prob(JB):          </th> <td>   0.229</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 1.818</td> <th>  Cond. No.          </th> <td>    125.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



La ligne `age` du tableau donne l'estimation du coefficient $\hat\beta_2$, 
l'écart-type estimé du coefficient, la valeur de la statistique 
$t$ du test $\mathrm{H}_0: \beta_2=0$ contre $\mathrm{H}_1: \beta_2\neq 0$
qui vaut ici $0.285$ et sa probabilité critique qui est de 0.777. Nous conservons donc
$\mathrm{H}_0$ et il ne semble pas y avoir de lien linéaire.

4. Représentez les données et discuter du bien fondé du test précédent.


```{code-cell} python
plt.plot(tpslibre.age, tpslibre.tempslibre, "*")
```




    [<matplotlib.lines.Line2D at 0x7f86812d25e0>]




    
![png](p1_6_1_lab_correction_inference_fr_files/p1_6_1_lab_correction_inference_fr_72_1.png)
    


Clairement on observe 2 régimes: entre 30 et 60 ans, peu de temps libre et avant 30 ou après 60 plus de temps libre. Il y a une influence de l'âge mais pas linéaire (plutôt constante par morceaux). Le test précédent est inadapté.

# L'obésité a t-elle une influence sur la pression sanguine ?
Une enquête a été conduite sur 102 individus afin d'étudier le lien
entre l'obésité (estimée par le ratio du poids de la personne sur le poids idéal obtenu dans la \og New York Metropolitan Life Tables\fg{}) et la pression sanguine en mm de mercure. Les résultats de
cette enquête sont contenus dans le fichier
`obesite.csv` (dans Fun Campus les données sont dans le répertoire `data/`). Nous nous proposons de savoir si ces
deux variables sont liées.

1. Quel est le type des variables ?


```{code-cell} python
obesite = pd.read_csv("data/obesite.csv", header=0, sep=";")
obesite.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>obesite</th>
      <th>pression</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>102.000000</td>
      <td>102.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.313039</td>
      <td>127.019608</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.257839</td>
      <td>18.184413</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.810000</td>
      <td>94.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.142500</td>
      <td>116.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.285000</td>
      <td>124.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.430000</td>
      <td>137.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.390000</td>
      <td>208.000000</td>
    </tr>
  </tbody>
</table>
</div>



Les variables sont quantitatives.

2. Comment calcule t-on le lien (le plus commun) entre ces deux variables ?

Il s'agit de la corrélation linéaire qui vaut ici


```{code-cell} python
obesite.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>obesite</th>
      <th>pression</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>obesite</th>
      <td>1.000000</td>
      <td>0.326139</td>
    </tr>
    <tr>
      <th>pression</th>
      <td>0.326139</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



La corrélation semble modérée.

3. Comment teste-t-on si l'obésite à une influence sur la 
   pression à l'aide de la régression ? Effectuer ce test 
   et conclure.

Nous avons vu à l'exercice précédent qu'il semble raisonnable de faire d'abord le graphique pour voir si le modèle de régression linéaire est adapté:


```{code-cell} python
plt.plot(obesite.obesite, obesite.pression, "o")
```




    [<matplotlib.lines.Line2D at 0x7fa7d07b4f10>]




    
![png](p1_6_1_lab_correction_inference_fr_files/p1_6_1_lab_correction_inference_fr_84_1.png)
    


Même si les points sont pas vraiment sur une droite nous pouvons quand même nous dire que ce modèle peut grossièrement convenir. Effectuons la régression simple et un test $t$ de nullité de pente


```{code-cell} python
reg = smf.ols("pression~1+obesite", data=obesite).fit()
reg.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>pression</td>     <th>  R-squared:         </th> <td>   0.106</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.097</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   11.90</td>
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 19 Jun 2023</td> <th>  Prob (F-statistic):</th> <td>0.000822</td>
</tr>
<tr>
  <th>Time:</th>                 <td>16:21:35</td>     <th>  Log-Likelihood:    </th> <td> -434.35</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   102</td>      <th>  AIC:               </th> <td>   872.7</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   100</td>      <th>  BIC:               </th> <td>   878.0</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>   96.8179</td> <td>    8.920</td> <td>   10.855</td> <td> 0.000</td> <td>   79.122</td> <td>  114.514</td>
</tr>
<tr>
  <th>obesite</th>   <td>   23.0014</td> <td>    6.667</td> <td>    3.450</td> <td> 0.001</td> <td>    9.774</td> <td>   36.229</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>38.449</td> <th>  Durbin-Watson:     </th> <td>   1.529</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  84.291</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 1.460</td> <th>  Prob(JB):          </th> <td>4.97e-19</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 6.362</td> <th>  Cond. No.          </th> <td>    10.8</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



La ligne `obesite` du tableau donne l'estimation du coefficient $\hat\beta_2$, 
l'écart-type estimé du coefficient, la valeur de la statistique 
$t$ du test $\mathrm{H}_0: \beta_2=0$ contre $\mathrm{H}_1: \beta_2\neq 0$
qui vaut ici $3.45$ et sa probabilité critique qui est de 0.001. Nous repoussons donc
$\mathrm{H}_0$ et il semble  y avoir un lien linéaire (assez grossier ; ce n'est pas avec ce modèle que l'on va prédire la pression sanguine).
