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

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa Bedin<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

# Modules python
Importer les modules pandas (comme `pd`) numpy (commme `np`)
matplotlib.pyplot (comme  `plt`), statsmodels.formula.api (comme `smf`)
et statsmodels.api (comme `sm`)


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
```

# Régression multiple (modèle du cours)

### Importation des données
Importer les données d'ozone dans le DataFrame pandas `ozone`


```python
ozone = pd.read_csv("data/ozone.txt", header=0, sep=";")
```

### Estimation du modèle du cours
Nous sommes intéressé par batir un modèle de prévision de l'ozone par 
une régression multiple. Ce régression expliquera
le maximum de la concentration en ozone du jour (variable `O3`) par 
- la température à midi notée `T12`
- la nébulosité à midi notée `Ne12`
- la vitesse du vent sur l'axe Est-Ouest notée `Vx`
Traditionnellement on introduit toujours la constante (le faire ici aussi).
Estimer le modèle par MCO et faire le résumé.


```python
reg = smf.ols('O3~T12+Ne12+Vx', data=ozone).fit()
reg.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>O3</td>        <th>  R-squared:         </th> <td>   0.682</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.661</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   32.87</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 12 Jul 2023</td> <th>  Prob (F-statistic):</th> <td>1.66e-11</td>
</tr>
<tr>
  <th>Time:</th>                 <td>15:42:24</td>     <th>  Log-Likelihood:    </th> <td> -200.50</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    50</td>      <th>  AIC:               </th> <td>   409.0</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    46</td>      <th>  BIC:               </th> <td>   416.7</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>     <td> </td>   
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
  <th>Intercept</th> <td>   84.5473</td> <td>   13.607</td> <td>    6.214</td> <td> 0.000</td> <td>   57.158</td> <td>  111.936</td>
</tr>
<tr>
  <th>T12</th>       <td>    1.3150</td> <td>    0.497</td> <td>    2.644</td> <td> 0.011</td> <td>    0.314</td> <td>    2.316</td>
</tr>
<tr>
  <th>Ne12</th>      <td>   -4.8934</td> <td>    1.027</td> <td>   -4.765</td> <td> 0.000</td> <td>   -6.961</td> <td>   -2.826</td>
</tr>
<tr>
  <th>Vx</th>        <td>    0.4864</td> <td>    0.168</td> <td>    2.903</td> <td> 0.006</td> <td>    0.149</td> <td>    0.824</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 0.211</td> <th>  Durbin-Watson:     </th> <td>   1.758</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.900</td> <th>  Jarque-Bera (JB):  </th> <td>   0.411</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.050</td> <th>  Prob(JB):          </th> <td>   0.814</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.567</td> <th>  Cond. No.          </th> <td>    148.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



### Résidus \$\varepsilon\$
Afficher le graphique des résidus (attribut `resid` du modèle estimé)
(avec \$\hat y\$ en abscisse et \$\varepsilon\$ en ordonnée).


```python
plt.plot(reg.predict(), reg.resid ,"+")
```




    [<matplotlib.lines.Line2D at 0x7fd7674f31c0>]




    
![png](p1_5_3_lab_correction_residus_fr_files/p1_5_3_lab_correction_residus_fr_8_1.png)
    


Aucune structuration des résidus visible. L'épaisseur (écart-type) des points semble 
un peu toujours la même mais ces résidus par construction n'ont pas la même variance, 
donc il est délicat de conclure sur l'hypothèse \$\mathrm{V}(\varepsilon_i)=\sigma^2\$.
De plus l'échelle des ordonnées dépend du problème, donc ces résidus sont peu 
praticables.

### Résidus \$\varepsilon\$
Afficher le graphique des résidus studentisés par validation croisée (avec \$\hat y\$ en abscisse et 
\$\varepsilon\$ en ordonnée). Pour cela utiliser la fonction/méthode `get_influence` 
qui renverra un objet (que l'on nommera `infl`) avec un attribut `resid_studentized_external`
contenant les résidus souhaités.


```python
infl = reg.get_influence()
plt.plot(reg.predict(), infl.resid_studentized_external,"+")
```




    [<matplotlib.lines.Line2D at 0x7fd7674baee0>]




    
![png](p1_5_3_lab_correction_residus_fr_files/p1_5_3_lab_correction_residus_fr_11_1.png)
    


Aucune structuration des résidus visible. L'épaisseur (écart-type) des points semble 
un peu toujours la même donc l'hypothèse \$\mathrm{V}(\varepsilon_i)=\sigma^2\$ semble
correcte. Aucun point en dehors de -2,2 donc pas d'individus aberrant.

### Points leviers
Représenter les \$h_{ii}\$ grâce à `plt.stem` en fonction du numéro de ligne


```python
index=np.arange(1, ozone.shape[0]+1)
plt.stem(index, infl.hat_matrix_diag)
```




    <StemContainer object of 3 artists>




    
![png](p1_5_3_lab_correction_residus_fr_files/p1_5_3_lab_correction_residus_fr_14_1.png)
    


Aucun  \$h_{ii}\$ notablement plus grand 
que les autres donc le plan d'expérience est correct.

# R²
Nous sommes intéressé par batir un modèle de prévision de l'ozone par 
une régression multiple. Cependant nous ne savons pas trop a priori
quelles sont les variables utiles. Batissons plusieurs modèles.

### Estimation du modèle du cours
Ce régression expliquera
le maximum de la concentration en ozone du jour (variable `O3`) par 
- la température à midi notée `T12`
- la nébulosité à midi notée `Ne12`
- la vitesse du vent sur l'axe Est-Ouest notée `Vx`
Traditionnellement on introduit toujours la constante (le faire ici aussi).
Estimer le modèle par MCO et faire le résumé.


```python
reg3 = smf.ols('O3~T12+Ne12+Vx', data=ozone).fit()
reg3.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>O3</td>        <th>  R-squared:         </th> <td>   0.682</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.661</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   32.87</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 12 Jul 2023</td> <th>  Prob (F-statistic):</th> <td>1.66e-11</td>
</tr>
<tr>
  <th>Time:</th>                 <td>15:43:23</td>     <th>  Log-Likelihood:    </th> <td> -200.50</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    50</td>      <th>  AIC:               </th> <td>   409.0</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    46</td>      <th>  BIC:               </th> <td>   416.7</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>     <td> </td>   
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
  <th>Intercept</th> <td>   84.5473</td> <td>   13.607</td> <td>    6.214</td> <td> 0.000</td> <td>   57.158</td> <td>  111.936</td>
</tr>
<tr>
  <th>T12</th>       <td>    1.3150</td> <td>    0.497</td> <td>    2.644</td> <td> 0.011</td> <td>    0.314</td> <td>    2.316</td>
</tr>
<tr>
  <th>Ne12</th>      <td>   -4.8934</td> <td>    1.027</td> <td>   -4.765</td> <td> 0.000</td> <td>   -6.961</td> <td>   -2.826</td>
</tr>
<tr>
  <th>Vx</th>        <td>    0.4864</td> <td>    0.168</td> <td>    2.903</td> <td> 0.006</td> <td>    0.149</td> <td>    0.824</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 0.211</td> <th>  Durbin-Watson:     </th> <td>   1.758</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.900</td> <th>  Jarque-Bera (JB):  </th> <td>   0.411</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.050</td> <th>  Prob(JB):          </th> <td>   0.814</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.567</td> <th>  Cond. No.          </th> <td>    148.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



### Estimation du modèle du cours
Ce régression expliquera
le maximum de la concentration en ozone du jour (variable `O3`) par 
- la température à six heures notée `T12`
- la température à midi notée `T15`
- la nébulosité à midi notée `Ne12`
- la vitesse du vent sur l'axe Est-Ouest notée `Vx`
- le maximum du jour d'avant/la veille `O3v`
Traditionnellement on introduit toujours la constante (le faire ici aussi).
Estimer le modèle par MCO et faire le résumé.


```python
ozone.head()
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
      <th>Date</th>
      <th>O3</th>
      <th>T12</th>
      <th>T15</th>
      <th>Ne12</th>
      <th>N12</th>
      <th>S12</th>
      <th>E12</th>
      <th>W12</th>
      <th>Vx</th>
      <th>O3v</th>
      <th>nebulosite</th>
      <th>vent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19960422</td>
      <td>63.6</td>
      <td>13.4</td>
      <td>15.0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>9.35</td>
      <td>95.6</td>
      <td>NUAGE</td>
      <td>EST</td>
    </tr>
    <tr>
      <th>1</th>
      <td>19960429</td>
      <td>89.6</td>
      <td>15.0</td>
      <td>15.7</td>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5.40</td>
      <td>100.2</td>
      <td>SOLEIL</td>
      <td>NORD</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19960506</td>
      <td>79.0</td>
      <td>7.9</td>
      <td>10.1</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>19.30</td>
      <td>105.6</td>
      <td>NUAGE</td>
      <td>EST</td>
    </tr>
    <tr>
      <th>3</th>
      <td>19960514</td>
      <td>81.2</td>
      <td>13.1</td>
      <td>11.7</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12.60</td>
      <td>95.2</td>
      <td>NUAGE</td>
      <td>NORD</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19960521</td>
      <td>88.0</td>
      <td>14.1</td>
      <td>16.0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>-20.30</td>
      <td>82.8</td>
      <td>NUAGE</td>
      <td>OUEST</td>
    </tr>
  </tbody>
</table>
</div>




```python
reg6 = smf.ols('O3~T12+T15+Ne12+Vx+O3v', data=ozone).fit()
reg6.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>O3</td>        <th>  R-squared:         </th> <td>   0.733</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.702</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   24.13</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 12 Jul 2023</td> <th>  Prob (F-statistic):</th> <td>1.34e-11</td>
</tr>
<tr>
  <th>Time:</th>                 <td>15:50:51</td>     <th>  Log-Likelihood:    </th> <td> -196.15</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    50</td>      <th>  AIC:               </th> <td>   404.3</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    44</td>      <th>  BIC:               </th> <td>   415.8</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     5</td>      <th>                     </th>     <td> </td>   
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
  <th>Intercept</th> <td>   61.4744</td> <td>   15.088</td> <td>    4.074</td> <td> 0.000</td> <td>   31.067</td> <td>   91.882</td>
</tr>
<tr>
  <th>T12</th>       <td>    0.4675</td> <td>    1.459</td> <td>    0.320</td> <td> 0.750</td> <td>   -2.474</td> <td>    3.409</td>
</tr>
<tr>
  <th>T15</th>       <td>    0.6306</td> <td>    1.409</td> <td>    0.447</td> <td> 0.657</td> <td>   -2.210</td> <td>    3.471</td>
</tr>
<tr>
  <th>Ne12</th>      <td>   -3.9958</td> <td>    1.017</td> <td>   -3.927</td> <td> 0.000</td> <td>   -6.046</td> <td>   -1.945</td>
</tr>
<tr>
  <th>Vx</th>        <td>    0.3282</td> <td>    0.168</td> <td>    1.955</td> <td> 0.057</td> <td>   -0.010</td> <td>    0.667</td>
</tr>
<tr>
  <th>O3v</th>       <td>    0.2631</td> <td>    0.093</td> <td>    2.826</td> <td> 0.007</td> <td>    0.075</td> <td>    0.451</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 0.238</td> <th>  Durbin-Watson:     </th> <td>   1.856</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.888</td> <th>  Jarque-Bera (JB):  </th> <td>   0.425</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.103</td> <th>  Prob(JB):          </th> <td>   0.809</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.598</td> <th>  Cond. No.          </th> <td>    759.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



### Comparer les R2
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

### Importer les données
Vous avez une variable à expliquer \$Y\$
et quatre variables explicatives dans le fichier `tprespartiel.dta`


```python
tp = pd.read_csv("tprespartiel.dta", header=0, sep=";")
tp.head()
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
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
      <th>X4</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.167625</td>
      <td>0.247608</td>
      <td>0.981144</td>
      <td>-0.365881</td>
      <td>-5.504742</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.427154</td>
      <td>0.662147</td>
      <td>0.394141</td>
      <td>0.438178</td>
      <td>-6.180432</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.406420</td>
      <td>0.809686</td>
      <td>0.639263</td>
      <td>-0.087607</td>
      <td>-5.997251</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.061812</td>
      <td>0.420397</td>
      <td>0.437492</td>
      <td>0.468991</td>
      <td>-10.555818</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.965881</td>
      <td>0.006918</td>
      <td>0.141291</td>
      <td>0.302681</td>
      <td>13.689773</td>
    </tr>
  </tbody>
</table>
</div>



### Estimation
Estimer par MCO les paramètres du modèle \$Y_i=\beta_0 + \beta_1 X_{i,1}+\cdots+
\beta_4 X_{i,4} + \varepsilon_i.\$
[`ols` de `smf`, méthode `fit` de la classe `OLS` et 
méthode `summary` pour l'instance/modèle ajusté]


```python
reg = smf.ols("Y~X1+X2+X3+X4", data=tp).fit()
```

### Analyser les résidus partiels
Que pensez-vous des résultats ?
[`plot_ccpr_grid` du sous module `sm.graphics`], les résidus partiels sont
appelés "Component-Component plus Residual"
(CCPR) dans le module statsmodels…


```python
sm.graphics.plot_ccpr_grid(reg)
```




    
![png](p1_5_3_lab_correction_residus_fr_files/p1_5_3_lab_correction_residus_fr_30_0.png)
    




    
![png](p1_5_3_lab_correction_residus_fr_files/p1_5_3_lab_correction_residus_fr_30_1.png)
    


De manière évidente le graphique pour la variable `X4` ne montre pas
des points disposés le long d'une droite ou un nuage sans structure. 
Il montre une structuration de type \$x\mapsto x^2\$

### Amélioration du modèle 
Remplacer $X_4$ par $X_5=X_4^2$ dans le modèle précédent. Que pensez-vous de
  la nouvelle modélisation ? On pourra comparer ce modèle à celui de la
  question précédente.
[`ols` de `smf`, méthode `fit` de la classe `OLS` et 
méthode `summary` pour l'instance/modèle ajusté]
On pourra utiliser les
opérations et fonctions dans les formules
(voir https://www.statsmodels.org/stable/example_formulas.html)


```python
reg2 = smf.ols("Y~X1+X2+X3+I(X4**2)", data=tp).fit()
```

### Analyser les résidus partiels
Analyser les résidus partiels du nouveau modèle et constater
qu'ils semblent corrects.
[`plot_ccpr_grid` du sous module `sm.graphics`], les résidus partiels sont
appelés "Component-Component plus Residual"
(CCPR) dans le module statsmodels…


```python
sm.graphics.plot_ccpr_grid(reg2)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /var/folders/rr/0nkmqmx92m7gl5wcxl5xqctc0000gn/T/ipykernel_1478/2018296583.py in <module>
    ----> 1 sm.graphics.plot_ccpr_grid(reg2)
    

    NameError: name 'reg2' is not defined


The graphs show points with no obvious structure
or arranged along straight lines. The model would appear to be correct. We can compare 
compare them (same number of variables) by R2


```python
reg.rsquared, reg2.rsquared
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /var/folders/rr/0nkmqmx92m7gl5wcxl5xqctc0000gn/T/ipykernel_1478/2703078329.py in <module>
    ----> 1 reg.rsquared, reg2.rsquared
    

    NameError: name 'reg2' is not defined



```python

```




    0.9966109930897685



et le R2 de la seconde modélisation apparait meilleur.

Faire le même travail pour `tp2bisrespartiel`.


```python
tp = pd.read_csv("tprespartiel.dta", header=0, sep=";")
tp.head()
reg = smf.ols("Y~X1+X2+X3+X4", data=tp).fit()
sm.graphics.plot_ccpr_grid(reg)
```

Nous voyons clairement une sinusoïde de type \$\sin(-2\pi X_4)\$ 
sur le dernier graphique. Changeons \$X_4\$


```python
reg2 = smf.ols("Y~X1+X2+X3+I(np.sin(-2*np.pi*X4))", data=tp).fit()
sm.graphics.plot_ccpr_grid(reg2)
```




    
![png](p1_5_3_lab_correction_residus_fr_files/p1_5_3_lab_correction_residus_fr_43_0.png)
    




    
![png](p1_5_3_lab_correction_residus_fr_files/p1_5_3_lab_correction_residus_fr_43_1.png)
    


Là encore les graphiques deviennent corrects et nous pouvons comparer 
les R2 et constater que la seconde modélisation améliore le R2.


```python
reg.rsqared
```


```python
reg2.rsquared
```
