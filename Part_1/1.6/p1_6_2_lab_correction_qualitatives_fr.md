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
  title: 'Correction du TP variables qualitatives'
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
matplotlib.pyplot (comme  `plt`), statsmodels.formula.api (comme `smf`)
et statsmodels.api (comme `sm`)


```{code-cell} python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
```

# Variables qualitatives et quantitatives pour l'ozone

### Importation
Importer les données `ozonecomplet.csv` et transformer les deux dernières 
variables en variables qualitatives et faites un résumé numérique par variable
\[méthode `astype` sur la colonne du DataFrame et
méthode `describe` sur l'instance DataFrame\]


```{code-cell} python
ozone = pd.read_csv("data/ozonecomplet.csv", header=0, sep=";")
ozone = ozone.drop(['nomligne'], axis=1)
ozone.Ne = ozone.Ne.astype("category")
ozone.Dv = ozone.Dv.astype("category")
ozone.describe(include="all")
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
      <th>O3</th>
      <th>T9</th>
      <th>T12</th>
      <th>T15</th>
      <th>Ne9</th>
      <th>Ne12</th>
      <th>Ne15</th>
      <th>Vx9</th>
      <th>Vx12</th>
      <th>Vx15</th>
      <th>O3v</th>
      <th>Dv</th>
      <th>Ne</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>112.000000</td>
      <td>112.000000</td>
      <td>112.000000</td>
      <td>112.000000</td>
      <td>112.000000</td>
      <td>112.000000</td>
      <td>112.000000</td>
      <td>112.000000</td>
      <td>112.000000</td>
      <td>112.000000</td>
      <td>112.000000</td>
      <td>112</td>
      <td>112</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>O</td>
      <td>s</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>50</td>
      <td>69</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>90.303571</td>
      <td>18.360714</td>
      <td>21.526786</td>
      <td>22.627679</td>
      <td>4.928571</td>
      <td>5.017857</td>
      <td>4.830357</td>
      <td>-1.214346</td>
      <td>-1.611004</td>
      <td>-1.690683</td>
      <td>90.571429</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>28.187225</td>
      <td>3.122726</td>
      <td>4.042321</td>
      <td>4.530859</td>
      <td>2.594916</td>
      <td>2.281860</td>
      <td>2.332259</td>
      <td>2.632742</td>
      <td>2.795673</td>
      <td>2.810198</td>
      <td>28.276853</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>42.000000</td>
      <td>11.300000</td>
      <td>14.000000</td>
      <td>14.900000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-7.878500</td>
      <td>-7.878500</td>
      <td>-9.000000</td>
      <td>42.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>70.750000</td>
      <td>16.200000</td>
      <td>18.600000</td>
      <td>19.275000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>-3.276450</td>
      <td>-3.564700</td>
      <td>-3.939200</td>
      <td>71.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>81.500000</td>
      <td>17.800000</td>
      <td>20.550000</td>
      <td>22.050000</td>
      <td>6.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>-0.866000</td>
      <td>-1.879400</td>
      <td>-1.549650</td>
      <td>82.500000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>106.000000</td>
      <td>19.925000</td>
      <td>23.550000</td>
      <td>25.400000</td>
      <td>7.000000</td>
      <td>7.000000</td>
      <td>7.000000</td>
      <td>0.694600</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>106.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>166.000000</td>
      <td>27.000000</td>
      <td>33.500000</td>
      <td>35.500000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>5.196200</td>
      <td>6.577800</td>
      <td>5.000000</td>
      <td>166.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### Premier modèle
 Effectuer une régression avec comme variables explicatives `T12`,`Ne` et `Dv`,
  combien estime t on de paramètres ?


```{code-cell} python
reg = smf.ols("O3~T12+Ne+Dv", data=ozone).fit()
reg.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>O3</td>        <th>  R-squared:         </th> <td>   0.642</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.625</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   38.04</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 20 Jun 2023</td> <th>  Prob (F-statistic):</th> <td>3.51e-22</td>
</tr>
<tr>
  <th>Time:</th>                 <td>18:42:24</td>     <th>  Log-Likelihood:    </th> <td> -474.83</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   112</td>      <th>  AIC:               </th> <td>   961.7</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   106</td>      <th>  BIC:               </th> <td>   978.0</td>
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
  <th>Intercept</th> <td>  -20.2681</td> <td>   11.751</td> <td>   -1.725</td> <td> 0.087</td> <td>  -43.565</td> <td>    3.029</td>
</tr>
<tr>
  <th>Ne[T.s]</th>   <td>    8.0648</td> <td>    3.854</td> <td>    2.092</td> <td> 0.039</td> <td>    0.423</td> <td>   15.707</td>
</tr>
<tr>
  <th>Dv[T.N]</th>   <td>   -0.2925</td> <td>    6.473</td> <td>   -0.045</td> <td> 0.964</td> <td>  -13.125</td> <td>   12.540</td>
</tr>
<tr>
  <th>Dv[T.O]</th>   <td>   -5.2526</td> <td>    6.121</td> <td>   -0.858</td> <td> 0.393</td> <td>  -17.389</td> <td>    6.884</td>
</tr>
<tr>
  <th>Dv[T.S]</th>   <td>   -3.0645</td> <td>    6.632</td> <td>   -0.462</td> <td> 0.645</td> <td>  -16.213</td> <td>   10.084</td>
</tr>
<tr>
  <th>T12</th>       <td>    5.0450</td> <td>    0.480</td> <td>   10.509</td> <td> 0.000</td> <td>    4.093</td> <td>    5.997</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 1.587</td> <th>  Durbin-Watson:     </th> <td>   1.283</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.452</td> <th>  Jarque-Bera (JB):  </th> <td>   1.264</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.257</td> <th>  Prob(JB):          </th> <td>   0.531</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.083</td> <th>  Cond. No.          </th> <td>    185.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



### Résumé du modèle
Où sont passés les coefficients associés au vent d'Est et au temps nuageux
dans le résumé du modèle ?

Il s'agit des modalités de références (première par ordre alphabétique).
La constante (intercept) correspond à  une température à midi (`T12`)
de 0 degré avec un vent d'Est et un temps nuageux. Quand le temps correspond
à cette définition alors on prévoit -20 microgramme par m3 d'ozone.
Le modèle dans cette plage n'est pas adapté (car nous n'avons pas de données)

### Changement de modalité de référence
Changer la modalité de référence du vent pour le vent du Nord,
\[fonction `C` dans la formule de la régression, voir https://www.statsmodels.org/stable/example_formulas.html)
 `Treatment` option `reference`\].
- Vérifier que la valeur de l'intercept a changé ainsi que toutes les valeurs 
  des estimateurs des paramètres associés au vent. 
- Vérifier que les $Y$ ajustés sont les mêmes.


```{code-cell} python
reg2 = smf.ols("O3~T12+Ne+C(Dv, Treatment(reference=1))", data=ozone).fit()
reg2.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>O3</td>        <th>  R-squared:         </th> <td>   0.642</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.625</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   38.04</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 20 Jun 2023</td> <th>  Prob (F-statistic):</th> <td>3.51e-22</td>
</tr>
<tr>
  <th>Time:</th>                 <td>18:42:16</td>     <th>  Log-Likelihood:    </th> <td> -474.83</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   112</td>      <th>  AIC:               </th> <td>   961.7</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   106</td>      <th>  BIC:               </th> <td>   978.0</td>
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
                   <td></td>                     <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                          <td>  -20.5606</td> <td>    9.351</td> <td>   -2.199</td> <td> 0.030</td> <td>  -39.100</td> <td>   -2.022</td>
</tr>
<tr>
  <th>Ne[T.s]</th>                            <td>    8.0648</td> <td>    3.854</td> <td>    2.092</td> <td> 0.039</td> <td>    0.423</td> <td>   15.707</td>
</tr>
<tr>
  <th>C(Dv, Treatment(reference=1))[T.E]</th> <td>    0.2925</td> <td>    6.473</td> <td>    0.045</td> <td> 0.964</td> <td>  -12.540</td> <td>   13.125</td>
</tr>
<tr>
  <th>C(Dv, Treatment(reference=1))[T.O]</th> <td>   -4.9600</td> <td>    4.086</td> <td>   -1.214</td> <td> 0.228</td> <td>  -13.062</td> <td>    3.142</td>
</tr>
<tr>
  <th>C(Dv, Treatment(reference=1))[T.S]</th> <td>   -2.7719</td> <td>    5.147</td> <td>   -0.539</td> <td> 0.591</td> <td>  -12.976</td> <td>    7.432</td>
</tr>
<tr>
  <th>T12</th>                                <td>    5.0450</td> <td>    0.480</td> <td>   10.509</td> <td> 0.000</td> <td>    4.093</td> <td>    5.997</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 1.587</td> <th>  Durbin-Watson:     </th> <td>   1.283</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.452</td> <th>  Jarque-Bera (JB):  </th> <td>   1.264</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.257</td> <th>  Prob(JB):          </th> <td>   0.531</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.083</td> <th>  Cond. No.          </th> <td>    127.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```{code-cell} python
np.all(np.abs(reg.predict() -reg2.predict())<1e-10)
```




    True



### Regroupement de modalité
- Regroupez Est et Nord et faites un nouveau modèle. \[méthode
  `map` sur la colonne puis `astype`\]
- Quel est le modèle retenu entre celui-ci et le précédent ? 
  Proposez deux tests pour répondre à cette question.
  [`sm.stats.anova_lm`]


```{code-cell} python
Dv2 = ozone.Dv.map({"E": "E+N", "N": "E+N", "O": "O", "S": "S"}).astype("category")
ozone["Dv2"] = Dv2
reg3 = smf.ols("O3~T12+Ne+Dv2", data=ozone).fit()
reg3.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>O3</td>        <th>  R-squared:         </th> <td>   0.642</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.629</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   47.99</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 20 Jun 2023</td> <th>  Prob (F-statistic):</th> <td>4.73e-23</td>
</tr>
<tr>
  <th>Time:</th>                 <td>19:16:25</td>     <th>  Log-Likelihood:    </th> <td> -474.83</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   112</td>      <th>  AIC:               </th> <td>   959.7</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   107</td>      <th>  BIC:               </th> <td>   973.3</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     4</td>      <th>                     </th>     <td> </td>   
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
  <th>Intercept</th> <td>  -20.5912</td> <td>    9.283</td> <td>   -2.218</td> <td> 0.029</td> <td>  -38.993</td> <td>   -2.189</td>
</tr>
<tr>
  <th>Ne[T.s]</th>   <td>    8.0574</td> <td>    3.833</td> <td>    2.102</td> <td> 0.038</td> <td>    0.459</td> <td>   15.656</td>
</tr>
<tr>
  <th>Dv2[T.O]</th>  <td>   -5.0338</td> <td>    3.729</td> <td>   -1.350</td> <td> 0.180</td> <td>  -12.426</td> <td>    2.359</td>
</tr>
<tr>
  <th>Dv2[T.S]</th>  <td>   -2.8571</td> <td>    4.767</td> <td>   -0.599</td> <td> 0.550</td> <td>  -12.307</td> <td>    6.592</td>
</tr>
<tr>
  <th>T12</th>       <td>    5.0502</td> <td>    0.464</td> <td>   10.875</td> <td> 0.000</td> <td>    4.130</td> <td>    5.971</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 1.559</td> <th>  Durbin-Watson:     </th> <td>   1.284</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.459</td> <th>  Jarque-Bera (JB):  </th> <td>   1.239</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.254</td> <th>  Prob(JB):          </th> <td>   0.538</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.082</td> <th>  Cond. No.          </th> <td>    126.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```{code-cell} python
sm.stats.anova_lm(reg3, reg)
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
      <td>107.0</td>
      <td>31563.863745</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>106.0</td>
      <td>31563.255456</td>
      <td>1.0</td>
      <td>0.608289</td>
      <td>0.002043</td>
      <td>0.964035</td>
    </tr>
  </tbody>
</table>
</div>



La ligne du résumé de `reg` correspondant à `Nord` donne le test de 
nullité du coefficient correspondant, ce qui correspond ici à la nullité
de l'écart de cette modalité `Nord` par rapport à celle de référence `Est` et
donc à avoir les deux modalités fusionnées.

# Teneur en folate dans les globules rouges
Nous disposons de la mesure de concentration (en $\mu\mathrm{g/l}$) de folate
(nom de variable `folate`) dans les globules rouge durant une
anesthésie chez $n=22$ patients. L'anesthésie est utilise une
ventilation choisie parmi trois méthodes:
- le gaz utilisé est un mélange 50-50 de $\mathrm{N}_2$O 
  (oxyde nitreux ou gaz hilarant) et d'$\mathrm{O}_2$ 
  pendant une durée 24h (codé `N2O+O2,24h`);
- le gaz utilisé est un mélange 50-50 de $\mathrm{N}_2$O 
  (oxyde nitreux ou gaz hilarant) et d'$\mathrm{O}_2$ 
  uniquement pendant la durée de l'opération (codé `N2O+O2,op`);
- pas d'oxyde nitreux, uniquement de l'oxygène pendant 24h (codé `O2,24h`).
Nous cherchons à savoir si ces trois méthodes de ventilations sont
équivalentes.

### Importation
Importer les données qui sont dans le fichier `gr.csv` et
résumer les de façon numérique.
\[méthode `astype` sur la colonne du DataFrame et
méthode `describe` sur l'instance DataFrame\]


```{code-cell} python
gr = pd.read_csv("data/gr.csv", header=0, sep=";")
gr["ventilation"]=gr["ventilation"].astype("category")
gr.describe(include="all")
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
      <th>folate</th>
      <th>ventilation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>22.000000</td>
      <td>22</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>3</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>N2O+O2,op</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>9</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>283.227273</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>51.284391</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>206.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>249.500000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>274.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>305.500000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>392.000000</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### Représentation graphique
Représenter graphiquement les données.
\[`plt.plot` ou méthode `groupby` sur l'instance de DataFrame et méthode
`boxplot` sur l'instance DataFrame groupé\]

Le plus simple est de faire soit des points par ventilation


```{code-cell} python
plt.plot(gr.ventilation, gr.folate, "o")
```




    [<matplotlib.lines.Line2D at 0x7ff3e4314730>]




    
![png](p1_6_2_lab_correction_qualitatives_fr_files/p1_6_2_lab_correction_qualitatives_fr_21_1.png)
    


Nous constatons que les effectifs dans chaque groupe sont faibles, 
que les moyennes par groupe semblent différentes et que les variabilités
semblent comparables.

Mais un boxplot est aussi intéressant 
quoique moins adapté pour ces faibles effectifs par groupe.


```{code-cell} python
gr.groupby(by='ventilation').boxplot(False)
plt.show()
```


    
![png](p1_6_2_lab_correction_qualitatives_fr_files/p1_6_2_lab_correction_qualitatives_fr_24_0.png)
    


### Méthode de ventilation
Répondre à la question suivante: les trois méthodes de
  ventilation sont elles équivalentes ?

Faisons un test $F$ entre les deux modèles emboités $\mathrm{H}_0: \ y_{ij}=\mu + \varepsilon_{ij}$ et
$\mathrm{H}_1: \ y_{ij}=\mu + \alpha_i + \varepsilon_{ij}$ avec l'erreur de première espèce de $\alpha=5\%$.


```{code-cell} python
modele1 = smf.ols("folate ~ 1 + ventilation", data=gr).fit()
modele0 = smf.ols("folate ~ 1", data=gr).fit()
sm.stats.anova_lm(modele0, modele1)
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
      <td>21.0</td>
      <td>55231.863636</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>19.0</td>
      <td>39716.097222</td>
      <td>2.0</td>
      <td>15515.766414</td>
      <td>3.711336</td>
      <td>0.043589</td>
    </tr>
  </tbody>
</table>
</div>



La valeur de la statistique de test vaut $3.71$ et sa probabilité critique 
vaut 0.04, plus petite que $\alpha$ nous repoussons donc $\mathrm{H}_0$. 
Le type de ventilation a un effet.

### Analyse du modèle
Analyser les résidus du modèle retenu et interpréter les
  coefficients
  \[`plt.plot`, `get_influence`, `resid_studentized_external`,
  `sm.qqplot`\]

Les erreurs du modèle sont sensées être iid 
de loi normale de moyenne 0 et de variance $\sigma^2. 
Les résidus studentisés (par VC) peuvent être tracés
en fonction de $\hat Y$ (qui est la moyenne du groupe
de ventilation.)


```{code-cell} python
infl = modele1.get_influence()
plt.plot(modele1.predict(), infl.resid_studentized_external, "o")
```




    [<matplotlib.lines.Line2D at 0x7ff3e4185b50>]




    
![png](p1_6_2_lab_correction_qualitatives_fr_files/p1_6_2_lab_correction_qualitatives_fr_31_1.png)
    


Nous retrouvons que la variabilité semble être plus élevé dans un groupe
mais compte tenu du faible nombre d'observations on ne peut franchement
parler de problème rédibitoire.

Pour envisager la normalité il est classique de regarder un QQ-plot
- Les résidus studentisés sont ordonnés (ordre croissant):
  $t^*_{(1)},\dotsc t^*_{(n)}$
- Soit $Z_{(1)},\dotsc,Z_{(n)}$ un $n$-échantillon tiré selon
  une loi $\mathcal{N}(0,1)$ puis ordonné dans l'ordre croissant. On estime
  alors la valeur moyenne des $Z_{(i)}$ (estimation notée $\bar Z_{(i)}$)
- On trace alors les $n$ couples $\bar Z_{(i)},t^*_{(i)}$



```{code-cell} python
sm.qqplot(infl.resid_studentized_external, line ='s')
```




    
![png](p1_6_2_lab_correction_qualitatives_fr_files/p1_6_2_lab_correction_qualitatives_fr_34_0.png)
    




    
![png](p1_6_2_lab_correction_qualitatives_fr_files/p1_6_2_lab_correction_qualitatives_fr_34_1.png)
    


La normalité des résidus semble correcte. Notre conclusion sur un effet du 
type de ventilation n'est pas amoindrie car le modèle de régression (ici 
nommé ANOVA à un facteur) semble correct.

# ANOVA à deux facteurs
Nous disposons de la hauteur moyenne de 8 provenances d'eucalyptus
camaldulensis: les graines de ces eucalyptus ont été récoltées dans
huit endroits du monde (ie les 8 provenances) et plantées aux environs
de Pointe-Noire (Congo). Au même âge sont mesurées les hauteurs
moyennes pour ces 8 provenances. Ces provenances sont plantées dans
une très grande parcelle que l'on soupçonne de ne pas être homogène
du simple fait de sa taille. Cette parcelle est donc divisée en sous 
parcelles appelées bloc que l'on espère être homogène. Les données
propose les hauteurs moyennes des arbres par bloc-provenance.

Nous souhaitons savoir si ces huit provenances sont identiques.

1. Importer les données qui sont dans le fichier `eucalyptus_camaldulensis.txt` et
   résumer les de façon numérique.


```{code-cell} python
camal = pd.read_csv("data/eucalyptus_camaldulensis.txt", header=0, sep=" ", decimal=",")
camal.bloc = camal.bloc.astype("category")
camal.provenance = camal.provenance.astype("category")
camal.describe(include="all")
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
      <th>hauteur</th>
      <th>bloc</th>
      <th>provenance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>56.000000</td>
      <td>56</td>
      <td>56</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>14</td>
      <td>8</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>b1</td>
      <td>pr1</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>149.660714</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>11.741576</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>128.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>141.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>147.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>159.250000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>174.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



2. Représenter graphiquement les données utilisées pour la réponse à la question.


```{code-cell} python
camal.groupby(by="provenance").boxplot(False)
plt.show()
```


    
![png](p1_6_2_lab_correction_qualitatives_fr_files/p1_6_2_lab_correction_qualitatives_fr_40_0.png)
    


Les provenances 2 et 4 semblent nettement supérieures.

3. Répondre à la question posée (les huit provenances sont elles identiques ?).
   Où intervient (indirectement) la variable `bloc` dans la statistique de test utilisée ?


```{code-cell} python
modele0 = smf.ols("hauteur~bloc", data=camal).fit()
modele1 = smf.ols("hauteur~bloc+provenance", data=camal).fit()
sm.stats.anova_lm(modele0, modele1)
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
      <td>42.0</td>
      <td>6254.250000</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>35.0</td>
      <td>988.166667</td>
      <td>7.0</td>
      <td>5266.083333</td>
      <td>26.645724</td>
      <td>3.087688e-12</td>
    </tr>
  </tbody>
</table>
</div>



La valeur de la statistique de test vaut $26.65$ et sa probabilité critique 
est quasi nulles, plus petite que $\alpha=1\%$ 
nous repoussons donc $\mathrm{H}_0$. 
La provenance a un effet (confirmant le graphique de la question précédente)

La statistique $F$ compare la variabilité entre provenance (numérateur) et
la variabilité résiduelle ($\hat\sigma^2$ au dénominateur). Pour améliorer
la sensibilité du test il est important d'avoir une petite variabilité résiduelle
et donc d'inclure les variables explicatives même si elles ne sont pas 
dans le questionnement initial, ici la variable `bloc`.

4. Analyser les résidus du modèle retenu. 
   Tracer les résidus en fonction de la variable `bloc`. 


```{code-cell} python
camal["rstudent"] = modele1.get_influence().resid_studentized_external
plt.plot(modele1.predict(), camal.rstudent, "*")
#.boxplot()
```




    [<matplotlib.lines.Line2D at 0x7fa2ad7905b0>]




    
![png](p1_6_2_lab_correction_qualitatives_fr_files/p1_6_2_lab_correction_qualitatives_fr_46_1.png)
    



```{code-cell} python
camal.loc[:,["rstudent", "bloc"]].groupby(by="bloc").boxplot(False)
```




    <Axes: >




    
![png](p1_6_2_lab_correction_qualitatives_fr_files/p1_6_2_lab_correction_qualitatives_fr_47_1.png)
    


Les résidus semblent corrects, le test est très significatif nous sommes
donc assez certains de notre conclusion: la provenance a bien un effet 
sur la hauteur.
