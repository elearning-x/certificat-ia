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
  title: 'Correction du TP choix de variables'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa Bedin<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

## Modules



Importer les modules pandas (comme `pd`) numpy (commme `np`)
matplotlib.pyplot (comme  `plt`) et statsmodels.formula.api (comme `smf`).


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
```

## Régression ridge sur les données d&rsquo;ozone



#### Importation des données



Importer les données d&rsquo;ozone `ozonecomplet.csv` et éliminer les deux dernières
variables (qualitatives) et faites un résumé numérique par variable [méthode
`astype` sur la colonne du DataFrame et méthode `describe` sur l&rsquo;instance
DataFrame]




```python
ozone = pd.read_csv("data/ozonecomplet.csv", header=0, sep=";")
ozone = ozone.drop(['nomligne', 'Ne', 'Dv'], axis=1)
ozone.describe()
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
    </tr>
  </tbody>
</table>
</div>



#### Sélection descendante/backward



Proposer une fonction qui permet la sélection descendante/backward. Elle utilisera
les formules de `statsmodels` et incluera toujours la constante. En entrée serviront
trois arguments: le DataFrame des données, la formule de départ et le critère (AIC ou BIC).
La fonction retournera le modèle estimé via `smf.ols`




```python
def olsbackward(data, start, crit="aic", verbose=False):
    """Backward selection for linear model with smf (with formula).

    Parameters:
    -----------
    data (pandas DataFrame): DataFrame with all possible predictors
            and response
    start (string): a string giving the starting model
            (ie the starting point)
    crit (string): "aic"/"AIC" or "bic"/"BIC"
    verbose (boolean): if True verbose print

    Returns:
    --------
    model: an "optimal" linear model fitted with statsmodels
           with an intercept and
           selected by forward/backward or both algorithm with crit criterion
    """
    # criterion
    if not (crit == "aic" or crit == "AIC" or crit == "bic" or crit == "BIC"):
        raise ValueError("criterion error (should be AIC/aic or BIC/bic)")
    # starting point
    formula_start = start.split("~")
    response = formula_start[0].strip()
    # explanatory variables for the 3 models
    start_explanatory = set([item.strip() for item in
                             formula_start[1].split("+")]) - set(['1'])
    # setting up the set "remove" which contains the possible
    # variable to remove
    lower_explanatory = set([])
    remove = start_explanatory - lower_explanatory
    # current point
    selected = start_explanatory
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(list(selected)))
    if crit == "aic" or crit == "AIC":
        current_score = smf.ols(formula, data).fit().aic
    elif crit == "bic" or crit == "BIC":
        current_score = smf.ols(formula, data).fit().bic
    if verbose:
        print("----------------------------------------------")
        print((current_score, "Starting", selected))
    # main loop
    while True:
        scores_with_candidates = []
        for candidate in remove:
            tobetested = selected - set([candidate])
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(list(tobetested)))
            if crit == "aic" or crit == "AIC":
                score = smf.ols(formula, data).fit().aic
            elif crit == "bic" or crit == "BIC":
                score = smf.ols(formula, data).fit().bic
            if verbose:
                print((score, "-", candidate))
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop(0)
        if current_score > best_new_score:
            remove = remove - set([best_candidate])
            selected = selected - set([best_candidate])
            current_score = best_new_score
            if verbose:
                print("----------------------------------------------")
                print((current_score, "New Current", selected))
        else:
            break
    if verbose:
        print("----------------------------------------------")
        print((current_score, "Final", selected))
    formula = "{} ~ {} + 1".format(response, ' + '.join(list(selected)))
    model = smf.ols(formula, data).fit()
    return model
```


```python
modelefinal = olsbackward(ozone,"O3~T9+T12+T15+Ne9+Ne12+Ne15+Vx9+Vx12+Vx15+O3v", verbose=True)
```

    ----------------------------------------------
    (925.1002020679311, 'Starting', {'Vx15', 'O3v', 'T15', 'Vx12', 'T12', 'Ne15', 'Ne12', 'T9', 'Ne9', 'Vx9'})
    (923.3317000545144, '-', 'Vx15')
    (953.3560377543913, '-', 'O3v')
    (923.363923324606, '-', 'T15')
    (923.1011713799451, '-', 'Vx12')
    (925.7333732341065, '-', 'T12')
    (923.1374214147936, '-', 'Ne15')
    (923.2052359512816, '-', 'Ne12')
    (923.1005187562648, '-', 'T9')
    (928.9798518527307, '-', 'Ne9')
    (924.2910760882157, '-', 'Vx9')
    ----------------------------------------------
    (923.1005187562648, 'New Current', {'Vx15', 'O3v', 'T15', 'Vx12', 'T12', 'Ne15', 'Ne12', 'Ne9', 'Vx9'})
    (921.3343370195821, '-', 'Vx15')
    (954.0035689902516, '-', 'O3v')
    (921.3657520095587, '-', 'T15')
    (921.1012966914407, '-', 'Vx12')
    (924.4976674984907, '-', 'T12')
    (921.1374374227578, '-', 'Ne15')
    (921.2193530155344, '-', 'Ne12')
    (927.2908775332714, '-', 'Ne9')
    (922.5113099067607, '-', 'Vx9')
    ----------------------------------------------
    (921.1012966914407, 'New Current', {'Vx15', 'O3v', 'T15', 'T12', 'Ne15', 'Ne12', 'Ne9', 'Vx9'})
    (919.5284389936069, '-', 'Vx15')
    (952.0058285995476, '-', 'O3v')
    (919.3697661183905, '-', 'T15')
    (922.529192489462, '-', 'T12')
    (919.1382882547716, '-', 'Ne15')
    (919.2255235409766, '-', 'Ne12')
    (925.2995877582118, '-', 'Ne9')
    (920.8093672514524, '-', 'Vx9')
    ----------------------------------------------
    (919.1382882547716, 'New Current', {'Vx15', 'O3v', 'T15', 'T12', 'Ne12', 'Ne9', 'Vx9'})
    (917.5415291187271, '-', 'Vx15')
    (950.0081756555403, '-', 'O3v')
    (917.3810597522813, '-', 'T15')
    (922.2442506031016, '-', 'T12')
    (917.2255825403324, '-', 'Ne12')
    (923.4905604684955, '-', 'Ne9')
    (918.8728607212622, '-', 'Vx9')
    ----------------------------------------------
    (917.2255825403324, 'New Current', {'Vx15', 'O3v', 'T15', 'T12', 'Ne9', 'Vx9'})
    (915.6416421263025, '-', 'Vx15')
    (948.1690191516109, '-', 'O3v')
    (915.4700257339703, '-', 'T15')
    (920.6465496282251, '-', 'T12')
    (926.2439264465465, '-', 'Ne9')
    (917.0243089102432, '-', 'Vx9')
    ----------------------------------------------
    (915.4700257339703, 'New Current', {'Vx15', 'O3v', 'T12', 'Ne9', 'Vx9'})
    (913.8666234696386, '-', 'Vx15')
    (947.3509351899581, '-', 'O3v')
    (944.669544463007, '-', 'T12')
    (924.8804583198719, '-', 'Ne9')
    (915.4175462175649, '-', 'Vx9')
    ----------------------------------------------
    (913.8666234696386, 'New Current', {'T12', 'O3v', 'Vx9', 'Ne9'})
    (942.7179924848651, '-', 'T12')
    (945.5729308215273, '-', 'O3v')
    (916.5904254533016, '-', 'Vx9')
    (925.5014911907435, '-', 'Ne9')
    ----------------------------------------------
    (913.8666234696386, 'Final', {'T12', 'O3v', 'Vx9', 'Ne9'})



```python
modelefinal.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>O3</td>        <th>  R-squared:         </th> <td>   0.762</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.753</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   85.75</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 12 Jul 2023</td> <th>  Prob (F-statistic):</th> <td>1.76e-32</td>
</tr>
<tr>
  <th>Time:</th>                 <td>16:34:59</td>     <th>  Log-Likelihood:    </th> <td> -451.93</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   112</td>      <th>  AIC:               </th> <td>   913.9</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   107</td>      <th>  BIC:               </th> <td>   927.5</td>
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
  <th>Intercept</th> <td>   12.6313</td> <td>   11.001</td> <td>    1.148</td> <td> 0.253</td> <td>   -9.177</td> <td>   34.439</td>
</tr>
<tr>
  <th>T12</th>       <td>    2.7641</td> <td>    0.475</td> <td>    5.825</td> <td> 0.000</td> <td>    1.823</td> <td>    3.705</td>
</tr>
<tr>
  <th>O3v</th>       <td>    0.3548</td> <td>    0.058</td> <td>    6.130</td> <td> 0.000</td> <td>    0.240</td> <td>    0.470</td>
</tr>
<tr>
  <th>Vx9</th>       <td>    1.2929</td> <td>    0.602</td> <td>    2.147</td> <td> 0.034</td> <td>    0.099</td> <td>    2.487</td>
</tr>
<tr>
  <th>Ne9</th>       <td>   -2.5154</td> <td>    0.676</td> <td>   -3.722</td> <td> 0.000</td> <td>   -3.855</td> <td>   -1.176</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 8.790</td> <th>  Durbin-Watson:     </th> <td>   1.944</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.012</td> <th>  Jarque-Bera (JB):  </th> <td>  17.762</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.156</td> <th>  Prob(JB):          </th> <td>0.000139</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.926</td> <th>  Cond. No.          </th> <td>    810.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python

```
