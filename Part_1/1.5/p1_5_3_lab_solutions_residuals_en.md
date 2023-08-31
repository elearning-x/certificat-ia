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
  title: 'Solutions to Lab Session on Residuals'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa Bedin<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

+++

# Python modules
Import modules pandas (as `pd`) numpy (as `np`)
matplotlib.pyplot (as `plt`), statsmodels.formula.api (as `smf`)
and statsmodels.api (as `sm`)


```{code-cell} python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
```

# Multiple Regression (course model)

### Data import
Import ozone data into pandas `ozone` DataFrame
[`read_csv` from `pandas`]


```{code-cell} python
ozone = pd.read_csv('data/ozone.txt', sep=';')
```

### Course model estimation
We are interested in building an ozone forecasting model using 
multiple regression. This regression will explain
the maximum ozone concentration of the day (variable `O3`) by 
- temperature at noon, noted `T12`
- cloud cover at midday, noted as `Ne12`
- the wind speed on the East-West axis, noted `Vx`.
Traditionally, we always introduce the constant (and do so here too).
Estimate the OLS model and summarize.

\[Use the `ols` method from `smf`, the `fit` method from the `OLS` class and the 
method `summary` for the adjusted instance/model\]


```{code-cell} python
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
  <th>Date:</th>             <td>Thu, 15 Jun 2023</td> <th>  Prob (F-statistic):</th> <td>1.66e-11</td>
</tr>
<tr>
  <th>Time:</th>                 <td>12:06:59</td>     <th>  Log-Likelihood:    </th> <td> -200.50</td>
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



### Residuals \$\varepsilon\$
Display residuals graph (`resid` attribute of estimated model)
(with \$\hat y\$  (`predict` method of estimated model) on the x-axis and \$\varepsilon\$ on the y-axis).

\[plt `plot`\]


```{code-cell} python
plt.plot(reg.predict(), reg.resid, 'o')
```




    [<matplotlib.lines.Line2D at 0x129368bb0>]




    
![png](media/Part_1/1.5/p1_5_3_lab_solutions_residuals_en_files/p1_5_3_lab_solutions_residuals_en_8_1.png)
    


No visible residue structuring. The thickness (standard deviation) of the points seems to be 
the same, but these residuals by construction do not have the same variance, 
so it's tricky to conclude on the \$mathrm{V}(\varepsilon_i)=\sigma^2\$ hypothesis.
What's more, the scale of the ordinates depends on the problem, so these residuals are not very practical.

### Studentized Residuals
Display the graph of residuals studentized by cross-validation (with \$\hat y\$ on the x-axis and 
\$\varepsilon\$ on the ordinate). To do this, use the `get_influence` function/method 
which will return an object (let's call it `infl`) with a `resid_studentized_external` attribute containing the desired residues.


```{code-cell} python
infl = reg.get_influence()
plt.plot(reg.predict(), infl.resid_studentized_external, 'o')
```




    [<matplotlib.lines.Line2D at 0x12b14e2b0>]




    
![png](media/Part_1/1.5/p1_5_3_lab_solutions_residuals_en_files/p1_5_3_lab_solutions_residuals_en_11_1.png)
    


No visible residue structuring. The thickness (standard deviation) of the points seems to be 
the same, so the hypothesis $\mathrm{V}(\varepsilon_i)=\sigma^2\$ appears to be correct. No points outside [-2,2], so no outliers.

### Leverage points
Represent \$h_{ii}\$ with `plt.stem` according to line number
[`np.arange`, DataFrame `shape` attribute, instance attribute 
`hat_matrix_diag` for `infl`]


```{code-cell} python
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




```{code-cell} python
n_data = ozone.shape[0]
plt.stem(np.arange(n_data), infl.hat_matrix_diag)
```




    <StemContainer object of 3 artists>




    
![png](media/Part_1/1.5/p1_5_3_lab_solutions_residuals_en_files/p1_5_3_lab_solutions_residuals_en_15_1.png)
    


No \$h_{ii}\$ significantly larger 
than the others, so the experimental design is correct.

# R²
We are interested in building an ozone forecasting model using 
multiple regression. However, we're not sure a priori
which variables are useful. Let's build several models.

### Estimation of the course model
This regression will explain
the maximum ozone concentration of the day (variable `O3`) by 
- temperature at noon, noted `T12`
- cloud cover at midday, noted as `Ne12`
- the wind speed on the East-West axis, noted `Vx`.
Traditionally, we always introduce the constant (and do so here too).
Estimate the OLS model and summarize.

[Use the `ols` method from `smf`, the `fit` method from the `OLS` class and the 
method `summary` for the adjusted instance/model]


```{code-cell} python
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
  <th>Date:</th>             <td>Thu, 15 Jun 2023</td> <th>  Prob (F-statistic):</th> <td>1.66e-11</td>
</tr>
<tr>
  <th>Time:</th>                 <td>12:58:57</td>     <th>  Log-Likelihood:    </th> <td> -200.50</td>
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



### Estimation of another model
This regression will explain
the maximum ozone concentration of the day (variable `O3`) by 
- temperature at noon, noted `T12`
- the temperature at 3pm, `T15`
- cloudiness at noon, noted as `Ne12`
- wind speed on east-west axis noted `Vx`
- the maximum of the previous day `O3v`.
Traditionally, we always introduce the constant (do so here too).
Estimate the OLS model and summarize.


```{code-cell} python
reg5 = smf.ols('O3~T12+T15+Ne12+Vx+O3v', data=ozone).fit()
reg5.summary()
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
  <th>Date:</th>             <td>Thu, 15 Jun 2023</td> <th>  Prob (F-statistic):</th> <td>1.34e-11</td>
</tr>
<tr>
  <th>Time:</th>                 <td>13:03:58</td>     <th>  Log-Likelihood:    </th> <td> -196.15</td>
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
  <th>T15</th>       <td>    0.6306</td> <td>    1.409</td> <td>    0.447</td> <td> 0.657</td> <td>   -2.210</td> <td>    3.471</td>
</tr>
<tr>
  <th>T12</th>       <td>    0.4675</td> <td>    1.459</td> <td>    0.320</td> <td> 0.750</td> <td>   -2.474</td> <td>    3.409</td>
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



### Compare R2
Compare the R2 of the 3- and 5-variable models 
and explain why this was expected.


```{code-cell} python
reg.rsquared, reg5.rsquared
```




    (0.6818780319375457, 0.7327420826035473)



The R2 increases with the number of variables added. Thus, R2 score cannot be used to compare fits for models with different numbers of variables.

# Partial residuals (to go further)
This exercise demonstrates the practical usefulness of the partial residuals discussed in TD.
The data are in the files `tprespartial.dta` and
`tpbisrespartiel.dta`, the aim of this exercise is to show that the analysis of partial
of partial residuals can improve modeling.

### Import data
You have one variable to explain \$Y\$
and four explanatory variables in the file `tprespartiel.dta`.


```{code-cell} python
tpres = pd.read_csv('data/tprespartiel.dta', sep=';')
tpres.head()
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
OLS estimation of model parameters \$Y_i= \beta_0 + \beta_1 X_{i,1}+ \cdots+
\beta_4 X_{i,4} + \varepsilon_i.\$
[`ols` from `smf`, method `fit` from class `OLS` and 
method `summary` for the adjusted instance/model]


```{code-cell} python
reg = smf.ols('Y~X1+X2+X3+X4', data=tpres).fit()
reg.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>Y</td>        <th>  R-squared:         </th> <td>   0.986</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.985</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   1678.</td>
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 15 Jun 2023</td> <th>  Prob (F-statistic):</th> <td>3.62e-87</td>
</tr>
<tr>
  <th>Time:</th>                 <td>13:15:40</td>     <th>  Log-Likelihood:    </th> <td> -122.64</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   100</td>      <th>  AIC:               </th> <td>   255.3</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    95</td>      <th>  BIC:               </th> <td>   268.3</td>
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
  <th>Intercept</th> <td>   -6.0869</td> <td>    0.250</td> <td>  -24.354</td> <td> 0.000</td> <td>   -6.583</td> <td>   -5.591</td>
</tr>
<tr>
  <th>X1</th>        <td>   20.3451</td> <td>    0.282</td> <td>   72.183</td> <td> 0.000</td> <td>   19.786</td> <td>   20.905</td>
</tr>
<tr>
  <th>X2</th>        <td>  -11.9437</td> <td>    0.287</td> <td>  -41.672</td> <td> 0.000</td> <td>  -12.513</td> <td>  -11.375</td>
</tr>
<tr>
  <th>X3</th>        <td>    0.7709</td> <td>    0.325</td> <td>    2.368</td> <td> 0.020</td> <td>    0.125</td> <td>    1.417</td>
</tr>
<tr>
  <th>X4</th>        <td>   -0.7813</td> <td>    0.269</td> <td>   -2.901</td> <td> 0.005</td> <td>   -1.316</td> <td>   -0.247</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 0.344</td> <th>  Durbin-Watson:     </th> <td>   2.188</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.842</td> <th>  Jarque-Bera (JB):  </th> <td>   0.512</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.035</td> <th>  Prob(JB):          </th> <td>   0.774</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.656</td> <th>  Cond. No.          </th> <td>    5.94</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



### Analyze partial residuals
What do you think of the results?
[In the `plot_ccpr_grid` sub-module of `sm.graphics`, partial residuals are called
called "Component-Component plus Residual"
(CCPR) in the statsmodels module...


```{code-cell} python
sm.graphics.plot_ccpr_grid(reg)
plt.show()
```


    
![png](media/Part_1/1.5/p1_5_3_lab_solutions_residuals_en_files/p1_5_3_lab_solutions_residuals_en_31_0.png)
    


Clearly, the graph for the variable `X4` does not show
points arranged along a straight line or a structureless cloud. 
It shows a \$x\mapsto x^2\$ type of structuring.

### Model improvement 
Replace $X_4$ by $X_5=X_4^2$ in the previous model. What do you think of
  the new model? You can compare this model with the one
  previous question.
[the `ols` method of the `smf` class, the `fit` method of the `OLS` class and the 
 instance attribute `rsquared`]
You can use the
operations and functions in formulas
(see https://www.statsmodels.org/stable/example_formulas.html)


```{code-cell} python
reg2 = smf.ols('Y~X1+X2+X3+np.square(X4)', data=tpres).fit()
reg2.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>Y</td>        <th>  R-squared:         </th> <td>   0.997</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.996</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   6984.</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 15 Jun 2023</td> <th>  Prob (F-statistic):</th> <td>2.31e-116</td>
</tr>
<tr>
  <th>Time:</th>                 <td>13:48:42</td>     <th>  Log-Likelihood:    </th> <td> -51.869</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   100</td>      <th>  AIC:               </th> <td>   113.7</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    95</td>      <th>  BIC:               </th> <td>   126.8</td> 
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
        <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>     <td>   -5.0387</td> <td>    0.136</td> <td>  -37.033</td> <td> 0.000</td> <td>   -5.309</td> <td>   -4.769</td>
</tr>
<tr>
  <th>X1</th>            <td>   19.9930</td> <td>    0.140</td> <td>  142.431</td> <td> 0.000</td> <td>   19.714</td> <td>   20.272</td>
</tr>
<tr>
  <th>X2</th>            <td>  -11.8953</td> <td>    0.140</td> <td>  -85.081</td> <td> 0.000</td> <td>  -12.173</td> <td>  -11.618</td>
</tr>
<tr>
  <th>X3</th>            <td>    0.9682</td> <td>    0.160</td> <td>    6.033</td> <td> 0.000</td> <td>    0.650</td> <td>    1.287</td>
</tr>
<tr>
  <th>np.square(X4)</th> <td>   -9.9740</td> <td>    0.548</td> <td>  -18.191</td> <td> 0.000</td> <td>  -11.063</td> <td>   -8.886</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 0.350</td> <th>  Durbin-Watson:     </th> <td>   1.685</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.840</td> <th>  Jarque-Bera (JB):  </th> <td>   0.070</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.012</td> <th>  Prob(JB):          </th> <td>   0.965</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.128</td> <th>  Cond. No.          </th> <td>    17.8</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



### Analyze partial residuals
Analyze the partial residuals of the new model and note that
that they appear to be correct.
[In the `plot_ccpr_grid` sub-module of `sm.graphics`, partial residuals are called
called "Component-Component plus Residual"
(CCPR) in the statsmodels module...


```{code-cell} python
sm.graphics.plot_ccpr_grid(reg2)
plt.show()
```


    
![png](media/Part_1/1.5/p1_5_3_lab_solutions_residuals_en_files/p1_5_3_lab_solutions_residuals_en_36_0.png)
    


The graphs show points with no obvious structure
or arranged along straight lines. The model would appear to be correct. We can compare 
compare them (same number of variables) by R2


```{code-cell} python
reg.rsquared, reg2.rsquared
```




    (0.9860422309885765, 0.9966109930897685)



Do the same for `tp2bisrespartiel`.


```{code-cell} python
tpbis = pd.read_csv('data/tpbisrespartiel.dta', sep=';')
tpbis.head()
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
      <td>4.421534</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.427154</td>
      <td>0.662147</td>
      <td>0.394141</td>
      <td>0.438178</td>
      <td>-7.886440</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.406420</td>
      <td>0.809686</td>
      <td>0.639263</td>
      <td>-0.087607</td>
      <td>-1.146733</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.061812</td>
      <td>0.420397</td>
      <td>0.437492</td>
      <td>0.468991</td>
      <td>-10.456241</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.965881</td>
      <td>0.006918</td>
      <td>0.141291</td>
      <td>0.302681</td>
      <td>4.468512</td>
    </tr>
  </tbody>
</table>
</div>




```{code-cell} python
reg = smf.ols('Y~X1+X2+X3+X4', data=tpbis).fit()
reg.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>Y</td>        <th>  R-squared:         </th> <td>   0.811</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.803</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   101.7</td>
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 15 Jun 2023</td> <th>  Prob (F-statistic):</th> <td>1.86e-33</td>
</tr>
<tr>
  <th>Time:</th>                 <td>13:54:11</td>     <th>  Log-Likelihood:    </th> <td> -289.47</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   100</td>      <th>  AIC:               </th> <td>   588.9</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    95</td>      <th>  BIC:               </th> <td>   602.0</td>
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
  <th>Intercept</th> <td>   -2.4646</td> <td>    1.325</td> <td>   -1.860</td> <td> 0.066</td> <td>   -5.096</td> <td>    0.167</td>
</tr>
<tr>
  <th>X1</th>        <td>   19.1063</td> <td>    1.495</td> <td>   12.783</td> <td> 0.000</td> <td>   16.139</td> <td>   22.074</td>
</tr>
<tr>
  <th>X2</th>        <td>  -12.3416</td> <td>    1.520</td> <td>   -8.120</td> <td> 0.000</td> <td>  -15.359</td> <td>   -9.324</td>
</tr>
<tr>
  <th>X3</th>        <td>    0.0469</td> <td>    1.726</td> <td>    0.027</td> <td> 0.978</td> <td>   -3.380</td> <td>    3.473</td>
</tr>
<tr>
  <th>X4</th>        <td>  -17.1295</td> <td>    1.428</td> <td>  -11.992</td> <td> 0.000</td> <td>  -19.965</td> <td>  -14.294</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>16.579</td> <th>  Durbin-Watson:     </th> <td>   1.951</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>   8.670</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.539</td> <th>  Prob(JB):          </th> <td>  0.0131</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.042</td> <th>  Cond. No.          </th> <td>    5.94</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```{code-cell} python
sm.graphics.plot_ccpr_grid(reg)
plt.show()
```


    
![png](media/Part_1/1.5/p1_5_3_lab_solutions_residuals_en_files/p1_5_3_lab_solutions_residuals_en_42_0.png)
    



```{code-cell} python
reg2 = smf.ols('Y~X1+X2+X3+np.sin(2*np.pi*X4)', data=tpbis).fit()
reg2.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>Y</td>        <th>  R-squared:         </th> <td>   0.998</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.998</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>1.546e+04</td>
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 15 Jun 2023</td> <th>  Prob (F-statistic):</th> <td>1.01e-132</td>
</tr>
<tr>
  <th>Time:</th>                 <td>13:56:37</td>     <th>  Log-Likelihood:    </th> <td> -48.664</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   100</td>      <th>  AIC:               </th> <td>   107.3</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    95</td>      <th>  BIC:               </th> <td>   120.4</td> 
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
             <td></td>               <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>              <td>   -5.0628</td> <td>    0.120</td> <td>  -42.134</td> <td> 0.000</td> <td>   -5.301</td> <td>   -4.824</td>
</tr>
<tr>
  <th>X1</th>                     <td>   20.0240</td> <td>    0.134</td> <td>  148.964</td> <td> 0.000</td> <td>   19.757</td> <td>   20.291</td>
</tr>
<tr>
  <th>X2</th>                     <td>  -12.0037</td> <td>    0.136</td> <td>  -88.033</td> <td> 0.000</td> <td>  -12.274</td> <td>  -11.733</td>
</tr>
<tr>
  <th>X3</th>                     <td>    1.1853</td> <td>    0.155</td> <td>    7.637</td> <td> 0.000</td> <td>    0.877</td> <td>    1.493</td>
</tr>
<tr>
  <th>np.sin(2 * np.pi * X4)</th> <td>  -10.0305</td> <td>    0.059</td> <td> -171.452</td> <td> 0.000</td> <td>  -10.147</td> <td>   -9.914</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 2.376</td> <th>  Durbin-Watson:     </th> <td>   2.059</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.305</td> <th>  Jarque-Bera (JB):  </th> <td>   2.394</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.343</td> <th>  Prob(JB):          </th> <td>   0.302</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.677</td> <th>  Cond. No.          </th> <td>    5.95</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```{code-cell} python
sm.graphics.plot_ccpr_grid(reg2)
plt.show()
```


    
![png](media/Part_1/1.5/p1_5_3_lab_solutions_residuals_en_files/p1_5_3_lab_solutions_residuals_en_44_0.png)
    

