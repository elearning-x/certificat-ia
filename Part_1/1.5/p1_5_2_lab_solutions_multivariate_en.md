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
  title: 'Solutions to Lab Session on Multivariate Regression'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa Bedin<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

# Python Modules
Import modules pandas (as `pd`), numpy (as `np`), matplotlib.pyplot (as `plt`) and statsmodels.formula.api (as `smf`). 
Also import `Axes3D` from `mpl_toolkits.mplot3d`.


```{code-cell} python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.formula.api as smf
```

# Multiple Regression on Ozone data (2 variables)

### Data import
Import ozone data into pandas `ozone` DataFrame.


```{code-cell} python
ozone = pd.read_csv("data/ozone.txt", header=0, sep=";")
```

### 3D representation
We're interested in building an ozone forecasting model using 
multiple regression. This regression will explain
the maximum ozone concentration of the day (variable `O3`) by 
- the temperature at midday (`T12`)
- wind speed on the East-West axis, noted `Vx`.
Let's graph the data with `O3` on the z axis, 
`T12` on the x-axis and `Vx` on the y-axis.


```{code-cell} python
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax = Axes3D(fig)
ax.scatter(ozone["T12"], ozone["Vx"],ozone["O3"])
ax.set_xlabel('T12')
ax.set_ylabel('Vx')
ax.set_zlabel('O3')
```




    Text(0.5, 0, 'O3')




    
![png](media/Part_1/1.5/p1_5_2_lab_solutions_multivariate_en_files/p1_5_2_lab_solutions_multivariate_en_6_1.png)
    


### Forecasting model
Write the above model.

\$y_i = \beta_1 + \beta_2 X_i + \beta_3 Z_i + \varepsilon_i\$
where 
- \$X_i\$ is the \$i^e\$ observation of the explanatory variable `T12` and
- \$Z_i\$ is the \$i^e\$ observation of the explanatory variable `Vx`.
- \$X_i\$ is the \$i^e\$ observation for the explanatory variable `O3`.
- \$varepsilon_i\$ is the \$i^e\$ coordinate of the error vector
  \$varepsilon\$
Traditionally, as is the case here, we always introduce the constant 
(variable associated with \$\beta_1\$).

### Model estimation
Use OLS to estimate the parameters of the model described above and summarize them.


```{code-cell} python
reg = smf.ols('O3~T12+Vx', data=ozone).fit()
reg.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>O3</td>        <th>  R-squared:         </th> <td>   0.525</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.505</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   25.96</td>
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 15 Jun 2023</td> <th>  Prob (F-statistic):</th> <td>2.54e-08</td>
</tr>
<tr>
  <th>Time:</th>                 <td>11:38:34</td>     <th>  Log-Likelihood:    </th> <td> -210.53</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    50</td>      <th>  AIC:               </th> <td>   427.1</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    47</td>      <th>  BIC:               </th> <td>   432.8</td>
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
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>   35.4530</td> <td>   10.745</td> <td>    3.300</td> <td> 0.002</td> <td>   13.838</td> <td>   57.068</td>
</tr>
<tr>
  <th>T12</th>       <td>    2.5380</td> <td>    0.515</td> <td>    4.927</td> <td> 0.000</td> <td>    1.502</td> <td>    3.574</td>
</tr>
<tr>
  <th>Vx</th>        <td>    0.8736</td> <td>    0.177</td> <td>    4.931</td> <td> 0.000</td> <td>    0.517</td> <td>    1.230</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 0.280</td> <th>  Durbin-Watson:     </th> <td>   1.678</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.869</td> <th>  Jarque-Bera (JB):  </th> <td>   0.331</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.165</td> <th>  Prob(JB):          </th> <td>   0.848</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.777</td> <th>  Cond. No.          </th> <td>    94.4</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



# Multiple Regression (course model) for Ozone Data

### Data import
Import ozone data into pandas `ozone` DataFrame


```{code-cell} python
ozone = pd.read_csv("data/ozone.txt", header=0, sep=";")
```

### Course model estimation
We are interested in building an ozone forecasting model using 
multiple regression. This regression will explain
the maximum ozone concentration of the day (variable `O3`) by 
- temperature at noon, noted `T12
- cloud cover at midday, noted as `Ne12
- the wind speed on the East-West axis, noted `Vx`.
Traditionally, we always introduce the constant (and do so here too).
Estimate the OLS model and summarize.


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
  <th>Time:</th>                 <td>11:40:13</td>     <th>  Log-Likelihood:    </th> <td> -200.50</td>
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



### Variability 
- Among the estimators of the coefficients of the effects of the variables
(excluding the constant) that is the most variable.
- Variability is indicated by
  - parameter variance
  - the standard deviation of the parameter
  - the estimated variance of the parameter
  - the parameter's estimated standard deviation
- Display estimate for \$\sigma^2\$

Reading the summary, the `std err` column gives the estimated
estimated standard deviations of the coordinates of \$\hat \beta\$ and the largest 
is that associated with the `Ne12` variable.


```{code-cell} python
reg.scale
```




    193.57873901889417



# Multiple regression on eucalyptus data

### Importing data
Import eucalytus data into pandas `eucalypt` DataFrame


```{code-cell} python
eucalypt = pd.read_csv("data/eucalyptus.txt", header=0, sep=";")
```

### data representation
Represent point cloud


```{code-cell} python
plt.plot(eucalypt["circ"],eucalypt["ht"],'o')
plt.xlabel("circ")
plt.ylabel("ht")
```




    Text(0, 0.5, 'ht')




    
![png](media/Part_1/1.5/p1_5_2_lab_solutions_multivariate_en_files/p1_5_2_lab_solutions_multivariate_en_23_1.png)
    


### Forecast model
Estimate (by OLS) the linear model explaining the height (`ht`) 
by the circumference variable (`circ`) and the square root of circumference.
circumference.  You can use
operations and functions in formulas
(see https://www.statsmodels.org/stable/example_formulas.html)


```{code-cell} python
regmult = smf.ols("ht ~ circ +  np.sqrt(circ)", data = eucalypt).fit()
regmult.summary()
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
  <th>Date:</th>             <td>Thu, 15 Jun 2023</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>11:42:06</td>     <th>  Log-Likelihood:    </th> <td> -2208.5</td>
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
        <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>     <td>  -24.3520</td> <td>    2.614</td> <td>   -9.314</td> <td> 0.000</td> <td>  -29.481</td> <td>  -19.223</td>
</tr>
<tr>
  <th>circ</th>          <td>   -0.4829</td> <td>    0.058</td> <td>   -8.336</td> <td> 0.000</td> <td>   -0.597</td> <td>   -0.369</td>
</tr>
<tr>
  <th>np.sqrt(circ)</th> <td>    9.9869</td> <td>    0.780</td> <td>   12.798</td> <td> 0.000</td> <td>    8.456</td> <td>   11.518</td>
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



### Graphical representation of the model
Graph the data, the forecast using the above model and the forecast using the
the forecast by the simple regression models seen in the "two models" exercise
in the simple regression tutorial.


```{code-cell} python
reg = smf.ols('ht~circ', data=eucalypt).fit()
regsqrt = smf.ols('ht~I(np.sqrt(circ))', data=eucalypt).fit()
```


```{code-cell} python
sel = eucalypt['circ'].argsort()
xs = eucalypt.circ.iloc[sel]
ys1 = regmult.predict()[sel]
ys2 = reg.predict()[sel]
ys3 = regsqrt.predict()[sel]
plt.plot(eucalypt['circ'], eucalypt['ht'], "o", xs, ys1, "-", xs, ys2, "--", xs, ys3, "-.")
```




    [<matplotlib.lines.Line2D at 0x125e63e50>,
     <matplotlib.lines.Line2D at 0x125e63e20>,
     <matplotlib.lines.Line2D at 0x125e63fa0>,
     <matplotlib.lines.Line2D at 0x125e6f1f0>]




    
![png](media/Part_1/1.5/p1_5_2_lab_solutions_multivariate_en_files/p1_5_2_lab_solutions_multivariate_en_28_1.png)
    

