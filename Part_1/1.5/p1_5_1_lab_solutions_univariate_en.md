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
  title: 'Solutions to Lab Session on Univariate Regression'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa Bedin<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

# Python modules
Import modules pandas (as `pd`) numpy (as `np`)
matplotlib.pyplot (as  `plt`) and statsmodels.formula.api (as `smf`)


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
```

# Simple Regression

### Data Loading
Load eucalyptus data into a pandas DataFrame `eucalypt`


```python
eucalypt = pd.read_csv("data/eucalyptus.txt", header=0, sep=";")
```

### Point Cloud
Plot the point cloud with `circ` as abscissa and `ht` as ordinate


```python
plt.plot(eucalypt['circ'], eucalypt['ht'], "o")
```




    [<matplotlib.lines.Line2D at 0x7f5d83a2eeb0>]




    
![png](p1_5_1_lab_solutions_univariate_en_files/p1_5_1_lab_solutions_univariate_en_6_1.png)
    


We can see that the points are roughly on a straight line,
so we can propose a linear regression.
For those who think it's too curved to be a straight line,
the "two models" exercise is a simple way of remedying this.

### Simple Regression
Perform a simple linear regression where `circ` is the explanatory variable
and `ht` is the variable to be explained. Store the result
in the `reg` object and 
1. summarize this modeling;
2. display the attribute containing the OLS-estimated parameters of the line;
3. display the attribute containing the estimated standard deviation of the error.


```python
reg = smf.ols('ht~circ', data=eucalypt).fit()
reg.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>ht</td>        <th>  R-squared:         </th> <td>   0.768</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.768</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   4732.</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 12 Jul 2023</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>15:29:05</td>     <th>  Log-Likelihood:    </th> <td> -2286.2</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1429</td>      <th>  AIC:               </th> <td>   4576.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  1427</td>      <th>  BIC:               </th> <td>   4587.</td>
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
  <th>Intercept</th> <td>    9.0375</td> <td>    0.180</td> <td>   50.264</td> <td> 0.000</td> <td>    8.685</td> <td>    9.390</td>
</tr>
<tr>
  <th>circ</th>      <td>    0.2571</td> <td>    0.004</td> <td>   68.792</td> <td> 0.000</td> <td>    0.250</td> <td>    0.264</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 7.943</td> <th>  Durbin-Watson:     </th> <td>   1.067</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.019</td> <th>  Jarque-Bera (JB):  </th> <td>   8.015</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.156</td> <th>  Prob(JB):          </th> <td>  0.0182</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.193</td> <th>  Cond. No.          </th> <td>    273.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
reg.params
```




    Intercept    9.037476
    circ         0.257138
    dtype: float64




```python
reg.scale
```




    1.4380406251002342



### Residuals
Plot the residuals with
1. the `circ` variable on the x-axis and the residuals on the y-axis;
2. on the x-axis, the fit \$\hat y\$$ and on the y-axis, the residuals;
3. on the x-axis, the table line number (index) and on the y-axis, the residuals.


```python
plt.plot(eucalypt['circ'], reg.resid, "o")
```




    [<matplotlib.lines.Line2D at 0x7f5d818f4f10>]




    
![png](p1_5_1_lab_solutions_univariate_en_files/p1_5_1_lab_solutions_univariate_en_13_1.png)
    


The errors according to the model are independent and all have
the same expectation of 0 and the same variance \$\sigma^2\$.
The residuals are a prediction of the errors and should have
the same properties. The variance here is not the same
(the band of points is thinner in places)
and the mean seems to fluctuate. However, these residuals
are quite satisfactory.


```python
plt.plot(reg.predict(), reg.resid, "o")
```




    [<matplotlib.lines.Line2D at 0x7f5d8186c790>]




    
![png](p1_5_1_lab_solutions_univariate_en_files/p1_5_1_lab_solutions_univariate_en_15_1.png)
    


Since we're only interpreting the visual form/appearance 
and only the abscissa scale has changed, we obtain the same
the same interpretation as in the previous graph.


```python
plt.plot(np.arange(1,eucalypt.shape[0]+1), reg.resid , "o")
```




    [<matplotlib.lines.Line2D at 0x7f5d818609a0>]




    
![png](p1_5_1_lab_solutions_univariate_en_files/p1_5_1_lab_solutions_univariate_en_17_1.png)
    


The fluctuations in the mean of the residuals can be seen again, but this graph is less suitable for the variance in this problem.

# Estimatation variability

### Data Loading
Import eucalytus data into pandas `eucalypt` DataFrame


```python
eucalypt = pd.read_csv("data/eucalyptus.txt", header=0, sep=";")
```

### Estimation on \$n=100\$ data
Create two empty lists `beta1` and `beta2`.
Perform the following steps 500 times
1. Randomly draw 100 rows from the table `eucalypt` without replacement.
2. On this draw perform a simple linear regression
   where `circ` is the explanatory variable and `ht` the variable to be explained. Store estimated parameters in `beta1` and `beta2`.


```python
beta1 = []
beta2 = []
rng = np.random.default_rng(seed=123) # fixe la graine du générateur, les tirages seront les mêmes
for k in range(500):
    lines = rng.choice(eucalypt.shape[0], size=10, replace=False)
    euca100 = eucalypt.iloc[lines]
    reg100 = smf.ols('ht~circ', data=euca100).fit()
    beta1.append(reg100.params[0])
    beta2.append(reg100.params[1])
    
```

### Variability of \$\hat \beta_2\$
Represent the variability of the random variable \$\hat \beta_2\$.


```python
plt.hist(beta2, bins=30)
plt.show()
```


    
![png](p1_5_1_lab_solutions_univariate_en_files/p1_5_1_lab_solutions_univariate_en_25_0.png)
    


This histogram is rather symmetrical, with an outlier around 0.5; apart from this outlier, it resembles the one we'd get with a normal distribution.

### Dependance of \$\hat \beta_1\$ and \$\hat \beta_2\$
Plot the \$\hat \beta_1\$ and \$\hat \beta_2\$ pairs and note the
note the variability of the estimate and the correlation
between the two parameters.


```python
plt.plot(beta1, beta2, "o")
```




    [<matplotlib.lines.Line2D at 0x7f5d8170ba90>]




    
![png](p1_5_1_lab_solutions_univariate_en_files/p1_5_1_lab_solutions_univariate_en_28_1.png)
    


Here we can see the very strong correlation (the cloud is along a straight line)
negative (the slope is negative). This result illustrates the theoretical
of the negative correlation between the two estimators.
We find our outlier indicating that one sample drawn
sample is different from the others. Among all the trees in the field, some (a significant number) are different, which is expected in this type of test.

# Two models

### Data Loading
Import eucalytus data into pandas `eucalypt` DataFrame


```python
eucalypt = pd.read_csv("data/eucalyptus.txt", header=0, sep=";")
```

### Point Cloud
Plot the point cloud with `circ` as abscissa and `ht` as ordinate
and note that the points are not exactly on a straight line
a straight line, but rather a "square root" curve.


```python
plt.plot(eucalypt['circ'], eucalypt['ht'], "o")
```




    [<matplotlib.lines.Line2D at 0x7f5d81749eb0>]




    
![png](p1_5_1_lab_solutions_univariate_en_files/p1_5_1_lab_solutions_univariate_en_34_1.png)
    


### Two simple regressions
1. Perform a simple linear regression where `circ` is the explanatory variable
   the explanatory variable and `ht` the variable to be explained.
   Store the result in the `reg` object.
2. Perform a simple linear regression where the square root of `circ` is the explanatory variable and `ht` is the explanatory variable.
   is the explanatory variable and `ht` is the variable to be explained.
   Store the result in the `regsqrt` object. You can use
   operations and functions in formulas
   (see https://www.statsmodels.org/stable/example_formulas.html)


```python
reg = smf.ols('ht~circ', data=eucalypt).fit()
regsqrt = smf.ols('ht~I(np.sqrt(circ))', data=eucalypt).fit()
```

### Comparison
Add the 2 fits (the straight line and the square root) to the scatterplot
and choose the best model.


```python
sel = eucalypt['circ'].argsort()
plt.plot(eucalypt['circ'], eucalypt['ht'], "o", eucalypt['circ'], reg.predict(), "-", eucalypt.circ.iloc[sel], regsqrt.predict()[sel], "-"  )
```




    [<matplotlib.lines.Line2D at 0x7f5d816a1550>,
     <matplotlib.lines.Line2D at 0x7f5d816a1520>,
     <matplotlib.lines.Line2D at 0x7f5d816a1760>]




    
![png](p1_5_1_lab_solutions_univariate_en_files/p1_5_1_lab_solutions_univariate_en_38_1.png)
    


Graphically, the "square root" model seems to fit the points better.
These two models can be compared using R².


```python
reg.rsquared
```




    0.7683202384330652




```python
regsqrt.rsquared
```




    0.7820637720850065



The highest R² selects the best simple regression,
indicating here that the "square root" model is better;
this is a numerical summary of our graphical observation...
