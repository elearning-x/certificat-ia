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
  title: ''
  version: ''
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa Bedin<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

+++

# Statistical Tests

+++

```{code-cell} python
import numpy as np
from  scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

# Part I. Power Function of a Test

## I. A. For the mean $\mu$ knowing $\sigma$
We start defining the two hypothesis we want to decide:

* $H_0$: $\mu=\mu_0$
* $H_1$: $\mu\neq \mu_0$

(where in our case $\mu_0=0$ is the value we want to test.)

We then write the statistics:
$$T(Y) = \frac{\frac{1}{n} \sum_{i=1}^n Y_i-\mu_0}
{
\sqrt{
\frac{\sigma^2}
{n}
}
}$$

We finally decide between $H_0$ and $H_1$ using the following procedure:
* if $T(Y)\in [-c_{\alpha}, c_{\alpha}]$ then $H_0$ is accepted
* else, $H_0$ is rejected.

where $\alpha$ is the level of the test and $c_{\alpha}$ verifies $P(|Z|>c_{\alpha})=\alpha$ where $Z\sim \mathcal{N}(0, 1)$

We want to study the effect of sample size and true mean value on the statistical test.

1. Draw n=100 samples from a normal distribution of $\mu=0.1$ and $\sigma^2=1$ and plot the histogram of the data along with the pdf of a centered reducted normal.


```{code-cell} python
np.random.seed(0)
mu = 0.01
sigma = 1
n = 100
normal_samples = stats.norm.rvs(loc=mu, scale=sigma, size=(n,))
```


```{code-cell} python
plt.hist(normal_samples, bins=10, density=True, edgecolor='k', label='observations ~ N(0.1, 1)')

plot_samples = np.arange(normal_samples.min(), normal_samples.max(), 0.1)
normal_pdf = stats.norm.pdf(plot_samples, loc=0, scale=sigma)
plt.plot(plot_samples, normal_pdf, label='H0: N(0,1)')
plt.legend()
plt.xlabel('samples')
plt.ylabel('density')
plt.show()
```


    
![png](output_6_0.png)
    


2. Test whether the mean of the underlying distribution is 0 or not. Try different values of mu or n for which the test is NOT successful.


```{code-cell} python
mu = 0.1

sigma = 1
n = 100
normal_samples = stats.norm.rvs(loc=mu, scale=sigma, size=(n,))

alpha = 0.05
c_alpha = stats.norm.ppf(1-alpha/2)
mu_0 = 0
mu_hat = normal_samples.mean()
t_value = np.sqrt(n)*(mu_hat-mu_0)
if np.abs(t_value) <= c_alpha:
    print('The test is passed')
else:
    print('The test is NOT passed')
```

    The test is passed


The test is NOT passed, for example for :
* mu=0.3 and n=100
* mu=0.1 and n=2000

We recall that the Power function is defined as $\mu\mapsto 1-P_{\mu}(T(Y)\in [-c_{\alpha}, c_{\alpha}])$, it corresponds to the probability that the $H_0$ is rejected as a function of the true mean value $\mu$.

3. For a given $\mu$ and $\sigma$, what is the law of $N = \frac{ \sum_{i=1}^n Y_i-n\mu}
{
\sigma \sqrt{n}
}$ ? Verify visually this observation by plotting the histogram of values of $N$ along with the theoretical density (hint: you can sample an array of size Txn from $\mathcal{N}(\mu, \sigma)$ using `stats.norm.rvs(loc=mu, scale=sigma, size=(T, n))` and then compute an array of size T containing the values of N, using operations such as `np.mean` on a well-chosen `axis`)

We have $\sum_{i=1}^n Y_i \sim \mathcal{N}(n\mu, \sqrt{n}\sigma)$ so $N \sim \mathcal{N}(0,1)$


```{code-cell} python
mu = 10
sigma = 42
T = 10000 # number of simulations
n = 100
np.random.seed(0)
Y_samples = stats.norm.rvs(loc=mu, scale=sigma, size=(T, n))

N = (Y_samples-mu).sum(axis=1) / (np.sqrt(n)*sigma)
plot_samples = np.arange(N.min(), N.max(), 0.1)
normal_pdf = stats.norm.pdf(plot_samples, loc=0, scale=1)
plt.hist(N, bins=40, density=True, edgecolor='k', label='simulations of N')
plt.plot(plot_samples, normal_pdf, label='N(0,1)')
plt.legend()
plt.xlabel('N')
plt.ylabel('density')
plt.show()
```


    
![png](output_12_0.png)
    


4. Write the statistic $T(Y)$ as a function of $N$ and deduce a simple way to plot the power function as function of $\mu$ for several $n$ and for $\alpha=0.05$. (Hint: you can use `scipy.stats.norm.cdf`)

* We have $T(Y)=N- \frac{\mu_0 - \mu}{\sqrt{\frac{\sigma^2}{n}}}$
* Setting $t_{\mu}:=\frac{\mu_0 - \mu}{\sqrt{\frac{\sigma^2}{n}}}$, We deduce $P_{\mu}(T(Y) \in [-c_{\alpha}, c_{\alpha}])= P(N\in [t_{\mu}-c_{\alpha}, t_{\mu}+c_{\alpha}] )$ which can be easily calculated using `scipy.stats.norm.cdf`.


```{code-cell} python
sigma = 1
mu0 = 0
alpha = 0.05
c_alpha = stats.norm.ppf(1-alpha/2)
mu_range = np.arange(-1, 1, 0.01)

for n in [10, 100, 1000]:
    t_range = np.sqrt(n) * (mu0- mu_range) / sigma
    power_test = 1-(stats.norm.cdf(t_range+c_alpha) - stats.norm.cdf(t_range-c_alpha))
    plt.plot(mu_range, power_test, label=f'n={n}')
plt.hlines(alpha, xmin=mu_range.min(), xmax=mu_range.max(), linestyle='--', color='k', label='alpha')
plt.xlabel('mu')
plt.ylabel('power')
plt.legend()
plt.show()
```


    
![png](output_15_0.png)
    


## I. B. For the std $\sigma$ when $\mu$ is known

We start defining the two hypothesis we want to decide:

* $H_0$: $\sigma\leq\sigma_0$
* $H_1$: $\sigma> \sigma_0$

(where in our case $\sigma_0=1$ is the value we want to test.)

We then write the statistics:
$$T(Y) = \sum_{i=1}^n \frac{(Y_i-\mu)^2}
{
\sigma_0^2
}$$

We finally decide between $H_0$ and $H_1$ using the following procedure:
* if $T(Y)\in [-c_{\alpha}, c_{\alpha}]$ then $H_0$ is accepted
* else, $H_0$ is rejected.
where $\alpha$ is the level of the test and $c_{\alpha}$ verifies $P(Z>c_{\alpha})=\alpha$ where $Z\sim \mathcal{T}(n-1)$

We want to study the effect of sample size and true mean value on the statistical test.

5. Draw n=100 samples from a normal distribution of $\mu=0$ and $\sigma^2=1.1$ and plot the histogram of the data along with the pdf of a centered reducted normal.


```{code-cell} python
np.random.seed(0)
mu = 0
sigma = np.sqrt(1.1)
n = 100
normal_samples = stats.norm.rvs(loc=mu, scale=sigma, size=(n,))
```


```{code-cell} python
plt.hist(normal_samples, bins=10, density=True, edgecolor='k', label='observations ~ N(0, 1.1)')

plot_samples = np.arange(normal_samples.min(), normal_samples.max(), 0.1)
normal_pdf = stats.norm.pdf(plot_samples, loc=0, scale=sigma)
plt.plot(plot_samples, normal_pdf, label='H0: N(0,1)')
plt.legend()
plt.xlabel('samples')
plt.ylabel('density')
plt.show()
```


    
![png](output_20_0.png)
    


6. Test whether the std of the underlying distribution is less than 1 or not. Try different values of $\sigma$ or n for which the test is NOT successful.


```{code-cell} python
mu = 0
sigma = np.sqrt(1.1)
n = 100
normal_samples = stats.norm.rvs(loc=mu, scale=sigma, size=(n,))

alpha = 0.05
c_alpha = stats.chi2.ppf(1-alpha, df=n-1)

t_value = np.sum(normal_samples**2)

if t_value <= c_alpha:
    print('The test is passed')
else:
    print('The test is NOT passed')
```

    The test is passed


The test is NOT passed, for example for :
* $\sigma^2=2$, $n=100$
* $\sigma^2=1.1$, $n=5000$

7. What is the law of $W=\frac{1}{\sigma^2}\sum_{i=1}^n (Y_i - \mu)^2$ ? Verify the law of $W$ visually with a plot for a given $\mu$ and $\sigma$.

$W$ is the sum of $n$ squared independent standard normal distribution, so it follows a Chi-squared distribution of $n-1$ degrees of freedom.


```{code-cell} python
mu = 2
sigma = 10
T = 1000000 # number of simulations
n = 10
np.random.seed(0)

Y_samples = np.random.normal(loc=mu, scale=sigma, size=(T, n))

mu_hat = Y_samples.mean(axis=1)
W = ((Y_samples - mu_hat[:, None])**2).sum(axis=1)/ (sigma**2)

plot_samples = np.arange(W.min(), W.max(), 0.1)
chi_pdf = stats.chi2.pdf(plot_samples, df=n-1, loc=0, scale=1)
plt.hist(W, bins=30, density=True, edgecolor='k', label='W samples')
plt.plot(plot_samples, chi_pdf, label='Chi(n-1)')
plt.legend()
plt.xlabel('N')
plt.ylabel('density')
plt.show()
```


    
![png](output_26_0.png)
    


8.  Write the statistic $T(Y)$ as a function of $W$ and deduce a simple way to plot the power function as function of $\sigma$ for several $n$ and for $\alpha=0.05$. (Hint: you can use `scipy.stats.chi2.cdf`)


```{code-cell} python
sigma0 = 1
mu = 0
alpha = 0.05
sigma_range = np.arange(0.01, 1.99, 0.01)

for n in [10, 100, 1000]:
    c_alpha = stats.chi2.ppf(1-alpha, df=n-1)
    #t_range = np.sqrt(n) * (mu0- mu_range) / sigma
    power_test = 1-stats.chi2.cdf(sigma0**2*c_alpha/sigma_range**2, df=n-1) +\
        +stats.chi2.cdf(-sigma0**2*c_alpha/sigma_range**2, df=n-1)
    #print(stats.chi2.cdf(+sigma0**2*c_alpha/sigma_range**2, df=n-1).sum())
    plt.plot(sigma_range, power_test, label=f'n={n}')
plt.hlines(alpha, xmin=sigma_range.min(), xmax=sigma_range.max(), linestyle='--', color='k', label='alpha')
plt.xlabel('sigma')
plt.ylabel('power')
#plt.legend()
plt.show()
```


    
![png](output_28_0.png)
    


# Part II. Real Data Application

## II. A. Loading the Data

We want to analyze the traveler's household income as a function of mode choice. To do this, we'll check whether household income is equal on average for travellers choosing different modes of transport.

We'll always assume that the underlying distribution is normal, so that we can apply the methods we've seen in class.

1. Load the dataset 'travel_choice.csv' into a pandas DataFrame (hint: use `pd.read_csv`)


```{code-cell} python
df = pd.read_csv('travel_choice.csv')
df.head()
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
      <th>individual</th>
      <th>plane</th>
      <th>train</th>
      <th>bus</th>
      <th>car</th>
      <th>hinc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>45.0</td>
    </tr>
  </tbody>
</table>
</div>



## II. B. Testing the equality of the variance

We start defining the two hypothesis we want to decide:

* $H_0$: $\sigma_0^2=\sigma_1^2$
* $H_1$: $\sigma_0^2\neq\sigma_1^2$

We then write the statistics:
$$T(Y_0, Y_1) = \frac{
\sum_{i=1}^{n_0}(Y_{0,i}- \bar{Y}_0)^2/(n_0-1)
}
{
\sum_{i=1}^{n_1}(Y_{1,i}- \bar{Y}_1)^2/(n_1-1)
}$$
where:
* $n_0$ and $n_1$ are the sample size of respectively $Y_0$ and $Y_1$
* $\bar{Y}_0$ and $\bar{Y}_1$ are the empirical means of $Y_0$ and $Y_1$.


We finally decide between $H_0$ and $H_1$ using the following procedure:
* if $T(Y_0, Y_1)\in [c_{\alpha}, d_{\alpha}]$ then $H_0$ is accepted
* else, $H_0$ is rejected.
where $\alpha$ is the level of the test and $(c_{\alpha}, d_{\alpha})$ verifies $P(Z\leq c_{\alpha})=\frac{\alpha}{2}$ and $P(Z\geq d_{\alpha})=\frac{\alpha}{2}$ where $Z\sim \mathcal{F}(n_0-1, n_1-1)$


2. Write a function that takes two vector samples `Y0` and `Y1` as arguments and tests whether the two samples have the same std at the `alpha=0.05` level.


```{code-cell} python
def is_std_equal(Y0, Y1, alpha=0.05):
    Y0_bar, Y1_bar = Y0.mean(), Y1.mean()
    n0, n1 = Y0.shape[0], Y1.shape[0]
    T_n = np.sum((Y0 - Y0_bar)**2) / (n0-1)
    T_d = np.sum((Y1 - Y1_bar)**2) / (n1-1)
    T = T_n / T_d
    
    c_alpha = stats.f.ppf(alpha/2, dfn=n0-1, dfd=n1-1)
    d_alpha = stats.f.ppf(1-alpha/2, dfn=n0-1, dfd=n1-1)

    return (T>=c_alpha) and (T<=d_alpha)
```

3. For each pair mode0 and mode1, test whether the household incomes of individuals taking each of these modes have the same stds. Tip 1: to go through all possible mode pairs, you can use two "for loops": `for mode1 in ['plane', 'train', 'bus', 'car']` and `for mode2 in ['plane', 'train', 'bus', 'car']`. Tip 2: to obtain the household income of people using mode0, you can use: `Y0 = df[df[mode0]==1.]['hinc']`.


```{code-cell} python
for mode0 in ['plane', 'train', 'bus', 'car']:
    for mode1 in ['plane', 'train', 'bus', 'car']:
        if mode0 != mode1:
            Y0 = df[df[mode0]==1.]['hinc']
            Y1 = df[df[mode1]==1.]['hinc']
            if not is_std_equal(Y0, Y1):
                print(mode0, mode1)
```

## II. C. Testing the equality of the mean

We start defining the two hypothesis we want to decide:

* $H_0$: $m_0=m_1$
* $H_1$: $m_0\neq m_1$

We then write the statistics:
$$T(Y_0, Y_1) = \frac{
\bar{Y}_0-\bar{Y}_1
}
{
\sqrt{(\frac{1}{n_0}+\frac{1}{n_1})\frac{1}{n_0+n_1-2}
(\sum_{i=1}^{n_0}(Y_{0,i}- \bar{Y}_0)^2 +
\sum_{i=1}^{n_1}(Y_{1,i}- \bar{Y}_1)^2)
}}$$
where:
* $n_0$ and $n_1$ are the sample size of respectively $Y_0$ and $Y_1$
* $\bar{Y}_0$ and $\bar{Y}_1$ are the empirical means of $Y_0$ and $Y_1$.


We finally decide between $H_0$ and $H_1$ using the following procedure:
* if $|T(Y_0, Y_1)|\leq Z$ then $H_0$ is accepted
* else, $H_0$ is rejected.
where $\alpha$ is the level of the test and $c_{\alpha}$ verifies $P(|Z|> c_{\alpha})=\alpha$ where $Z\sim \mathcal{T}(n_0+n_1-2)$

4. Write a function that takes two vector samples `Y0` and `Y1` as arguments and tests whether the two samples have the same mean at the `alpha=0.05` level.


```{code-cell} python
def is_mean_equal(Y0, Y1, alpha=0.05):
    Y0_bar, Y1_bar = Y0.mean(), Y1.mean()
    n0, n1 = Y0.shape[0], Y1.shape[0]
    std0 = np.sum((Y0 - Y0_bar)**2)
    std1 = np.sum((Y1 - Y1_bar)**2)
    coeff = (1/n0+1/n1)/(n0+n1-2)
    T = (Y1_bar-Y0_bar) / np.sqrt(coeff*(std0+std1))
    
    c_alpha = stats.t.ppf(1-alpha, df=n0+n1-2)
    #return T, c_alpha#np.abs(T) <= c_alpha
    #is_equal = bool(np.abs(T) <= c_alpha)
    #return is_equal
    return np.abs(T) <= c_alpha
```

5. Find out which pair of modes has the same average.


```{code-cell} python
for mode0 in ['plane', 'train', 'bus', 'car']:
    for mode1 in ['plane', 'train', 'bus', 'car']:
        if mode0 != mode1:
            Y0 = df[df[mode0]==1.]['hinc']
            Y1 = df[df[mode1]==1.]['hinc']
            if is_mean_equal(Y0, Y1):
                print(mode0, mode1)
```

    plane car
    car plane


In the rest of the course we'll look at other methods for studying how explanatory variables (e.g. choice of means of transport) influence an observed variable (e.g. household income). To this end, we will examine different regression methodologies.

6. Add a new column called "mode" to the dataframe "df" with chosen mode ("plane", "train", "bus", or "car").


```{code-cell} python
def set_mode(row):
    for mode in ['plane', 'train', 'bus', 'car']:
        if row[mode] > 0:
            return mode
        
row = df.iloc[0]
print(set_mode(row))
df['mode'] = df.apply(set_mode, axis=1)
```

    car


7. Plot the "mean" and "standard" deviation for each mode chosen. (Tip: you can use again `sns.pointplot` with `errorbar='sd`.)


```{code-cell} python
sns.pointplot(data=df,  x='mode', y='hinc', estimator='mean', errorbar='sd', order=['train', 'bus', 'car', 'plane'])
plt.show()
```


    
![png](output_50_0.png)
    



```{code-cell} python

```
