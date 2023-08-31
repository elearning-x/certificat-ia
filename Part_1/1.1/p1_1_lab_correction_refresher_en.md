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
  title: 'Solution to Probability Refresher'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa Bedin<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

+++

# Python Modules


```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, special
import pandas as pd
import seaborn as sns
from tqdm import tqdm
```

# Normal Law

In this section, we'll use the numpy and scipy packages to generate samples following a normal distribution. We will then display the histogram of the sample and use different methods to display the distribution and the cumulative function: using the explicit formula, using the scipy library and using an approximation from the histogram.

1. Create a numpy array called "normal_samples" containing N=500 samples extracted from a normal distribution of mean mu=5 and standard deviation sigma=20, using `np.random.normal` from the numpy package.
To ensure that each time you run the cell, you get exactly the same numpy array, write `np.random.seed(0)` in the first row of the cell (you can choose any seed you like).


```python
np.random.seed(0)
N = 500
mu = 5
sigma = 20
normal_samples = np.random.normal(loc=mu, scale=sigma, size=N)
```

2. Same question but using `stats.norm.rvs` from scipy instead of numpy. Don't forget to define the seed as in the previous question.


```python
np.random.seed(0)
normal_samples_scipy = stats.norm.rvs(loc=mu, scale=sigma, size=N)
```

3. Check that normal_samples and normal_samples_scipy are equal (tip: you can add the difference between the two arrays and check that it's close to zero, or you can use `np.allclose`).


```python
np.allclose(normal_samples, normal_samples_scipy)
```

4. Plot the histogram of normal samples using matplotlib's `plt.hist` (tip: you can set the arguments bins=10 and density=True).


```python
plt.hist(normal_samples, bins=10, edgecolor='k', density=True)
plt.xlabel('values')
plt.ylabel('density')
plt.show()
```

5. Calculate numerically using scipy's function ̀`stats.norm.ppf` (which gives the inverse of the cumulative distribution function) the values $x_1$ and $x_{99}$ such that if $X$ follows a normal distribution of mean mu and variance sigma, then $P(X\leq x_1) = 0.01$ and $P(X\leq x_{99}) = 0.99$.


```python
x1 = stats.norm.ppf(0.01, loc=mu, scale=sigma)
x99 = stats.norm.ppf(0.99, loc=mu, scale=sigma)
```

6. Recall the expression for the density $f$ of a normal distribution $\mathcal{N}(\mu, \sigma^2)$. Create a numpy array "pdf_express" containing the density values of $f(x)$ for $x\in[x_1, x_{99}]$ (hint: You can start by defining a numpy array "x_samples" of evenly spaced values between $x_1$ and $x_{99}$ with `np.arange` or `np.linspace`. Then you can use the usual numpy array operations such as `np.exp`)

$$f(x) = \frac{1}{\sqrt{2 \pi \sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$


```python
x_samples = np.arange(x1, x99+0.1, 0.1)
#x_samples = np.linspace(x1, x99, 1000)
pdf_express = np.exp(-(x_samples - mu)**2/(2*sigma**2)) / np.sqrt(2*np.pi * sigma**2)
```

7. Create a numpy array "pdf_scipy" containing the density values of $f(x)$ for $x\in[x_1, x_{99}]$  using `stats.norm.pdf`. Verify that "pdf_express" and "pdf_scipy" are equal. (Hint: you can use the same "x_samples" created in the previous question)


```python
pdf_scipy = stats.norm.pdf(x_samples, loc=mu, scale=sigma)
print(np.allclose(pdf_express, pdf_scipy))
```

In the following tutorials, we'll often use `stats.[law_name].pdf` to calculate the density of a law (very useful when we don't know or remember its explicit form).

8. In this question, we'd like to extract the pdf directly from the histogram. We first notice that `plt.hist` returns 3 results:
* values (size 10). In fact, this array is an estimate (based on "normal_samples" observations) of the density value at the center of the bin.
* bin edges (size 11). `(edges[1:]+edges[:-1])/2` gives the histogram centers.
* Plotting the histogram

Question: Create two tables: "bin_centers" and "pdf_estimated" containing the bin center values and the bin center density function values respectively. Plot the histogram along with the pdfs calculated with the closed form, scipy and with estimation.

NB: In the following tutorials, we'll use this density estimation technique for samples whose distribution is unknown. This will be particularly useful for tutorial 2.


```python
pdf_estimated, edges, _ = plt.hist(normal_samples, bins=10, edgecolor='k', density=True)
bin_centers = (edges[1:]+edges[:-1]) / 2
plt.plot(x_samples, pdf_express, label='closed form', linestyle='dashed')
plt.plot(x_samples, pdf_scipy, label='scipy', linestyle='dotted')
plt.plot(bin_centers, pdf_estimated, label='estimated')
plt.legend()
plt.xlabel('values')
plt.ylabel('density')
plt.show()
```

We recall the closed form of the cumulative distribution function of a normal distribution:
$$\mathbb{P}(X\leq x) = \int_{-\infty}^x f(y) dy = \frac{1}{2} [ 1+ \text{erf}(\frac{x-\mu}{\sigma \sqrt{2}})]$$
where "erf" is the "error function" and $\text{erf}(x) = \frac{2}{ \sqrt{\pi} } \int_0^x e^{-t^2} dt$ and can be computed with `scipy.special.erf`.

9. Plot the cumulative distribution obtained with:
* the closed form (use the form recalled above)
* the function `stats.norm.cdf` from scipy
* the bin_centers and cdf_estimated computed in the previous question. (hint: you can use `np.cumsum` to obtain the cumulative sum on the cdf_estimated and then multiply by the number of bins).


```python
cdf_scipy = stats.norm.cdf(x_samples, loc=mu, scale=sigma)

cdf_express = ( 1 + special.erf( (x_samples - mu)/(sigma*np.sqrt(2))) )  /2

cdf_estimated = np.cumsum(pdf_estimated)*pdf_estimated.shape[0]
plt.plot(x_samples, cdf_express, label='closed form', linestyle='dashed', color='orange')
plt.plot(x_samples, cdf_scipy, label='scipy', linestyle='dotted', color='green')
plt.plot(bin_centers, cdf_estimated, label='estimated', color='darkred')
plt.xlabel('values')
plt.ylabel('probability')
plt.legend()
plt.show()
```

# Limits of Random Variables

## Almost-Sure Convergence

We consider the following sequence of random variables defined for $n \in \mathbb{N}^*$ :
$$u_n = \sum_{i=1}^n\frac{E_i}{2^i},$$
where $E_1 \sim \mathcal{B}(\frac{1}{2})$ is a bernouilli random variable and $(E_n)_n$ are i.i.d.

1. Generate T=20000 sequences $[u_1, \dots, u_N]$ with N=100. Tip: use `np.cumsum` and be careful: to avoid numerical overflow when calculating the powers of 2, divide successively by 2 in a for loop. What's more, if you manage to vectorize the calculation on simulations (for instance, using numpy arrays of size T), the calculation should be faster...


```python
np.random.seed(0)
T, N = 20000, 100
all_En = np.random.choice([0., 1.], size=(N, T), replace=True, p=[0.5, 0.5])
print(all_En[:2, 0])
for n in range(N-1, -1, -1):
    all_En[n:] = all_En[n:] / 2.
u_n = np.cumsum(all_En, axis=0)
u_n.shape, all_En[:2, 0]
```

2. Plot several sequences $[u_1, \dots, u_N]$ as a function of $n \in [1, N]$. What convergence does this plot illustrate?


```python
for sim in tqdm(range(1000)):
    plt.plot(np.arange(1, N+1), u_n[:, sim], alpha=0.5)
plt.xlabel('n')
plt.ylabel('u_n')
plt.title('Multiple Simulations of u_n')
plt.show()
```

We observe that all trajectories converge. This illustrates asymptotically sure convergence.

3. Plot the probability $\mathbb{P}(|u_n-u_N|<\epsilon)$ for $n \in[1, N]$ with $N=100$ and $\epsilon=0.001$. Tip: for each n, count the number of simulations such that $|u_n[s] -u_N[s]|<\epsilon$ and divide by T. (NB: you can vectorize the operation using `np.sum` and `np.abs`).

What convergence does this plot illustrate?


```python
epsilon = 0.001
prob = np.sum(np.abs(u_n - u_n[-1][np.newaxis])<epsilon, axis=1) / T
plt.plot(np.arange(1, N+1), prob)
plt.xlabel('n')
plt.ylabel('P(|u_n-u_N|<epsilon)')
plt.show()
```

This graph illustrates convergence in probability (in fact, convergence a.s. implies convergence in probability).

4. Plot the histogram of $u_N$. What convergence does this plot illustrate? To which law does $u_n$ converge?


```python
plt.hist(u_n[-1], bins=30, density=True, edgecolor='k')
plt.xlabel('u_n value')
plt.ylabel('Density')
plt.show()
```

This plot illustrates convergence in law (in fact, convergence a.s. implies convergence in probability which implies convergence in law). In fact $u_n$ converges a.s. to a uniform random variable.

## Convergence in Probability

We define the following sequence of independent variables for $n \in \mathbb{N}^*$, $B_n\sim \mathcal{B}(\frac{1}{n})$ and we define an arbitrary (let's say uniform) random variable $X\sim \mathcal{U}([0, 1])$. We are interested in the following sequence for $n \in \mathbb{N}^*$ :
$$X_n = X+ B_n.$$

5. Plot the probability $\mathbb{P}(|X_n-X|<\frac{1}{2})$ for $n \in[1, N]$ with $N=100$. What is the convergence of $X_n$ ?


```python
N = 100
N_range = np.arange(1, N+1)
plt.plot(N_range, 1. / N_range)
plt.xlabel('n')
plt.ylabel('P(|X_n-X|<epsilon)')
plt.show()
```

$X_n$ converges in probability to $X$. Note that if the $X_n$ are independent, then they cannot converge in probability to a random variable $X$ unless $X$ is a.s. constant...

6. Simulate for several $(X_n-X)_{n\leq N}$ T times for $N=100$ (Optional tip: if you want to vectorize your operations you can, for instance, use a "for loop" over N and use `np.random.choice` with arguments `p=[1-1/n, 1/n]` and `size=T` and append the result to a list. Then use `np.stack` to obtain a numpy array of size NxT from the completed list) 


```python
np.random.seed(0)
T, N = 2000, 100
all_Xn = []
X = np.random.uniform(0, 1, size=T)
for n in range(1, N+1):
    Xn = X + np.random.choice([0, 1], size=T, replace=True, p=[1-1/n, 1/n])
    all_Xn.append(Xn)
all_Xn = np.stack(all_Xn)
all_Xn.shape
```

7. Plot some of the simulations as in II.A.2 and justify why $X_n$ does not converge almost surely to $X$.


```python
for sim in tqdm(range(100)):
    plt.plot(np.arange(1, N+1), all_Xn[:, sim]-X[sim], linestyle='', marker='.', alpha=0.2)
plt.xlabel('n')
plt.ylabel('u_n')
plt.title('Multiple Simulations of u_n')
plt.show()
```

Although the gaps between the observation of $1+X$ terms will become large, the sequence will always bounce between $X$ and $1+X$ with some nonzero frequency.

## Convergence in Law

We consider the following sequence of independent binomial random variables defined for $n \in \mathbb{N}^*$ :
$$X_n \sim \mathcal{B}(n, \frac{\lambda}{n}),$$
where $\lambda>0$.

8. Simulate for several $(X_n)_{n\leq N}$ T times for $N=1000$ and $\lambda=10$ (Optional tip: if you want to vectorize your operations, you can use a for loop over N and use `np.random.binomial` with arguments `p=lambd/n` and `size=T` and append the result to a list. Then use `np.stack` to obtain a numpy array of size NxT from that list)


```python
np.random.seed(0)
lambd = 10
N = 1000
T = 20000
all_Xn = []
for n in range(1, N+1):
    Xn= np.random.binomial(n, min(1, lambd / n), size=T)
    all_Xn.append(Xn)
all_Xn = np.stack(all_Xn)
all_Xn.shape
```

9. Plot the proability mass function (pmf) of a Poisson distribution with parameters $\lambda$, and the probabilities of the observed values for simulations of $X_N$. Hints:
* to plot the pmf, it's almost like question I.A.7, except that you should use `scipy.poisson.pmf`.
* to plot probabilities, you can use `sns.histplot` with the argument 'stat=probability' from the seaborn package (optional: you can also use `plt.hist` with the argument "density=False" and a well-chosen "weights").


```python
xmin = stats.poisson.ppf(0.000001, mu=lambd)
xmax = stats.poisson.ppf(0.999999, mu=lambd)
x_samples = np.arange(xmin, xmax, 1)
pdf_poisson = stats.poisson.pmf(x_samples, mu=lambd)

#weights = np.ones_like(all_Xn[-1])/len(all_Xn[-1])
#plt.hist(all_Xn[-1], bins=100, weights=weights, edgecolor='k')
sns.histplot(all_Xn[-1], stat='probability', label='u_n histogram')
plt.plot(x_samples, pdf_poisson, label=f'density of poiss({lambd})')
plt.xlabel('u_n')
plt.legend()
plt.show()
```

10. Plot $X_n - X_{n-1}$ as a function of $n$ for several simulations and justify the $X_n$ that do not converge almost-surely.


```python
for sim in tqdm(range(3)):
    plt.plot(np.arange(2, N+1), all_Xn[1:, sim]-all_Xn[:-1, sim], marker='.', linestyle='')#, alpha=0.2)
plt.xlabel('n')
plt.ylabel('u_n - u_{n-1}')
plt.title('Multiple Simulations of u_n - u_{n-1}')
plt.show()
```

## Strong Law of Large Numbers and Central Limit Theorem

We simulate a throw of dice with a random variable $X$ of uniform distribution on {1, 2, 3, 4, 5, 6}.

11. What is the expectation $m$ and standard deviation $\sigma$ of $X$? Hint: you can calculate numerically using `np.mean` and `np.std` applied to the array `np.arange(1, 7)`.


```python
mu = np.arange(1, 7).mean()
sigma = np.arange(1, 7).std()
print(f'mean={mu}, std={sigma:.2f}')
```

12. Generate a sample of $X$ of size $N=5000$ with `np.random.choice`. Use `np.cumsum` and `np.arange` to obtain an array of
successive values of $p_n = \frac{1}{n}\sum_{i=1}^n X_i$ for $n \in [1, N]$.


```python
N = 5000
T = 20000
X = np.random.choice(np.arange(1, 7), size=(N, T))
p_n = np.cumsum(X, axis=0) / np.arange(1, N+1)[:, np.newaxis]
```

13. Plot several simulations of $[p_1, \dots, p_N]$ as a function of $n\in [1, N]$, as well as a horizontal line to identify the expectation of $X$ (hint: use `plt.hlines`). What theorem does this plot illustrate? What convergence is associated with this theorem?


```python
for sim in tqdm(range(20)):
    plt.plot(np.arange(1, N+1), p_n[:, sim], alpha=0.5)
plt.hlines(y=mu, xmin=1, xmax=N, label='mu', color='k')
plt.xlabel('n')
plt.ylabel('p_n')
plt.legend()
plt.title('Multiple Simulations of p_n')
plt.show()
```

This plot illustrate the strong law of large numberss that states that $p_n$ converges almost surely to the expectation of $X$.

14. Plot several simulations of $[p_1-m, \dots, \sqrt{N}(p_N-m)]$ as a function of $n\in [1, N]$ (where $m$ is the expectation of $X$) and justify what $\sqrt{n}(p_n - m)$ doesn't converge almost-surely.


```python
for sim in tqdm(range(20)):
    plt.plot(np.arange(1, N+1), np.sqrt(np.arange(1, N+1))*(p_n[:, sim]-mu), alpha=0.5)
plt.xlabel('n')
plt.ylabel('sqrt{n}(p_n-mu)')
plt.title('Multiple Simulations of p_n')
plt.show()
```

15. Plot the histogram of $\sqrt{N}(p_N-m)$ with the fdc of $\mathcal{N}(0, \sigma^2)$. Which theorem does this plot illustrate?


```python
x1 = stats.norm.ppf(0.01, loc=0, scale=sigma)
x99 = stats.norm.ppf(0.99, loc=0, scale=sigma)
x_samples = np.linspace(x1, x99, 100)
pdf_scipy = stats.norm.pdf(x_samples, loc=0, scale=sigma)

plt.hist(np.sqrt(N)*(p_n[-1]-mu), bins=30, density=True, edgecolor='k')
plt.plot(x_samples, pdf_scipy, label='N(0, sigma^2)')
plt.xlabel('p_N')
plt.ylabel('density')
plt.legend()
plt.show()
```

This plot illustrates the central limit theorem: $\sqrt{n}(p_n-m)$ converges to a normal distribution.

# Real Data Analysis

This section introduces a dataset we'll be studying in the next lab sessions. In this section we'll have the opportunity to see how to use the pandas library to manipulate data structure.

For information the data comes from: https://www.met.ie/climate/30-year-averages

1. Load into a pandas dataframe named "df" the dataset minimal_temperature_GB.txt with `pd.read_csv` and display the first 5 rows by writing `df.head()` on the last cell.


```python
df = pd.read_csv('minimal_temperature_GB.txt')
df.head()
```

2. Print the length L of the dataframe and the columns of the dataframe.


```python
print(f'The length of the dataframe is: {len(df)}')
print(f'The columns of the dataframe are: {list(df.columns)}')
```

The aim of questions 3 to 6 is to transform the dataframe into a dataframe of length 12L with 2 columns: the month number "month", the minimum temperature for that month "Tmin".

The steps described in questions 3 to 6 are not the only way to achieve this, so you can skip these questions and adopt your own steps.

3. Write a "get_month_name" function that takes as input an integer month_id between 1 and 13 and returns the name of the corresponding column. (Hint: you can use python "f-strings" or the python method `.format()`)


```python
def get_month_name(month_id):
    return f'm{month_id}Tmin'
print(get_month_name(1))
```

4. Write a "get_month_df" function that takes month_id as input and returns a pandas DataFrame of length L, but with only two columns: one containing the month_id and the other containing the minimum temperature. (Tip: you can start by creating two lists or arrays of equal length L: one containing the minimum temperature for the month in question and the other containing only the "month_id". Then create a pandas dataframe using `pd.DataFrame` using these lists)


```python
def get_month_df(month_id):
    temp_column = df[get_month_name(month_id)]
    month_id_column = [month_id] * len(temp_column)
    df_month = pd.DataFrame({'month': month_id_column, 'Tmin': temp_column})
    return df_month

df_month = get_month_df(1)
df_month.head()
```

5. Create a pandas dataframe of length 12L with only two columns: one containing the month_id and the other containing the minimum temperature. (Hint: you can create an "all_months" list of 12 dataframes by calling the function get_month_df(month_id) for month_id ranging from 1 to 13. You can then use pd.concat(all_months) to create the expected dataframe).


```python
all_months = []
for month in range(1, 13):
    df_month = get_month_df(month)
    all_months.append(df_month)
df_month = pd.concat(all_months)
df_month.head()
```

6. Use the `sns.boxplot` function from the `seaborn` package to display quantiles of minimum temperature as a function of month.


```python
sns.boxplot(data=df_month, x='month', y='Tmin')
plt.show()
```
