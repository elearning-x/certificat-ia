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

```{list-table} 
:header-rows: 0
:widths: 33% 34% 33%

* - ![Logo](media/logo_IPParis.png)
  - Lisa BEDIN<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER
  - Licence CC BY-NC-ND
```

+++

In this tutorial, we will explore two key aspects of probability theory: the convergence of random variables and the crucial distinction between different types of convergence.
We will delve into the practical implications of these concepts within the context of two prominent theorems: the Strong Law of Large Numbers and the Central Limit Theorem.

# Python Modules


```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from tqdm import tqdm
```

# Limits of Random Variables

In the first part of this tutorial, we will lay the foundation by examining the limits of random variables.
We will simulate sequences of random variables and visualize their behavior to gain insight into convergence properties.

## Almost-Sure Convergence

We consider the following sequence of random variables defined for $n \in \mathbb{N}^*$ :
$$u_n = \sum_{i=1}^n\frac{E_i}{2^i},$$
where $E_1 \sim \mathcal{B}(\frac{1}{2})$ is a bernouilli random variable and $(E_n)_n$ are i.i.d.

1. Generate T=20000 sequences $[u_1, \dots, u_N]$ with N=100. Tip: use `np.cumsum` and be careful: to avoid numerical overflow when calculating the powers of 2, divide successively by 2 in a for loop. What's more, if you manage to vectorize the calculation on simulations (for instance, using numpy arrays of size T), the calculation should be faster...


```{code-cell} python
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


```{code-cell} python
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


```{code-cell} python
epsilon = 0.001
prob = np.sum(np.abs(u_n - u_n[-1][np.newaxis])<epsilon, axis=1) / T
plt.plot(np.arange(1, N+1), prob)
plt.xlabel('n')
plt.ylabel('P(|u_n-u_N|<epsilon)')
plt.show()
```

This graph illustrates convergence in probability (in fact, convergence a.s. implies convergence in probability).

4. Plot the histogram of $u_N$. What convergence does this plot illustrate? To which law does $u_n$ converge?


```{code-cell} python
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


```{code-cell} python
N = 100
N_range = np.arange(1, N+1)
plt.plot(N_range, 1. / N_range)
plt.xlabel('n')
plt.ylabel('P(|X_n-X|<epsilon)')
plt.show()
```

$X_n$ converges in probability to $X$. Note that if the $X_n$ are independent, then they cannot converge in probability to a random variable $X$ unless $X$ is a.s. constant...

6. Simulate for several $(X_n-X)_{n\leq N}$ T times for $N=100$ (Optional tip: if you want to vectorize your operations you can, for instance, use a "for loop" over N and use `np.random.choice` with arguments `p=[1-1/n, 1/n]` and `size=T` and append the result to a list. Then use `np.stack` to obtain a numpy array of size NxT from the completed list) 


```{code-cell} python
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


```{code-cell} python
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


```{code-cell} python
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
* to plot the pmf, it's almost like question I.A.7 in the previous tutorial, except that you should use `scipy.poisson.pmf`.
* to plot probabilities, you can use `sns.histplot` with the argument 'stat=probability' from the seaborn package (optional: you can also use `plt.hist` with the argument "density=False" and a well-chosen "weights").


```{code-cell} python
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


```{code-cell} python
for sim in tqdm(range(3)):
    plt.plot(np.arange(2, N+1), all_Xn[1:, sim]-all_Xn[:-1, sim], marker='.', linestyle='')#, alpha=0.2)
plt.xlabel('n')
plt.ylabel('u_n - u_{n-1}')
plt.title('Multiple Simulations of u_n - u_{n-1}')
plt.show()
```

# Strong Law of Large Numbers and Central Limit Theorem

Moving on to the second part, we will apply our knowledge of convergence to explore two cornerstone theorems in probability theory.
We will not only understand the statements of these theorems but also use simulations to witness their practical significance.


We simulate a throw of dice with a random variable $X$ of uniform distribution on {1, 2, 3, 4, 5, 6}.

11. What is the expectation $m$ and standard deviation $\sigma$ of $X$? Hint: you can calculate numerically using `np.mean` and `np.std` applied to the array `np.arange(1, 7)`.


```{code-cell} python
mu = np.arange(1, 7).mean()
sigma = np.arange(1, 7).std()
print(f'mean={mu}, std={sigma:.2f}')
```

12. Generate a sample of $X$ of size $N=5000$ with `np.random.choice`. Use `np.cumsum` and `np.arange` to obtain an array of
successive values of $p_n = \frac{1}{n}\sum_{i=1}^n X_i$ for $n \in [1, N]$.


```{code-cell} python
N = 5000
T = 20000
X = np.random.choice(np.arange(1, 7), size=(N, T))
p_n = np.cumsum(X, axis=0) / np.arange(1, N+1)[:, np.newaxis]
```

13. Plot several simulations of $[p_1, \dots, p_N]$ as a function of $n\in [1, N]$, as well as a horizontal line to identify the expectation of $X$ (hint: use `plt.hlines`). What theorem does this plot illustrate? What convergence is associated with this theorem?


```{code-cell} python
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


```{code-cell} python
for sim in tqdm(range(20)):
    plt.plot(np.arange(1, N+1), np.sqrt(np.arange(1, N+1))*(p_n[:, sim]-mu), alpha=0.5)
plt.xlabel('n')
plt.ylabel('sqrt{n}(p_n-mu)')
plt.title('Multiple Simulations of p_n')
plt.show()
```

15. Plot the histogram of $\sqrt{N}(p_N-m)$ with the fdc of $\mathcal{N}(0, \sigma^2)$. Which theorem does this plot illustrate?


```{code-cell} python
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

