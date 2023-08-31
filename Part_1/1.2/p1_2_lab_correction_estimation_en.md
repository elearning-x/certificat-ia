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
  title: 'Solution to Point Estimation'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa Bedin<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

+++

Point estimation involves estimating a parameter of a distribution (e.g. the mean, or the parameters of the parametric density function...) from the observation of a sample. In this tutorial, we'll be:
* observing numerically that the empirical mean is an asymptotically normal estimator of the mean,
* looking at several methods for performing point estimation on simulated data,
* comparing several fitted distributions on real data.

# Python Modules


```{code-cell} python
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.special
import matplotlib.pyplot as plt
import random
```

# Empirical Mean
## Data Simulation

We are interested in the age of the population in a given location. We assume that the population is bimodal, with each mode following a Poisson distribution centered on 40 and 10 respectively. There are twice as many individuals in the first mode as in the second.

1. Simulate the age of the population for 300000 individuals with `stats.poisson.rvs`. You can use `np.concatenate` to merge the two tables.


```{code-cell} python
np.random.seed(0)
population_ages1 = stats.poisson.rvs(loc=0, mu=30, size=2000000)
population_ages2 = stats.poisson.rvs(loc=0, mu=10, size=1000000)
population_ages = np.concatenate((population_ages1, population_ages2))
```

2. Plot the histogram of age $X$ in the population (you can use `plt.histogram` with the argument `density=True` and well-chosen `bins`).


```{code-cell} python
plt.hist(population_ages, bins=30, density=True, edgecolor="k")
plt.ylabel('density')
plt.xlabel('age')
plt.show()
```

## Point Estimation

3. We assume we can access only $n=100$ individuals. Create a vector named `sample_ages` containing the ages of these individuals (you can use `np.random.choice` and set the parameter size to 100).


```{code-cell} python
np.random.seed(0)
n = 100
sample_ages = np.random.choice(a=population_ages, size=n)
```

4. We note by $(x_1, \dots, x_n)$ the ages of the $n$ observed individuals. Print both the empirical mean $\mu_n = \frac{1}{n}\sum_{i=1}^n x_i$ and the theoretical mean $\mu=\mathbb{E}[X]$.


```{code-cell} python
print(f"estimated mean: {sample_ages.mean():.2f}")
print(f"real mean: {population_ages.mean():.2f}")
```

## Asymptotic Normality

Now we would like to study the distribution of the empirical mean $\mu_n$ for a given size of observed populations. To do so, we will sample populations of size `n`, `T` times. Then we will compute the `number_trials` corresponding empirical means.

5. Create a table of shape Txn with elements sampled from `population_ages`. You can use the `np.random.choice` with the argument `size=(T, n)`. You can set `T=10000` and `n=100`.


```{code-cell} python
np.random.seed(0)
T = 10000 # number of trials
n = 100 # population size

trials_samples = np.random.choice(a=population_ages, size=(T, n))
```

6. Create a vector of size `T` containing the emperical mean of each trial. You can use `np.mean` with the argument `axis=1`.


```{code-cell} python
mu_n = np.mean(trials_samples, axis=1)
```

7. Plot the histogram of $\sqrt{n}(\mu_n-\mu)$.


```{code-cell} python
mu_star = population_ages.mean()
plt.hist(np.sqrt(n)*(mu_n-mu_star), bins=30, density=True, edgecolor="k")
plt.xlabel('normalized empirical mean')
plt.ylabel('density')
plt.show()
```

8. What do you observe when you change `n` ? What is the Fischer information of the age of the population ?

We observe that $\sqrt{n}(\mu_n-\mu) \sim \mathcal{N}(0, V)$ where $V$ does not depend on $n$ **(asymptotic normality)**.


```{code-cell} python
fischer = 1 / np.std(np.sqrt(n)*(mu_n-mu_star))
print(f'fischer={fischer:.2f}')
```

# MOM and MLE applied to Gamma Distribution

The probability density function of Gamma distribution with shape parameter $k$ and scale parameter $\theta$ is:
$$f:x\in\mathbb{R}_+ \mapsto \frac{1}{\Gamma(k)\theta^k} x^{k-1} e^{-\frac{x}{\theta}},$$
where the $\Gamma$ function is defined as follow:
$$\Gamma:z\in\mathbb{R}_+ \mapsto \int_0^{+\infty} t^{z-1} e^{-t}.$$

## Data Simulation

1. Simulate `n=1000` observations following a gamma distribution with parameter $\theta=0.2$ and $k=3$. You can use `scipy.stats.gamma.rvs` with argument `size=n`, `a=k` and `scale=theta` to draw `n` samples from a gamma distribution with paramters `k`, `theta`.


```{code-cell} python
np.random.seed(0) 
k_real, theta_real = 3, 0.2
n = 1000
gamma_sample = stats.gamma.rvs(a=k_real, scale=theta_real, size=n)
```

2. Plot the histogram of these samples along with the probability density function of the gamma law (you can compute the pdf of a gamma function `scipy.stats.gamma.pdf`. You can set the argument `x` to a range from 0 to 2 with steps 0.001, using the `numpy` function `np.arange`).


```{code-cell} python
# calculate pdf
plot_samples = np.arange(0, 2, 0.001)
pdf_real = stats.gamma.pdf(x=plot_samples, a=k_real, scale=theta_real)

# plot histogram and pdf
plt.hist(gamma_sample, density=True, bins=30, label='histogram',edgecolor="k")
plt.plot(plot_samples, pdf_real, label='pdf')
plt.xlabel('x')
plt.ylabel('density')
plt.legend()
plt.show()
```

## Method of Moments

3. Show that if $X$ follows a Gamma distribution of parameters $(k, \theta)$ the two first orders moments verify:
$$\mu_1:=\mathbb{E}[X] = k \theta$$
$$\mu_2:=\mathbb{E}[X^2] = (k+1)k \theta^2$$

4. Show that the parameters $(k, \theta)$ can be expressed thanks to the two first order momentums:
$$k=\frac{\mu_1^2}{\mu_2-\mu_1^2}$$
$$\theta=\frac{\mu_2}{\mu_1}-\mu_1$$

5. Estimate `k_MOM` and `theta_MOM`.


```{code-cell} python
mu1 = np.mean(gamma_sample)
mu2 = np.mean(gamma_sample**2)

k_MOM = mu1**2/(mu2 - mu1**2)
theta_MOM = mu2/mu1 - mu1
print("k_MOM = ", k_MOM)
print("theta_MOM = ", theta_MOM)
```

## Maximum Likelihood Estimation (numpy)

6. Write the log-likelihood function for the samples $x=(x_1, \dots, x_n)$.

$$\log f(x) = (k-1)\sum_i\log x_i - n \log \Gamma(k) - n k \log \theta - \frac{1}{\theta} \sum_i x_i$$
$$\log f(x) =
n((k-1)\overline{\log x} - \log \Gamma(k) - k \log \theta - \frac{\overline{x}}{\theta})$$

7. What $\theta$ minimizes the log-likelihood ?

$$\theta = \frac{\overline{x}}{k}$$

8. Plot the log-likelihood with respect to $k$. You can use `scipy.special.gamma` to get the gamma function, and you can create a vector `k_samples` containing values from 2.9 to 3.5 with step 0.00001 (use `np.arange`).


```{code-cell} python
k_sample = np.arange(2.9, 3.5, 0.00001)
n = gamma_sample.shape[0]

log_x_bar = np.log(gamma_sample).mean()
x_bar = gamma_sample.mean()

theta_opt = x_bar / k_sample

log_likelihood = (k_sample-1)*log_x_bar

log_likelihood -= x_bar / theta_opt

log_likelihood -= np.log(scipy.special.gamma(k_sample))

log_likelihood -= k_sample*np.log(theta_opt)

log_likelihood *= n
```


```{code-cell} python
plt.plot(k_sample, log_likelihood)
plt.xlabel('k')
plt.ylabel('log likelihood')
plt.show()
```

9. Estimate `k_MLE` and `theta_MLE`. (you cane use `np.argmax` to get the index of the maximum of the log-likelihood)


```{code-cell} python
k_MLE = k_sample[log_likelihood.argmax()]
theta_MLE = x_bar / k_MLE
print("k_MLE = ", k_MLE)
print("theta_MLE = ", theta_MLE)
```

## Maximum Likelihood Estimation (scipy)

10. Use the `scipy.stats.gamma.fit` function to estimate the parameters of the gamma distribution.


```{code-cell} python
k_MLE_s, loc_MLE_s, theta_MLE_s = stats.gamma.fit(gamma_sample)
```

11. Plot the pdfs of all the estimations (MOM, MLE numpy, MLE scipy) along with the pdf of the real distribution.


```{code-cell} python
# calculate pdfs
plot_samples = np.arange(0, 2, 0.001) # where we evaluate the distibution
pdf_real = stats.gamma.pdf(x=plot_samples, a=k_real, scale=theta_real)
pdf_MOM = stats.gamma.pdf(x=plot_samples, a=k_MOM, scale=theta_MOM)
pdf_MLE = stats.gamma.pdf(x=plot_samples, a=k_MLE, scale=theta_MLE)
pdf_MLE_s = stats.gamma.pdf(x=plot_samples, a=k_MLE_s, loc=loc_MLE_s, scale=theta_MLE_s)

plt.plot(plot_samples, pdf_real, label='real')
plt.plot(plot_samples, pdf_MOM, label='MOM')
plt.plot(plot_samples, pdf_MLE, label='MLE (numpy)')
plt.plot(plot_samples, pdf_MLE_s, label='MLE (scipy)')
plt.legend()
plt.xlabel('x')
plt.ylabel('pdf')

plt.show()
```

12. Print the entropy between the estimated distributions and the real distribution. (You can use `scipy.stats.entropy(p,q)` which gives $\sum_{i=1}^n p_i \log(\frac{p_i}{q_i})$ where $p=(p_1, \dots, p_n)$ [resp. $q=(q_1, \dots, q_n)$] are the values of estimated [resp. real] pdf evaluated on $n$ points).


```{code-cell} python
print(f'entropy real: {stats.entropy(pdf_real, pdf_real):.6f}')
print(f'entropy MOM : {stats.entropy(pdf_MOM, pdf_real):.6f}')
print(f'entropy MLE (numpy): {stats.entropy(pdf_MLE, pdf_real):.6f}')
print(f'entropy MLE (scipy): {stats.entropy(pdf_MLE_s, pdf_real):.6f}')
```

# Real Data Application

## Loading data
1. Load into a `pandas` data frame named `df_monthly`, the `minimal_temperature_GB.txt` table of the 30-year average monthly minimum temperature at 84291 locations in Great Britain. You can use `pd.read_csv`. (for information, the data come from: https://www.met.ie/climate/30-year-averages)


```{code-cell} python
df_monthly = pd.read_csv('minimal_temperature_GB.txt')#[:100]
df_monthly.head()
```

2. Draw the histogram (with 10 bins) for each month and extract the pdf for each bin. To extract the pdf per bin, you'll notice that `plt.hist` returns 3 results:
* values (size 10)
* bin edges (size 11). `(edges[1:]+edges[:-1])/2` gives bin centers.
* The plot


```{code-cell} python
for month in [f'm{k}Tmin' for k in range(1, 3)]:
    month_sample = df_monthly[month]
    
    pdf_values, pdf_edges, _ = plt.hist(month_sample, bins=10, density=True, label='histogram',edgecolor="k")
    pdf_bins = (pdf_edges[1:]+pdf_edges[:-1])/2
    
    plt.plot(pdf_bins, pdf_values, label='point-wise pdf')
    plt.xlabel('minimum temperature')
    plt.ylabel('density')
    plt.title(month)
    plt.show()
```

## Parameter Estimation
3. Adjust the optimal parameters of the **normal** distribution using MLE on each month's data. You need to record the following scores:
* store in a list `entropy_norm`,
the entropy between the estimated pdf and the actual pdf over the 10 bins (the actual pdf is obtained as in the previous question).
* store in a list `lll_norm`, the log-likelihood of the estimated pdf on the observed data. You can use `np.log` and `np.mean` successively on the pdf calculated on the real data.

You can also plot the estimated pdf with histograms by month.


```{code-cell} python
month_lst = [f'm{k}Tmin' for k in range(1, 13)]
entropy_norm, lll_norm, loc_norm, scale_norm = [], [], [], []

for month in month_lst:
    month_sample = df_monthly[month]
    
    # == MLE ==
    loc, scale = stats.norm.fit(month_sample)
    loc_norm.append(loc)
    scale_norm.append(scale)
    
    # == PDFs ==
    pdf_values, pdf_edges, _ = plt.hist(month_sample, bins=10, density=True, alpha=0.5,edgecolor="k")
    pdf_bins = (pdf_edges[1:]+pdf_edges[:-1])/2
    pdf_norm = stats.norm.pdf(x=pdf_bins, loc=loc, scale=scale)
    
    entropy = stats.entropy(pdf_norm[pdf_values>0], pdf_values[pdf_values>0])
    entropy_norm.append(entropy)
    
    # == log-likelihood ==
    pdf_norm_ll = stats.norm.pdf(x=month_sample, loc=loc, scale=scale)
    lll = np.mean(np.log(pdf_norm_ll))
    lll_norm.append(lll)

    # == plots ==
    plt.plot(pdf_bins, pdf_values, label='pdf on bins')
    plt.plot(pdf_bins, pdf_norm, label='MLE')
    plt.xlabel('minimum temperature')
    plt.ylabel('density')
    plt.legend()
    
    plt.show()
```

4. Do the same with the **cauchy** distribution. Be sure to use the suffix `_cauchy` (not `_norm`) to store the scores.


```{code-cell} python
month_lst = [f'm{k}Tmin' for k in range(1, 13)]
entropy_cauchy, lll_cauchy, loc_cauchy, scale_cauchy = [], [], [], []

for month in month_lst:
    month_sample = df_monthly[month]
    
    # == MLE ==
    loc, scale = stats.cauchy.fit(month_sample)
    loc_cauchy.append(loc)
    scale_cauchy.append(scale)
    
    # == PDFs ==
    pdf_values, pdf_edges, _ = plt.hist(month_sample, bins=10, density=True, alpha=0.5,edgecolor="k")
    pdf_bins = (pdf_edges[1:]+pdf_edges[:-1])/2
    pdf_cauchy = stats.cauchy.pdf(x=pdf_bins, loc=loc, scale=scale)
    
    # == entropy ==
    entropy = stats.entropy(pdf_cauchy[pdf_values>0], pdf_values[pdf_values>0])
    entropy_cauchy.append(entropy)
    
    # == log-likelihood ==
    pdf_cauchy_ll = stats.cauchy.pdf(x=month_sample, loc=loc, scale=scale)
    lll = np.mean(np.log(pdf_cauchy_ll))
    lll_cauchy.append(lll)

    plt.plot(pdf_bins, pdf_values, label='pdf on bins')
    plt.plot(pdf_bins, pdf_cauchy, label='MLE')
    plt.xlabel('minimum temperature')
    plt.ylabel('density')
    plt.legend()
    
    plt.show()
```

5. Plot the entropy per month for the two estimated distributions.


```{code-cell} python
plt.plot(np.arange(1, len(entropy_norm)+1), entropy_norm, label='normal MLE')
plt.plot(np.arange(1, len(entropy_norm)+1), entropy_cauchy, label='cauchy MLE')
plt.legend()
plt.xlabel('month')
plt.ylabel('entropy')
plt.show()
```

6. Plot the log-likelihood per month for the two estimated distributions.


```{code-cell} python
plt.plot(np.arange(1, len(lll_norm)+1), lll_norm, label='normal MLE')
plt.plot(np.arange(1, len(lll_norm)+1), lll_cauchy, label='cauchy MLE')
plt.legend()
plt.xlabel('month')
plt.ylabel('log-likelihood')
plt.show()
```
