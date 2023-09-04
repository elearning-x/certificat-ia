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
<span>Lisa BEDIN<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

+++

Point estimation is a fundamental concept in statistics, where we endeavor to estimate a parameter of a distribution based on observations from a sample.
This parameter could be anything from the mean, variance, or even the parameters of a parametric density function that best describes the underlying population distribution.
In this tutorial, we will explore various methods for making these estimations.

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

In this first section, we will numerically demonstrate that the empirical mean approaches a normal distribution as the sample size increases—an essential insight for understanding the behavior of estimators.
To make this concept more tangible, we will analyze a bimodal distribution.
Bimodal distributions are often encountered in real-world scenarios, and they present unique challenges compared to the more common theoretical distributions like the Normal or Bernoulli distributions discussed in our previous tutorials.

## Data Simulation

We are interested in the age of the population in a given location. We assume that the population is bimodal, with each mode following a Poisson distribution centered on 40 and 10 respectively. There are twice as many individuals in the first mode as in the second.

1. Simulate the age of the population for 300000 individuals with `stats.poisson.rvs`. You can use `np.concatenate` to merge the two tables.


```{code-cell} python

```

2. Plot the histogram of age $X$ in the population (you can use `plt.histogram` with the argument `density=True` and well-chosen `bins`).


```{code-cell} python

```

## Point Estimation

3. We assume we can access only $n=100$ individuals. Create a vector named `sample_ages` containing the ages of these individuals (you can use `np.random.choice` and set the parameter size to 100).


```{code-cell} python

```

4. We note by $(x_1, \dots, x_n)$ the ages of the $n$ observed individuals. Print both the empirical mean $\mu_n = \frac{1}{n}\sum_{i=1}^n x_i$ and the theoretical mean $\mu=\mathbb{E}[X]$.


```{code-cell} python

```

## Asymptotic Normality

Now we would like to study the distribution of the empirical mean $\mu_n$ for a given size of observed populations. To do so, we will sample populations of size `n`, `T` times. Then we will compute the `number_trials` corresponding empirical means.

5. Create a table of shape Txn with elements sampled from `population_ages`. You can use the `np.random.choice` with the argument `size=(T, n)`. You can set `T=10000` and `n=100`.


```{code-cell} python

```

6. Create a vector of size `T` containing the emperical mean of each trial. You can use `np.mean` with the argument `axis=1`.


```{code-cell} python

```

7. Plot the histogram of $\sqrt{n}(\mu_n-\mu)$.


```{code-cell} python

```

8. What do you observe when you change `n` ? What is the Fischer information of the age of the population ?


```{code-cell} python

```

# MOM and MLE applied to Gamma Distribution

In this section, we will explore various methods for performing point estimation using simulated data.
Specifically, we will implement two widely-used techniques, the Method of Moments and Maximum Likelihood Estimation (MLE), to estimate the parameters of a Gamma Distribution.
We will implement these methods using the versatile numpy library and compare our results with the implementations available in Scipy.


The probability density function of Gamma distribution with shape parameter $k$ and scale parameter $\theta$ is:
$$f:x\in\mathbb{R}_+ \mapsto \frac{1}{\Gamma(k)\theta^k} x^{k-1} e^{-\frac{x}{\theta}},$$
where the $\Gamma$ function is defined as follow:
$$\Gamma:z\in\mathbb{R}_+ \mapsto \int_0^{+\infty} t^{z-1} e^{-t}.$$

## Data Simulation

1. Simulate `n=1000` observations following a gamma distribution with parameter $\theta=0.2$ and $k=3$. You can use `scipy.stats.gamma.rvs` with argument `size=n`, `a=k` and `scale=theta` to draw `n` samples from a gamma distribution with paramters `k`, `theta`.


```{code-cell} python

```

2. Plot the histogram of these samples along with the probability density function of the gamma law (you can compute the pdf of a gamma function `scipy.stats.gamma.pdf`. You can set the argument `x` to a range from 0 to 2 with steps 0.001, using the `numpy` function `np.arange`).


```{code-cell} python

```

## Method of Moments

3. Show that if $X$ follows a Gamma distribution of parameters $(k, \theta)$ the two first orders moments verify:

4. Show that the parameters $(k, \theta)$ can be expressed thanks to the two first order momentums:

5. Estimate `k_MOM` and `theta_MOM`.


```{code-cell} python

```

## Maximum Likelihood Estimation (numpy)

6. Write the log-likelihood function for the samples $x=(x_1, \dots, x_n)$.

7. What $\theta$ minimizes the log-likelihood ?

8. Plot the log-likelihood with respect to $k$. You can use `scipy.special.gamma` to get the gamma function, and you can create a vector `k_samples` containing values from 2.9 to 3.5 with step 0.00001 (use `np.arange`).


```{code-cell} python

```


```{code-cell} python

```

9. Estimate `k_MLE` and `theta_MLE`. (you cane use `np.argmax` to get the index of the maximum of the log-likelihood)


```{code-cell} python

```

## Maximum Likelihood Estimation (scipy)

10. Use the `scipy.stats.gamma.fit` function to estimate the parameters of the gamma distribution.


```{code-cell} python

```

11. Plot the pdfs of all the estimations (MOM, MLE numpy, MLE scipy) along with the pdf of the real distribution.


```{code-cell} python

```

12. Print the entropy between the estimated distributions and the real distribution. (You can use `scipy.stats.entropy(p,q)` which gives $\sum_{i=1}^n p_i \log(\frac{p_i}{q_i})$ where $p=(p_1, \dots, p_n)$ [resp. $q=(q_1, \dots, q_n)$] are the values of estimated [resp. real] pdf evaluated on $n$ points).


```{code-cell} python

```

# Real Data Application

In this section, we will extend our exploration to real data analysis.
We'll compare several fitted distributions using two essential metrics: entropy and likelihood.
This hands-on experience will provide valuable insights into the practical application of point estimation techniques and the assessment of their accuracy on real-world datasets.

## Loading data
1. Load into a `pandas` data frame named `df_monthly`, the `minimal_temperature_GB.txt` table of the 30-year average monthly minimum temperature at 84291 locations in Great Britain located in the `data/` folder. You can use `pd.read_csv`. (for information, the data come from: https://www.met.ie/climate/30-year-averages)


```{code-cell} python

```

2. Draw the histogram (with 10 bins) for each month and extract the pdf for each bin. To extract the pdf per bin, you'll notice that `plt.hist` returns 3 results:
* values (size 10)
* bin edges (size 11). `(edges[1:]+edges[:-1])/2` gives bin centers.
* The plot


```{code-cell} python

```

## Parameter Estimation
3. Adjust the optimal parameters of the **normal** distribution using MLE on each month's data. You need to record the following scores:
* store in a list `entropy_norm`,
the entropy between the estimated pdf and the actual pdf over the 10 bins (the actual pdf is obtained as in the previous question).
* store in a list `lll_norm`, the log-likelihood of the estimated pdf on the observed data. You can use `np.log` and `np.mean` successively on the pdf calculated on the real data.

You can also plot the estimated pdf with histograms by month.


```{code-cell} python

```

4. Do the same with the **cauchy** distribution. Be sure to use the suffix `_cauchy` (not `_norm`) to store the scores.


```{code-cell} python

```

5. Plot the entropy per month for the two estimated distributions.


```{code-cell} python

```

6. Plot the log-likelihood per month for the two estimated distributions.


```{code-cell} python

```
