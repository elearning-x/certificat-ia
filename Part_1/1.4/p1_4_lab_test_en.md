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
  title: 'Solution to Statistical Tests'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa BEDIN<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

+++

# Python Modules

```{code-cell} python
import numpy as np
from  scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

# Power Function of a Test

## For the mean $\mu$ knowing $\sigma$
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

```


```{code-cell} python

```

2. Test whether the mean of the underlying distribution is 0 or not. Try different values of mu or n for which the test is NOT successful.


```{code-cell} python

```

We recall that the Power function is defined as $\mu\mapsto 1-P_{\mu}(T(Y)\in [-c_{\alpha}, c_{\alpha}])$, it corresponds to the probability that the $H_0$ is rejected as a function of the true mean value $\mu$.

3. For a given $\mu$ and $\sigma$, what is the law of $N = \frac{ \sum_{i=1}^n Y_i-n\mu}
{
\sigma \sqrt{n}
}$ ? Verify visually this observation by plotting the histogram of values of $N$ along with the theoretical density (hint: you can sample an array of size Txn from $\mathcal{N}(\mu, \sigma)$ using `stats.norm.rvs(loc=mu, scale=sigma, size=(T, n))` and then compute an array of size T containing the values of N, using operations such as `np.mean` on a well-chosen `axis`)


```{code-cell} python

```

4. Write the statistic $T(Y)$ as a function of $N$ and deduce a simple way to plot the power function as function of $\mu$ for several $n$ and for $\alpha=0.05$. (Hint: you can use `scipy.stats.norm.cdf`)


```{code-cell} python

```

## For the std $\sigma$ when $\mu$ is known

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

```


```{code-cell} python

```

6. Test whether the std of the underlying distribution is less than 1 or not. Try different values of $\sigma$ or n for which the test is NOT successful.


```{code-cell} python

```

7. What is the law of $W=\frac{1}{\sigma^2}\sum_{i=1}^n (Y_i - \mu)^2$ ? Verify the law of $W$ visually with a plot for a given $\mu$ and $\sigma$.


```{code-cell} python

```

8.  Write the statistic $T(Y)$ as a function of $W$ and deduce a simple way to plot the power function as function of $\sigma$ for several $n$ and for $\alpha=0.05$. (Hint: you can use `scipy.stats.chi2.cdf`)


```{code-cell} python

```

# Real Data Application

## Loading the Data

We want to analyze the traveler's household income as a function of mode choice. To do this, we'll check whether household income is equal on average for travellers choosing different modes of transport.

We'll always assume that the underlying distribution is normal, so that we can apply the methods we've seen in class.

1. Load the dataset 'travel_choice.csv' located in the `data/` folder into a pandas DataFrame (hint: use `pd.read_csv`)


```{code-cell} python

```

## Testing the equality of the variance

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

```

3. For each pair mode0 and mode1, test whether the household incomes of individuals taking each of these modes have the same stds. Tip 1: to go through all possible mode pairs, you can use two "for loops": `for mode1 in ['plane', 'train', 'bus', 'car']` and `for mode2 in ['plane', 'train', 'bus', 'car']`. Tip 2: to obtain the household income of people using mode0, you can use: `Y0 = df[df[mode0]==1.]['hinc']`.


```{code-cell} python

```

## Testing the equality of the mean

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

```

5. Find out which pair of modes has the same average.


```{code-cell} python

```

In the rest of the course we'll look at other methods for studying how explanatory variables (e.g. choice of means of transport) influence an observed variable (e.g. household income). To this end, we will examine different regression methodologies.

6. Add a new column called "mode" to the dataframe "df" with chosen mode ("plane", "train", "bus", or "car").


```{code-cell} python

```

7. Plot the "mean" and "standard" deviation for each mode chosen. (Tip: you can use again `sns.pointplot` with `errorbar='sd`.)


```{code-cell} python

```
