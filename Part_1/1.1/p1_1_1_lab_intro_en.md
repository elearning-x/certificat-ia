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
  title: 'Python Refresher'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa BEDIN<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

+++

The primary goal of this tutorial is to acquaint you with essential Python libraries for random variable analysis and data manipulation.
We will explore the capabilities of the numpy and scipy packages for generating and analyzing random variables, as well as the pandas and seaborn libraries for efficient data analysis.

# Python Modules


```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, special
import pandas as pd
import seaborn as sns
```

# Normal Law

In this section, we will focus on understanding the normal distribution.
We will harness the power of numpy and scipy to generate random samples that follow a normal distribution.
Through practical examples, you will learn how to create histograms to visualize the distribution of your data.
We will also explore various methods to display the probability density function and cumulative distribution function, including explicit formulas, scipy library functions, and approximations derived from histograms.

1. Create a numpy array called "normal_samples" containing N=500 samples extracted from a normal distribution of mean mu=5 and standard deviation sigma=20, using `np.random.normal` from the numpy package.
To ensure that each time you run the cell, you get exactly the same numpy array, write `np.random.seed(0)` in the first row of the cell (you can choose any seed you like).


```{code-cell} python

```

2. Same question but using `stats.norm.rvs` from scipy instead of numpy. Don't forget to define the seed as in the previous question.


```{code-cell} python

```

3. Check that normal_samples and normal_samples_scipy are equal (tip: you can add the difference between the two arrays and check that it's close to zero, or you can use `np.allclose`).


```{code-cell} python

```

4. Plot the histogram of normal samples using matplotlib's `plt.hist` (tip: you can set the arguments bins=10 and density=True).


```{code-cell} python

```

5. Calculate numerically using scipy's function ̀`stats.norm.ppf` (which gives the inverse of the cumulative distribution function) the values $x_1$ and $x_{99}$ such that if $X$ follows a normal distribution of mean mu and variance sigma, then $P(X\leq x_1) = 0.01$ and $P(X\leq x_{99}) = 0.99$.


```{code-cell} python

```

6. Recall the expression for the density $f$ of a normal distribution $\mathcal{N}(\mu, \sigma^2)$. Create a numpy array "pdf_express" containing the density values of $f(x)$ for $x\in[x_1, x_{99}]$ (hint: You can start by defining a numpy array "x_samples" of evenly spaced values between $x_1$ and $x_{99}$ with `np.arange` or `np.linspace`. Then you can use the usual numpy array operations such as `np.exp`)

$$f(x) = \frac{1}{\sqrt{2 \pi \sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$


```{code-cell} python

```

7. Create a numpy array "pdf_scipy" containing the density values of $f(x)$ for $x\in[x_1, x_{99}]$  using `stats.norm.pdf`. Verify that "pdf_express" and "pdf_scipy" are equal. (Hint: you can use the same "x_samples" created in the previous question)


```{code-cell} python

```

In the following tutorials, we'll often use `stats.[law_name].pdf` to calculate the density of a law (very useful when we don't know or remember its explicit form).

8. In this question, we'd like to extract the pdf directly from the histogram. We first notice that `plt.hist` returns 3 results:
* values (size 10). In fact, this array is an estimate (based on "normal_samples" observations) of the density value at the center of the bin.
* bin edges (size 11). `(edges[1:]+edges[:-1])/2` gives the histogram centers.
* Plotting the histogram

Question: Create two tables: "bin_centers" and "pdf_estimated" containing the bin center values and the bin center density function values respectively. Plot the histogram along with the pdfs calculated with the closed form, scipy and with estimation.

NB: In the following tutorials, we'll use this density estimation technique for samples whose distribution is unknown. This will be particularly useful for tutorial 2.


```{code-cell} python

```

We recall the closed form of the cumulative distribution function of a normal distribution:
$$\mathbb{P}(X\leq x) = \int_{-\infty}^x f(y) dy = \frac{1}{2} [ 1+ \text{erf}(\frac{x-\mu}{\sigma \sqrt{2}})]$$
where "erf" is the "error function" and $\text{erf}(x) = \frac{2}{ \sqrt{\pi} } \int_0^x e^{-t^2} dt$ and can be computed with `scipy.special.erf`.

9. Plot the cumulative distribution obtained with:
* the closed form (use the form recalled above)
* the function `stats.norm.cdf` from scipy
* the bin_centers and cdf_estimated computed in the previous question. (hint: you can use `np.cumsum` to obtain the cumulative sum on the cdf_estimated and then multiply by the number of bins).


```{code-cell} python

```

# Real Data Analysis

In this section we will introduce a real-world dataset that will be the subject of upcoming lab sessions.
Here, we will uncover the versatility of the pandas library, which allows us to efficiently manipulate and explore complex data structures.

The dataset we will be working with originates from https://www.met.ie/climate/30-year-averages and provides valuable insights into climate patterns.

1. Load into a pandas dataframe named "df" the dataset minimal_temperature_GB.txt located in the `data/` folder with `pd.read_csv` and display the first 5 rows by writing `df.head()` on the last cell.


```{code-cell} python

```

2. Print the length L of the dataframe and the columns of the dataframe.


```{code-cell} python

```

The aim of questions 3 to 6 is to transform the dataframe into a dataframe of length 12L with 2 columns: the month number "month", the minimum temperature for that month "Tmin".

The steps described in questions 3 to 6 are not the only way to achieve this, so you can skip these questions and adopt your own steps.

3. Write a "get_month_name" function that takes as input an integer month_id between 1 and 13 and returns the name of the corresponding column. (Hint: you can use python "f-strings" or the python method `.format()`)


```{code-cell} python

```

4. Write a "get_month_df" function that takes month_id as input and returns a pandas DataFrame of length L, but with only two columns: one containing the month_id and the other containing the minimum temperature. (Tip: you can start by creating two lists or arrays of equal length L: one containing the minimum temperature for the month in question and the other containing only the "month_id". Then create a pandas dataframe using `pd.DataFrame` using these lists)


```{code-cell} python

```

5. Create a pandas dataframe of length 12L with only two columns: one containing the month_id and the other containing the minimum temperature. (Hint: you can create an "all_months" list of 12 dataframes by calling the function get_month_df(month_id) for month_id ranging from 1 to 13. You can then use pd.concat(all_months) to create the expected dataframe).


```{code-cell} python

```

6. Use the `sns.boxplot` function from the `seaborn` package to display quantiles of minimum temperature as a function of month.


```{code-cell} python

```
