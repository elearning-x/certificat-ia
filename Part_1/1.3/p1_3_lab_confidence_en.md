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
  title: 'Solution to Confidence Intervals'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa BEDIN<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

+++

Confidence intervals are powerful tools that provide estimates of population parameters along with measures of uncertainty. In this Python tutorial, we explore different methods to construct confidence intervals for binomial proportions, bimodal distributions, and scenarios with unknown distributions using the bootstrap method.

Part I focuses on confidence intervals for binomial proportions. We begin by examining two popular techniques, Wald's and Wilson's confidence intervals, which are commonly used when dealing with binary outcomes.

Moving on to Part II, we delve into asymptotic confidence intervals. 

Finally, in Part III, we venture into confidence intervals for scenarios where the underlying distribution is unknown. Here, we introduce the bootstrap method, a resampling technique that relies on empirical data to estimate confidence intervals. Through random sampling with replacement, we create a distribution of sample statistics to obtain robust estimates, regardless of the population's true distribution.

We will assess the coverage probability of these intervals, which measures their accuracy in capturing the true parameter. Additionally, we explore the confidence interval length as a measure of precision, helping us choose the most suitable method for a given situation.

# Python Modules


```{code-cell} python
from tqdm import tqdm
import numpy as np
from scipy import stats
from statsmodels.stats import proportion
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

# Confidence Intervals for Binomial Proportion

## Walds and Wilson Confidence Interval

1. Simulate a sample named `sample_data` of 100 i.i.d. variables following a Bernouilli distribution with success probability `p=0.3` (hint: use `np.random.binomial` with argument `n=1`).


```{code-cell} python

```

We recall that for $(Y_1, \dots, Y_n)$ sampled from a Bernouilli distribution we estimate the success probability with $\hat{\theta}_n=\frac{1}{n}\sum_{i=1}^n Y_i$ (which is also the empirical average of the samples).

The Walds CI expression at precision $\alpha$:
$$I_{n, \alpha} =
\hat{\theta}_n
\pm
z_{\alpha}\sqrt{
\frac{\hat{\theta}_n(1-\hat{\theta}_n)}{n}
}
,$$
where $z_{\alpha}$ verifies for $Z\sim \mathcal{N}(0, 1)$: $\mathbb{P}(Z\in[-z_{\alpha}, z_{\alpha}])=1-\frac{\alpha}{2}$.

2. Estimate the probability of success from `sample_data` (hint: you can use `np.mean` on sample_data) and calculate $z_{\alpha}$ for $\alpha=0.05$ (hint: you can use `scipy.stats.norm.ppf` to calculate the inverse of the normal cumulative distribution function).


```{code-cell} python

```

3. Calculate the Walds confidence interval at the 0.95 confidence level. 


```{code-cell} python

```

4. Use `statsmodels.stats.proportion.proportion_confint` to calculate Walds' CI at a confidence level of 0.95.


```{code-cell} python

```

We recall the Wilson CI expression at precision $\alpha$:
$$I_{n, \alpha} = \frac{
\hat{\theta}_n+\frac{z^2_{\alpha}}{2n}
\pm
z_{\alpha}\sqrt{
\frac{\hat{\theta}_n(1-\hat{\theta}_n)}{n}
+\frac{z_{\alpha}^2}{4n^2}
}
}{
1+\frac{z_{\alpha}^2}{n}
}.$$

5. Calculate the Wilson confidence interval at the 0.95 confidence level.


```{code-cell} python

```

6. Use `statsmodels.stats.proportion.proportion_confint` to calculate Wilson' CI at a confidence level of 0.95.


```{code-cell} python

```

## Coverage Probability

In what follows, we will study the influence of sample size on the accuracy of different types of confidence intervals. To carry out the simulations efficiently, we must first ensure that we are able to calculate confidence intervals for T samples of size n in a vectorized way.

7.  Fortunately, `proportion_confint` can take an array of size T as an argument and calculate the corresponding T confidence intervals. Using np.random.binomial, simulate an array of samples of size `Txn` following a Bernouilli distribution and calculate the number of successes per simulation (hint `np.sum` with `axis=1`). Then use proportion_confint to calculate T walds confidence intervals. Take T=10000 and n=100


```{code-cell} python

```

8. Calculate the probability that the value of parameter p lies within the confidence interval (named the coverage probability). (hint: `(CImin<=p)*(CImax>=p)` returns an array containing for each index i: 1 if p is between CImin[i] and CImax[i], 0 otherwise).


```{code-cell} python

```

9. Write a `get_Binomial_coverage` function which takes as argument
* a probability of success p
* a number of simulations T
* a number of data n
* a confidence interval conf_level
* a method for calculating the interval m

and returns the probability that p is within the confidence interval calculated with method m for a sample of size n with confidence level conv_level.


```{code-cell} python

```

10. Calculate a list containing the probability of coverage for n in the interval 2, 100 (hint: you can use the comprehension list for more efficient calculations). Make two lists, one for the Walds confidence interval and the other for the Wilson confidence interval.


```{code-cell} python

```

11. Plot the probability as a function of sample size for both CI methods.


```{code-cell} python

```

## Confidence Interval Length

12. Carry out questions 9 to 11, except that this time we're interested in the length of the confidence interval. We'll write a function `get_Binomial_length` that takes as its argument:
* a probability of success p
* a number of simulations T
* a number of data n
* a confidence interval conf_level
* a method for calculating the interval m

and returns the average length of confidence intervals estimated with method m at conf_level on T samples of size n.


```{code-cell} python

```


```{code-cell} python

```


```{code-cell} python

```

13. Vary p and comment.

# Asymptotic Confidence Intervals

We take as example the same bimodal distribution for the age of a population we used in last TP, with each mode following a Poisson distribution centered on 40 years and 10 years respectively. There are twice as many individuals in the first mode as in the second.

1. Model the distribution for 3000000 indivudals and plot the histogram.


```{code-cell} python

```

2. Draw a sample `sample_ages` of size 100 and calculate an estimate of the mean from this sample.


```{code-cell} python

```

## CI with Normal Approximation

The standard deviation of the sample converges in probability to the standard deviation of the population. We'll construct a confidence interval similar to Walds', except that instead of expressing the standard deviation as a function of the estimated mean, we'll use the standard deviation directly estimated on the sample. (indeed, we do not know a closed form for expressing an estimation of the std as a fonction of the estimated mean in this case...)

3. Calculate the confidence interval using the normal approximation.


```{code-cell} python

```

4. As in the first part, we want to study the effect of sample size on coverage probability. Write a function `get_NormalCI_coverage` which takes as argument:
* the population population_ages
* the number of simulations to run T
* sample size n
* confidence level conf_level

and returns the coverage probability for size n of the normal confidence interval. (Hint: np.random.choice can be used with the argument size=(T, n) to draw T samples of size n. np.mean and np.std should be used with the argument axis=1).


```{code-cell} python

```

5. Write the same style of function, but this time return the average length of the CI for T simulations of samples of size n.


```{code-cell} python

```

## CI with Student Approximation

Let $\hat{\mu}_n$, $\hat{\sigma}_n$ be the mean and standard deviation of the sample of size n. Let $\mu$ be the population mean. We want to know the distribution of $$U\sim \sqrt{n}\frac{\hat{\mu}_n-\mu}{\hat{\sigma}_n}.$$

6. To do this, draw T samples of size n from population_ages (np.random.choice) and calculate T values of the random variable $U$ (Hint use np.mean, np.std with axis=1).


```{code-cell} python

```

7. For each value of U, calculate the pdf of a Student distribution with n-1 degrees of freedom (hint: use scipy.stats.t.pdf). Print the log-likelihood of the Student distribution on the observed values of U (Hint: np.log followed by np.mean on the student_pdf table).


```{code-cell} python

```

8. Plot the histogram and the pdf.


```{code-cell} python

```

9. Carry out questions 4 and 5 using the student's approximation.


```{code-cell} python

```


```{code-cell} python

```

10. Calculate a list containing the probability of coverage for n in the interval 2, 100 (hint: you can use the comprehension list for more efficient calculations). Make two lists, one for the Normal approximation and the other for Student approximation.


```{code-cell} python

```

11. Plot the probability as a function of sample size for both CI methods.


```{code-cell} python

```

12. Carry out questions 10 and 11 for CI interval length.


```{code-cell} python

```


```{code-cell} python

```

# CI for Unknown Distributions: Boostrap Method

Now we're interested in the median of the population, but we don't know a simple distribution to approximate the distribution.

1. To see for yourself, plot for T=10000 samples of size n=20 the histogram of the estimated median minus the population median. (hint: np.random.choice with size=(T,n) and np.median with axis=1).


```{code-cell} python

```

Bootstrap is a statistical resampling method used to estimate the sampling distribution of a statistic and to construct confidence intervals. It is particularly useful when the underlying distribution of the data is unknown or difficult to model. To calculate a confidence interval for the median of a population using the bootstrap method, we follow these steps:

* Step 1. Sample with Replacement: We start by taking a random sample (with replacement) from the observed data. The size of this bootstrap sample is typically the same as the size of the observed data.
* Step 2. Calculate the Sample Median: Next, we calculate the median of the bootstrap sample. This is our bootstrap statistic.
* Step 3. Repeat the Process: We repeat steps 1 and 2 a large number of times (often thousands of times) to create a distribution of bootstrap statistics.
* Step 4. Construct the Confidence Interval: From the distribution of bootstrap statistics, we calculate the percentiles that correspond to our desired confidence level. For example, for a 95% confidence interval, we would take the 2.5th percentile and the 97.5th percentile of the bootstrap statistics.

Final Result: The confidence interval is given by the range between the percentiles obtained in step 4. It provides an estimate of the population median along with a measure of its uncertainty.

2. Perform steps 1 to 3 of the boostrap method with B=10000 boostrap samples. (Hint: you can use np.random.choice with arguments size=(B, n) and replace=True, then np.median with axis=1).


```{code-cell} python

```

3. Perform step 4 and print the confidence interval (Hint: you can use np.quantile).


```{code-cell} python

```

4. Plot proba coverage as a function of sample size, for B=1000 and for a size in the interval (2, 200, 10). (Hint: np.quantile with axis=1).


```{code-cell} python

```


```{code-cell} python

```


```{code-cell} python

```

# Real Data Analysis
The following data is derived from an available dataset in statmodels:

"The data, collected as part of a 1987 intercity mode choice study, are a sub-sample of 210 non-business trips between Sydney, Canberra and Melbourne in which the traveler chooses a mode from four alternatives (plane, car, bus and train). The sample, 840 observations, is choice based with over-sampling of the less popular modes (plane, train and bus) and under-sampling of the more popular mode, car. The level of service data was derived from highway and transport networks in Sydney, Melbourne, non-metropolitan N.S.W. and Victoria, including the Australian Capital Territory."

1. Load into a pandas dataframe named "df" the dataset travel_choice.csv located in the `data/` folder with `pd.read_csv` and display the first 5 rows by writing `df.head()` on the last cell.


```{code-cell} python

```

The aim of questions 2 and 3 is to add a new column called "mode" to the dataframe "df" with chosen mode ("plane", "train", "bus", or "car").

The steps outlined in questions 2 and 3 are not the only way to achieve this, so feel free to skip these questions and adopt your own steps.

2. Write a function called "set_mode" which takes a row of the dataframe as argument and returns the mode chosen for this row. You can test your function on the first dataset row obtained with `df.iloc[0]`.


```{code-cell} python

```

3. Create the new "mode" column by applying "set_mode" to each row of df. (Tip: you can use the `df.apply` method with the correct axis). 


```{code-cell} python

```

4. Plot the confidence intervals of the "mean" for each mode chosen. (Tip: you can use seaborn's `sns.pointplot` with `estimator='mean'` and `errorbar='ci'` and 'mode' on the x-axis and 'hinc' on the y-axis).


```{code-cell} python

```
