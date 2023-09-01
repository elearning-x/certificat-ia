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
<span>Lisa Bedin<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

+++

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

Confidence intervals are powerful tools that provide estimates of population parameters along with measures of uncertainty. In this Python tutorial, we explore different methods to construct confidence intervals for binomial proportions, bimodal distributions, and scenarios with unknown distributions using the bootstrap method.

Part I focuses on confidence intervals for binomial proportions. We begin by examining two popular techniques, Wald's and Wilson's confidence intervals, which are commonly used when dealing with binary outcomes.

Moving on to Part II, we delve into asymptotic confidence intervals. 

Finally, in Part III, we venture into confidence intervals for scenarios where the underlying distribution is unknown. Here, we introduce the bootstrap method, a resampling technique that relies on empirical data to estimate confidence intervals. Through random sampling with replacement, we create a distribution of sample statistics to obtain robust estimates, regardless of the population's true distribution.

We will assess the coverage probability of these intervals, which measures their accuracy in capturing the true parameter. Additionally, we explore the confidence interval length as a measure of precision, helping us choose the most suitable method for a given situation.

# Confidence Intervals for Binomial Proportion

## Walds and Wilson Confidence Interval

1. Simulate a sample named `sample_data` of 100 i.i.d. variables following a Bernouilli distribution with success probability `p=0.3` (hint: use `np.random.binomial` with argument `n=1`).


```{code-cell} python
np.random.seed(0)
p = 0.3
n = 100

sample_data = np.random.binomial(n=1, p=p, size=n)
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
p_n = np.mean(sample_data)

conf_level = 0.95
z_score = stats.norm.ppf(1 - (1 - conf_level) / 2)
```

3. Calculate the Walds confidence interval at the 0.95 confidence level. 


```{code-cell} python
std_n = np.sqrt(p_n * (1 - p_n))

CImin = p_n - z_score * std_n / np.sqrt(n)
CImax = p_n + z_score * std_n / np.sqrt(n)

print(f'estimated success proba: {p_n:.2f}')
print(f'Walds CI at 0.95 confidence level (numpy): [{CImin:.4f}, {CImax:.4f}]')
```

4. Use `statsmodels.stats.proportion.proportion_confint` to calculate Walds' CI at a confidence level of 0.95.


```{code-cell} python
CImin, Cmax = proportion.proportion_confint(sample_data.sum(), n, alpha=0.05, method='normal')
print(f'Walds CI at 0.95 confidence level (statsmodels): [{CImin:.4f}, {CImax:.4f}]')
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
first_order =  z_score**2 / (2 * n)
second_order = z_score * np.sqrt(p_n*(1-p_n)/n + z_score**2 / (4*n**2))
denominator = 1 + z_score**2 / n

CImin = (p_n + first_order - second_order) / denominator
CImax = (p_n + first_order + second_order) / denominator

print(f'Wilson CI at 0.95 confidence level (numpy): [{CImin:.4f}, {CImax:.4f}]')
```

6. Use `statsmodels.stats.proportion.proportion_confint` to calculate Wilson' CI at a confidence level of 0.95.


```{code-cell} python
CImin, Cmax = proportion.proportion_confint(sample_data.sum(), n, alpha=0.05, method='wilson')
print(f'Wilson CI at 0.95 confidence level (statsmodels): [{CImin:.4f}, {CImax:.4f}]')
```

## Coverage Probability

In what follows, we will study the influence of sample size on the accuracy of different types of confidence intervals. To carry out the simulations efficiently, we must first ensure that we are able to calculate confidence intervals for T samples of size n in a vectorized way.

7.  Fortunately, `proportion_confint` can take an array of size T as an argument and calculate the corresponding T confidence intervals. Using np.random.binomial, simulate an array of samples of size `Txn` following a Bernouilli distribution and calculate the number of successes per simulation (hint `np.sum` with `axis=1`). Then use proportion_confint to calculate T walds confidence intervals. Take T=10000 and n=100


```{code-cell} python
np.random.seed(0)
T = 10000
n = 100
sample_data = np.random.binomial(n=1, p=p, size=(T, n))
sum_data = np.sum(sample_data, axis=1)
    
CImin, CImax = proportion.proportion_confint(sum_data, n, alpha=1-conf_level, method='normal')
```

8. Calculate the probability that the value of parameter p lies within the confidence interval (named the coverage probability). (hint: `(CImin<=p)*(CImax>=p)` returns an array containing for each index i: 1 if p is between CImin[i] and CImax[i], 0 otherwise).


```{code-cell} python
proba_covered = np.mean((CImin<=p)*(CImax>=p))
print(f'the coverage probability is: {proba_covered:.4f}')
```

9. Write a `get_Binomial_coverage` function which takes as argument
* a probability of success p
* a number of simulations T
* a number of data n
* a confidence interval conf_level
* a method for calculating the interval m

and returns the probability that p is within the confidence interval calculated with method m for a sample of size n with confidence level conv_level.


```{code-cell} python
def get_Binomial_coverage(p, T, n, conf_level=0.95, m='wilson'):
    sample_data = np.random.binomial(n=1, p=p, size=(T, n))
    sum_data = np.sum(sample_data, axis=1)
    
    CImin, CImax = proportion.proportion_confint(sum_data, n, alpha=1-conf_level, method=m)
    
    proba_covered = ((CImin<=p)*(CImax>=p)).sum() / T
    return proba_covered
```

10. Calculate a list containing the probability of coverage for n in the interval 2, 100 (hint: you can use the comprehension list for more efficient calculations). Make two lists, one for the Walds confidence interval and the other for the Wilson confidence interval.


```{code-cell} python
np.random.seed(0)
T = 10000
sample_sizes = np.arange(2, 100)
p = 0.5

NORMAL_proba_covered = [get_Binomial_coverage(p, T, n, m='normal') for n in sample_sizes]
WILSON_proba_covered = [get_Binomial_coverage(p, T, n, m='wilson') for n in sample_sizes]
```

11. Plot the probability as a function of sample size for both CI methods.


```{code-cell} python
plt.plot(sample_sizes, WILSON_proba_covered, label='wilson')
plt.plot(sample_sizes, NORMAL_proba_covered, label='normal')
plt.xlabel('sample size')
plt.ylabel('proba coverage')
plt.legend()
plt.show()
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
def get_Binomial_length(p, T, n, conf_level=0.95, m='wilson'):
    sample_data = np.random.binomial(n=1, p=p, size=(T, n))
    sum_data = np.sum(sample_data, axis=1)
    
    CImin, CImax = proportion.proportion_confint(sum_data, n, alpha=1-conf_level, method=m)
    
    interval_length = np.abs(CImin-CImax).mean()
    return interval_length
```


```{code-cell} python
np.random.seed(0)
T = 10000
sample_sizes = np.arange(2, 100)
p = 0.5

NORMAL_length = [get_Binomial_length(p, T, n, m='normal') for n in sample_sizes]
WILSON_length = [get_Binomial_length(p, T, n, m='wilson') for n in sample_sizes]
```


```{code-cell} python
plt.plot(sample_sizes, NORMAL_length, label='wilson')
plt.plot(sample_sizes, WILSON_length, label='normal')
plt.xlabel('sample size')
plt.ylabel('average CI length')
plt.legend()
plt.show()
```

13. Vary p and comment.

# Asymptotic Confidence Intervals

We take as example the same bimodal distribution for the age of a population we used in last TP, with each mode following a Poisson distribution centered on 40 years and 10 years respectively. There are twice as many individuals in the first mode as in the second.

1. Model the distribution for 3000000 indivudals and plot the histogram.


```{code-cell} python
np.random.seed(0)
population_ages1 = stats.poisson.rvs(loc=0, mu=30, size=2000000)
population_ages2 = stats.poisson.rvs(loc=0, mu=10, size=1000000)
population_ages = np.concatenate((population_ages1, population_ages2))

plt.hist(population_ages, bins=30, density=True, edgecolor="k")
plt.ylabel('density')
plt.xlabel('age')
plt.show()
```

2. Draw a sample `sample_ages` of size 100 and calculate an estimate of the mean from this sample.


```{code-cell} python
np.random.seed(0)
n = 100
sample_ages = np.random.choice(a=population_ages, size=n)

mu_estimated = sample_ages.mean()
mu_star = population_ages.mean()

print(f"estimated mean: {mu_estimated:.2f}")
print(f"real mean: {mu_star:.2f}")
```

## CI with Normal Approximation

The standard deviation of the sample converges in probability to the standard deviation of the population. We'll construct a confidence interval similar to Walds', except that instead of expressing the standard deviation as a function of the estimated mean, we'll use the standard deviation directly estimated on the sample. (indeed, we do not know a closed form for expressing an estimation of the std as a fonction of the estimated mean in this case...)

3. Calculate the confidence interval using the normal approximation.


```{code-cell} python
std_estimated = sample_ages.std()

conf_level = 0.95

z_score = stats.norm.ppf(1 - (1 - conf_level) / 2)

CImin = mu_estimated - z_score*std_estimated/np.sqrt(sample_ages.shape[0])
CImax = mu_estimated + z_score*std_estimated/np.sqrt(sample_ages.shape[0])

print(f"CI: [{CImin:.2f}, {CImax:.2f}]")
```

4. As in the first part, we want to study the effect of sample size on coverage probability. Write a function `get_NormalCI_coverage` which takes as argument:
* the population population_ages
* the number of simulations to run T
* sample size n
* confidence level conf_level

and returns the coverage probability for size n of the normal confidence interval. (Hint: np.random.choice can be used with the argument size=(T, n) to draw T samples of size n. np.mean and np.std should be used with the argument axis=1).


```{code-cell} python
def get_NormalCI_coverage(population_ages, T, n, conf_level=0.95):
    mean_star = np.mean(population_ages)
    trials_samples = np.random.choice(a=population_ages, size=(T, n))
    mean_n = np.mean(trials_samples, axis=1)
    std_n = np.std(trials_samples, axis=1)
    
    z_score = stats.norm.ppf(conf_level + (1 - conf_level) / 2)
    
    CImin = mean_n - z_score*std_n/np.sqrt(n)
    CImax = mean_n + z_score*std_n/np.sqrt(n)
    
    proba_covered = ((CImin<=mean_star)*(CImax>=mean_star)).mean()
    return proba_covered
```

5. Write the same style of function, but this time return the average length of the CI for T simulations of samples of size n.


```{code-cell} python
def get_NormalCI_length(population_ages, T, n, conf_level=0.95):
    mean_star = np.mean(population_ages)
    trials_samples = np.random.choice(a=population_ages, size=(T, n))
    mean_n = np.mean(trials_samples, axis=1)
    std_n = np.std(trials_samples, axis=1)
    
    z_score = stats.norm.ppf(conf_level + (1 - conf_level) / 2)
    
    CImin = mean_n - z_score*std_n/np.sqrt(n)
    CImax = mean_n + z_score*std_n/np.sqrt(n)
    
    interval_length = np.abs(CImin-CImax).mean()
    return interval_length
```

## CI with Student Approximation

Let $\hat{\mu}_n$, $\hat{\sigma}_n$ be the mean and standard deviation of the sample of size n. Let $\mu$ be the population mean. We want to know the distribution of $$U\sim \sqrt{n}\frac{\hat{\mu}_n-\mu}{\hat{\sigma}_n}.$$

6. To do this, draw T samples of size n from population_ages (np.random.choice) and calculate T values of the random variable $U$ (Hint use np.mean, np.std with axis=1).


```{code-cell} python
np.random.seed(0)
T = 10000 # number of trials
n = 1000 # population size

trials_samples = np.random.choice(a=population_ages, size=(T, n))

mu_n = np.mean(trials_samples, axis=1)
std_n = np.std(trials_samples, axis=1)
mu_star = np.mean(population_ages)

mu_star = population_ages.mean()

U = np.sqrt(n)*(mu_n-mu_star)/std_n
```

7. For each value of U, calculate the pdf of a Student distribution with n-1 degrees of freedom (hint: use scipy.stats.t.pdf). Print the log-likelihood of the Student distribution on the observed values of U (Hint: np.log followed by np.mean on the student_pdf table).


```{code-cell} python
student_pdf = stats.t.pdf(x=U, df=n-1)

log_likelihood = np.mean(np.log(student_pdf))
print(f'The log-likelihood is: {log_likelihood:.2f}')
```

8. Plot the histogram and the pdf.


```{code-cell} python
plt.title(f'log-likelihood: {log_likelihood:.2f}')
plt.hist(U, bins=100, density=True, edgecolor='k')
plt.plot(U, student_pdf, label='pdf', linestyle='', marker='.', markersize=1)
plt.xlabel('normalized empirical mean')
plt.ylabel('density')
plt.show()
```

9. Carry out questions 4 and 5 using the student's approximation.


```{code-cell} python
def get_StudentCI_coverage(population_ages, T, n, conf_level=0.95):
    mean_star = np.mean(population_ages)
    trials_samples = np.random.choice(a=population_ages, size=(T, n))
    mean_n = np.mean(trials_samples, axis=1)
    std_n = np.std(trials_samples, axis=1)
    
    degrees_of_freedom = n-1
    t_score = stats.t.ppf(conf_level + (1 - conf_level) / 2, df=degrees_of_freedom)
    
    CImin = mean_n - t_score*std_n/np.sqrt(n)
    CImax = mean_n + t_score*std_n/np.sqrt(n)
    
    proba_covered = ((CImin<=mean_star)*(CImax>=mean_star)).sum() / T
    return proba_covered
```


```{code-cell} python
def get_StudentCI_length(population_ages, T, n, conf_level=0.95):
    mean_star = np.mean(population_ages)
    trials_samples = np.random.choice(a=population_ages, size=(T, n))
    mean_n = np.mean(trials_samples, axis=1)
    std_n = np.std(trials_samples, axis=1)
    
    degrees_of_freedom = n-1
    t_score = stats.t.ppf(conf_level + (1 - conf_level) / 2, df=degrees_of_freedom)
    
    CImin = mean_n - t_score*std_n/np.sqrt(n)
    CImax = mean_n + t_score*std_n/np.sqrt(n)
    
    interval_length = np.abs(CImin-CImax).mean()
    return interval_length
```

10. Calculate a list containing the probability of coverage for n in the interval 2, 100 (hint: you can use the comprehension list for more efficient calculations). Make two lists, one for the Normal approximation and the other for Student approximation.


```{code-cell} python
np.random.seed(0)
T = 10000 # number of trials
conf_level = 0.95
population_sizes = np.arange(2, 100, 1)# population size

NORMAL_proba_covered = [get_NormalCI_coverage(population_ages, T, n, conf_level) for n in population_sizes]
STUDENT_proba_covered = [get_StudentCI_coverage(population_ages, T, n, conf_level) for n in population_sizes]
```

11. Plot the probability as a function of sample size for both CI methods.


```{code-cell} python
plt.plot(population_sizes, NORMAL_proba_covered, color='b', label='normal approx')
plt.plot(population_sizes, STUDENT_proba_covered, color='k', label='student approx')
plt.plot(population_sizes, [conf_level]*len(population_sizes), color='darkred', linestyle='--', label='conf level')
plt.legend()
plt.xlabel('population size')
plt.ylabel('real mean covered proba')
plt.title('coverage plot')
plt.show()
```

12. Carry out questions 10 and 11 for CI interval length.


```{code-cell} python
np.random.seed(0)
T = 10000 # number of trials
conf_level = 0.95
population_sizes = np.arange(20, 100, 1)# population size

NORMAL_length = [get_NormalCI_length(population_ages, T, n, conf_level) for n in population_sizes]
STUDENT_length = [get_StudentCI_length(population_ages, T, n, conf_level) for n in population_sizes]
```


```{code-cell} python
plt.plot(population_sizes, NORMAL_length, color='b', label='normal approx')
plt.plot(population_sizes, STUDENT_length, color='k', label='student approx')
plt.legend()
plt.xlabel('population size')
plt.ylabel('CI length')
plt.title('coverage plot')
plt.show()
```

# CI for Unknown Distributions: Boostrap Method

Now we're interested in the median of the population, but we don't know a simple distribution to approximate the distribution.

1. To see for yourself, plot for T=10000 samples of size n=20 the histogram of the estimated median minus the population median. (hint: np.random.choice with size=(T,n) and np.median with axis=1).


```{code-cell} python
np.random.seed(0)
T = 100000 # number of trials
n = 20 # population size

trials_samples = np.random.choice(a=population_ages, size=(T, n))

median_n = np.median(trials_samples, axis=1)
median_star = np.median(population_ages)

plt.hist((median_n-median_star), bins=100, density=True, edgecolor='k')
plt.xlabel('normalized empirical mean')
plt.ylabel('density')
plt.show()
```

Bootstrap is a statistical resampling method used to estimate the sampling distribution of a statistic and to construct confidence intervals. It is particularly useful when the underlying distribution of the data is unknown or difficult to model. To calculate a confidence interval for the median of a population using the bootstrap method, we follow these steps:

* Step 1. Sample with Replacement: We start by taking a random sample (with replacement) from the observed data. The size of this bootstrap sample is typically the same as the size of the observed data.
* Step 2. Calculate the Sample Median: Next, we calculate the median of the bootstrap sample. This is our bootstrap statistic.
* Step 3. Repeat the Process: We repeat steps 1 and 2 a large number of times (often thousands of times) to create a distribution of bootstrap statistics.
* Step 4. Construct the Confidence Interval: From the distribution of bootstrap statistics, we calculate the percentiles that correspond to our desired confidence level. For example, for a 95% confidence interval, we would take the 2.5th percentile and the 97.5th percentile of the bootstrap statistics.

Final Result: The confidence interval is given by the range between the percentiles obtained in step 4. It provides an estimate of the population median along with a measure of its uncertainty.

2. Perform steps 1 to 3 of the boostrap method with B=10000 boostrap samples. (Hint: you can use np.random.choice with arguments size=(B, n) and replace=True, then np.median with axis=1).


```{code-cell} python
# Number of bootstrap samples
np.random.seed(0)
B = 10000
n = sample_ages.shape[0]

bootstrap_sample = np.random.choice(sample_ages, size=(B, n), replace=True)
bootstrap_sample_means = np.median(bootstrap_sample, axis=1)
```

3. Perform step 4 and print the confidence interval (Hint: you can use np.quantile).


```{code-cell} python
conf_level = 0.95
lower_quantile = (1 - conf_level) / 2
upper_quantile = 1 - lower_quantile
lower_ci = np.quantile(bootstrap_sample_means, lower_quantile)
upper_ci = np.quantile(bootstrap_sample_means, upper_quantile)

print(f"Bootstrap CI: [{lower_ci:.2f}, {upper_ci:.2f}]")
```

4. Plot proba coverage as a function of sample size, for B=1000 and for a size in the interval (2, 200, 10). (Hint: np.quantile with axis=1).


```{code-cell} python
def get_BootstrapCI_coverage(population_ages, T, B, n, conf_level=0.95):
    mean_star = np.median(population_ages)
    trials_samples = np.random.choice(a=population_ages, size=(T, n))
    
    bootstrap_indices = np.random.randint(0, n, size=(T, B, n))

    boost_samples = trials_samples[np.arange(T)[:, None, None], bootstrap_indices]
    
    mean_n = np.median(boost_samples, axis=2)

    Qmin = (1 - conf_level) / 2
    Qmax = 1 - lower_quantile
    
    CImin = np.quantile(mean_n, Qmin, axis=1)
    CImax = np.quantile(mean_n, Qmax, axis=1)
    #print(CImin.shape, CImax.shape)
    
    proba_covered = ((CImin<=mean_star)*(CImax>=mean_star)).sum() / T
    return proba_covered
```


```{code-cell} python
np.random.seed(0)
T = 100 # number of trials
conf_level = 0.95
B = 1000 # number of boostraps
population_sizes = np.arange(2, 200, 10)# population size

BOOTSTRAP_proba_covered = [get_BootstrapCI_coverage(population_ages, T, B, n, conf_level)
                           for n in tqdm(population_sizes, total=len(population_sizes))]

```


```{code-cell} python
plt.plot(population_sizes, BOOTSTRAP_proba_covered, color='g', label='boostrap method')
plt.plot(population_sizes, [conf_level]*len(population_sizes), color='darkred', linestyle='--', label='conf level')
plt.legend()
plt.xlabel('population size')
plt.ylabel('real mean covered proba')
plt.title('coverage plot')
plt.show()
```

# Real Data Analysis
The following data is derived from an available dataset in statmodels:

"The data, collected as part of a 1987 intercity mode choice study, are a sub-sample of 210 non-business trips between Sydney, Canberra and Melbourne in which the traveler chooses a mode from four alternatives (plane, car, bus and train). The sample, 840 observations, is choice based with over-sampling of the less popular modes (plane, train and bus) and under-sampling of the more popular mode, car. The level of service data was derived from highway and transport networks in Sydney, Melbourne, non-metropolitan N.S.W. and Victoria, including the Australian Capital Territory."

1. Load into a pandas dataframe named "df" the dataset travel_choice.csv located in the `data/` folder with `pd.read_csv` and display the first 5 rows by writing `df.head()` on the last cell.


```{code-cell} python
df = pd.read_csv('data/travel_choice.csv')
df.head()
```

The aim of this part is to add a new column called "mode" to the dataframe "df" with chosen mode ("plane", "train", "bus", or "car").

The steps outlined in questions 3 and 4 are not the only way to achieve this, so feel free to skip these questions and adopt your own steps.

2. Write a function called "set_mode" which takes a row of the dataframe as argument and returns the mode chosen for this row. You can test your function on the first dataset row obtained with `df.iloc[0]`.


```{code-cell} python
# Define a function to map the values
def set_mode(row):
    for mode in ['plane', 'train', 'bus', 'car']:
        if row[mode] > 0:
            return mode
        
row = df.iloc[0]
print(set_mode(row))
```

3. Create the new "mode" column by applying "set_mode" to each row of df. (Tip: you can use the `df.apply` method with the correct axis). 


```{code-cell} python
df['mode'] = df.apply(set_mode, axis=1)
```

4. Plot the confidence intervals of the "mean" for each mode chosen. (Tip: you can use seaborn's `sns.pointplot` with `estimator='mean'` and `errorbar='ci'` and 'mode' on the x-axis and 'hinc' on the y-axis).


```{code-cell} python
sns.pointplot(data=df,  x='mode', y='hinc', estimator='mean', errorbar='ci', order=['train', 'bus', 'car', 'plane'])
plt.show()
```
