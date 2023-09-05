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
  title: 'Lab Session on Qualitative Variables'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa BEDIN<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

+++

## Python Modules



Import the modules pandas (as `pd`), numpy (as `np`), matplotlib.pyplot (as `plt`), statsmodels.formula.api (as `smf`), and statsmodels.api (as `sm`).




```{code-cell} python

```

## Qualitative and Quantitative Variables for Ozone



### Data Import



Import the data `ozonecomplet.csv` and convert the last two variables into qualitative variables. Then, provide a numerical summary for each variable.
[ Use the `astype` method on the DataFrame column and the `describe` method on the DataFrame instance. ]




```{code-cell} python

```

### First Model



Perform a regression with explanatory variables `T12`, `Ne`, and `Dv`. How many parameters are estimated?




```{code-cell} python

```

### Model Summary



Where did the coefficients associated with East wind and cloudy weather go in the model summary?



### Change of Reference Modality



Change the reference modality to North wind. [ Use the `C` function in the regression formula, see [https://www.statsmodels.org/stable/example_formulas.html](https://www.statsmodels.org/stable/example_formulas.html)) with the `Treatment` option `reference` ].

-   Verify that the value of the intercept has changed, as well as all the parameter estimator values associated with wind.
-   Verify that the adjusted $Y$ values remain the same.




```{code-cell} python

```

### Modalities Grouping



-   Group East and North winds and create a new model. [ Use the `map` method on the column, then `astype` ]
-   Which model is preferred between this one and the previous one? Propose two tests to answer this question. [ `sm.stats.anova_lm` ]




```{code-cell} python

```

The summary row in `reg` corresponding to `Nord` provides the test for the nullity of the corresponding coefficient. In this case, it corresponds to the nullity of the difference between the modality `Nord` and the reference modality `East` that is  the two modalities have been merged).



## Red Blood Cell Folate Content



We have measurements of folate concentration (in $\mu\mathrm{g/l}$) in red blood cells during anesthesia for $n=22$ patients. Anesthesia involves three ventilation methods:

-   50-50 mixture of $\mathrm{N}<sub>2</sub>$O (nitrous oxide or laughing gas) and $\mathrm{O}_2$ for 24 hours (coded `N2O+O2,24h`)
-   50-50 mixture of $\mathrm{N}<sub>2</sub>$O and $\mathrm{O}_2$ only during surgery (coded `N2O+O2,op`)
-   Pure oxygen ventilation for 24 hours (coded `O2,24h`)

We aim to determine if these three ventilation methods are equivalent.



### Data Import



Import the data from the file `gr.csv` and provide a numerical summary.
[ Use the `astype` method on the DataFrame column and the `describe` method on the DataFrame instance. ]




```{code-cell} python

```

### Graphical Representation



Graphically represent the data.
[ Use `plt.plot` or the `groupby` method on the DataFrame instance and the `boxplot` method on the grouped DataFrame instance. ]
The simplest way is to create points for each ventilation method.




```{code-cell} python

```

We observe that the sample sizes in each group are small, the means seem different, and the variabilities appear comparable. A boxplot is also informative, although less suitable for small sample sizes per group.




```{code-cell} python

```

### Ventilation Method



Answer the following question: Are the three ventilation methods equivalent?
Conduct an $F$ test between two nested models $\mathrm{H}_0: \ y_{ij}=\mu + \varepsilon_{ij}$ and $\mathrm{H}_1: \ y_{ij}=\mu + \alpha_i + \varepsilon_{ij}$ with a significance level of $\alpha=5\%$.




```{code-cell} python

```

The test statistic value is $3.71$, and its critical probability is $0.04$, smaller than $\alpha$. Thus, we reject $\mathrm{H}_0$. The type of ventilation has an effect.



### Model Analysis



Analyze the residuals of the retained model and interpret the coefficients.
[ Use `plt.plot`, `get_influence`, `resid_studentized_external`, `sm.qqplot` ]

The model errors are expected to be independently and identically distributed with a normal distribution of mean $0$ and variance $\sigma^2$. The studentized residuals (by VC) can be plotted against the predicted values $\hat Y$ (the group ventilation mean).




```{code-cell} python

```

We observe that the variability seems higher in one group. However, due to the low number of observations, we cannot conclusively conclude a serious issue.

To assess normality, it&rsquo;s common to use a QQ-plot:

-   The studentized residuals are ordered (ascending order): $t^*_{(1)},\dotsc t^*_{(n)}$
-   Let $Z_{(1)},\dotsc,Z_{(n)}$ be an $n$-sample drawn from a $\mathcal{N}(0,1)$ distribution and ordered in ascending order. We then estimate the mean value of $Z_{(i)}$ (denoted as $\bar Z_{(i)}$).
-   Plot the $n$ pairs $\bar Z_{(i)},t^*_{(i)}$




```{code-cell} python

```

## Two-Factor ANOVA



We have the average height measurements of 8 provenances of eucalyptus camaldulensis. The seeds of these eucalyptus trees were collected from eight locations around the world (i.e., 8 provenances) and planted near Pointe-Noire (Congo). At the same age, the average heights are measured for these 8 provenances. These provenances are planted in a very large plot suspected of not being homogeneous due to its size. Hence, the plot is divided into subplots called blocks, which are hoped to be homogeneous. The data provide the average tree heights per block-provenance combination.

We want to determine if these eight provenances are identical.

1.  Data Import
    Import the data from the file `eucalyptus_camaldulensis.txt` and provide a numerical summary.
    [ Use the `astype` method on the DataFrame columns and the `describe` method on the DataFrame instance. ]




```{code-cell} python

```

1.  Graphical Representation
    Graphically represent the data used to answer the question.




```{code-cell} python

```

1.  Answer the Question
    Are the eight provenances identical? Where does the `bloc` variable indirectly intervene in the used test statistic?




```{code-cell} python

```

1.  Residual Analysis
    Analyze the residuals of the retained model. Plot the residuals against the `bloc` variable.




```{code-cell} python

```
