---
jupytext:
  cell_metadata_filter: all, -hidden, -heading_collapsed, -run_control, -trusted
  notebook_metadata_filter: all, -jupytext.text_representation.jupytext_version, -jupytext.text_representation.format_version,
    -language_info.version, -language_info.codemirror_mode.version, -language_info.codemirror_mode,
    -language_info.file_extension, -language_info.mimetype, -toc
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
  title: Lab Session on Model Selection for Logistic Regression
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa BEDIN<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

+++

The dataset we want to process is `spambase.data`. This file contains 4601 observations and 58 variables have been measured. It corresponds to the analysis of emails. The last column of `spambase.data` denotes whether the e-mail was considered spam (1) or not (0), i.e. unsolicited commercial e-mail. This is the variable to be predicted. Most of the attributes indicate whether a particular word or character was frequently occuring in the e-mail. The run-length attributes (55-57) measure the length of sequences of consecutive capital letters. Here are the definitions of the attributes (see [UCI](http://archive.ics.uci.edu/dataset/94/spambase) for details):

-   48 continuous real : percentage of words in the e-mail that match WORD, i.e. 100 \* (number of times the WORD appears in the e-mail) / total number of words in e-mail.
-   6 continuous real : percentage of characters in the e-mail that match CHAR, i.e. 100 \* (number of CHAR occurences) / total characters in e-mail
-   1 continuous real : average length of uninterrupted sequences of capital letters
-   1 continuous integer : length of longest uninterrupted sequence of capital letters
-   1 continuous integer : sum of length of uninterrupted sequences of capital letters or total number of capital letters in the e-mail
-   1 nominal {0,1} class attribute of type spam ie denotes whether the e-mail was considered spam (1) or not (0), i.e. unsolicited commercial e-mail.

Using the data from `spambase.data`, select the best logistic regression model among all modeling already presented (logistic regression, variable selection on logistic regression, ridge, lasso, and elastic-net). The threshold will be chosen to be equal to 0.5. The dataset will be separated into training set (3/4) and validation set (1/4). This data-splitting can be obtained using `sklearn.model_selection.GroupShuffleSplit`. The validation set will be used to make the selection between the best modeling. The criterion will be the percentage of missclassified. A data-frame containing the various predictions (one column per model and as many rows as there are observations in validation set) is mandatory.

\[`logit` de statsmodels.formula.api, `StandardScaler` de `sklearn.preprocessing`, `LogisticRegressionCV` de `sklearn.linear_model`, et `Pipeline` de `sklearn.pipeline` \]

```{code-cell} ipython3

```
