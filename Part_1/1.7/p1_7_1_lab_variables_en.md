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
  title: 'Lab Session on Variable Selection'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa BEDIN<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

+++

## Modules



Import the modules pandas (as `pd`), numpy (as `np`), matplotlib.pyplot (as `plt`), and statsmodels.formula.api (as `smf`).




```{code-cell} python

```

## Ridge Regression on Ozone Data



### Data Import



Import the ozone data `ozonecomplet.csv` (in Fun Campus, data is located in the `data/` directory) and remove the last two variables (categorical). Then provide a summary of numerical variables.
[Use the `astype` method on the DataFrame column and the `describe` method on the DataFrame instance.]




```{code-cell} python

```

### Backward Selection



Propose a function that performs backward selection. It will use the formulas from `statsmodels` and will always include the intercept. The function will take three arguments as input: the DataFrame of data, the initial formula, and the criterion (AIC or BIC). The function will return the estimated model using `smf.ols`.




```{code-cell} python

```
