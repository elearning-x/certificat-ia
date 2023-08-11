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
  title: 'TP choix de variables'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa Bedin<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

## Modules



Importer les modules pandas (comme `pd`) numpy (commme `np`)
matplotlib.pyplot (comme  `plt`) et statsmodels.formula.api (comme `smf`).


```{code-cell} python

```

## Régression ridge sur les données d&rsquo;ozone



#### Importation des données



Importer les données d&rsquo;ozone `ozonecomplet.csv` (dans FunStudio les données sont dans le répertoire `data/`) et éliminer les deux dernières
variables (qualitatives) et faites un résumé numérique par variable \[méthode
`astype` sur la colonne du DataFrame et méthode `describe` sur l&rsquo;instance
DataFrame\]




```{code-cell} python

```

#### Sélection descendante/backward



Proposer une fonction qui permet la sélection descendante/backward. Elle utilisera
les formules de `statsmodels` et incluera toujours la constante. En entrée serviront
trois arguments: le DataFrame des données, la formule de départ et le critère (AIC ou BIC).
La fonction retournera le modèle estimé via `smf.ols`




```{code-cell} python

```
