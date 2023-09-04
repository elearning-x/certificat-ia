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
  title: 'TP variables qualitatives'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa BEDIN<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

+++

# Modules python
Importer les modules pandas (comme `pd`) numpy (commme `np`)
matplotlib.pyplot (comme  `plt`), statsmodels.formula.api (comme `smf`)
et statsmodels.api (comme `sm`)


```{code-cell} python

```

# Variables qualitatives et quantitatives pour l'ozone

### Importation
Importer les données `ozonecomplet.csv` (dans Fun Campus les données sont dans le répertoire `data/`) et transformer les deux dernières 
variables en variables qualitatives et faites un résumé numérique par variable
\[méthode `astype` sur la colonne du DataFrame et
méthode `describe` sur l'instance DataFrame\]


```{code-cell} python

```

### Premier modèle
 Effectuer une régression avec comme variables explicatives `T12`,`Ne` et `Dv`,
  combien estime t on de paramètres ?


```{code-cell} python

```

### Résumé du modèle
Où sont passés les coefficients associés au vent d'Est et au temps nuageux
dans le résumé du modèle ?



### Changement de modalité de référence
Changer la modalité de référence du vent pour le vent du Nord,
\[fonction `C` dans la formule de la régression, voir https://www.statsmodels.org/stable/example_formulas.html)
 `Treatment` option `reference`\].
- Vérifier que la valeur de l'intercept a changé ainsi que toutes les valeurs 
  des estimateurs des paramètres associés au vent. 
- Vérifier que les $Y$ ajustés sont les mêmes.


```{code-cell} python

```

### Regroupement de modalité
- Regroupez Est et Nord et faites un nouveau modèle. \[méthode
  `map` sur la colonne puis `astype`\]
- Quel est le modèle retenu entre celui-ci et le précédent ? 
  Proposez deux tests pour répondre à cette question.
  [`sm.stats.anova_lm`]


```{code-cell} python

```

# Teneur en folate dans les globules rouges
Nous disposons de la mesure de concentration (en $\mu\mathrm{g/l}$) de folate
(nom de variable `folate`) dans les globules rouge durant une
anesthésie chez $n=22$ patients. L'anesthésie est utilise une
ventilation choisie parmi trois méthodes:
- le gaz utilisé est un mélange 50-50 de $\mathrm{N}_2$O 
  (oxyde nitreux ou gaz hilarant) et d'$\mathrm{O}_2$ 
  pendant une durée 24h (codé `N2O+O2,24h`);
- le gaz utilisé est un mélange 50-50 de $\mathrm{N}_2$O 
  (oxyde nitreux ou gaz hilarant) et d'$\mathrm{O}_2$ 
  uniquement pendant la durée de l'opération (codé `N2O+O2,op`);
- pas d'oxyde nitreux, uniquement de l'oxygène pendant 24h (codé `O2,24h`).
Nous cherchons à savoir si ces trois méthodes de ventilations sont
équivalentes.

### Importation
Importer les données qui sont dans le fichier `gr.csv` (dans Fun Campus les données sont dans le répertoire `data/`) et
résumer les de façon numérique.
\[méthode `astype` sur la colonne du DataFrame et
méthode `describe` sur l'instance DataFrame\]


```{code-cell} python

```

### Représentation graphique
Représenter graphiquement les données.
\[`plt.plot` ou méthode `groupby` sur l'instance de DataFrame et méthode
`boxplot` sur l'instance DataFrame groupé\]


```{code-cell} python

```

### Méthode de ventilation
Répondre à la question suivante: les trois méthodes de
  ventilation sont elles équivalentes ?


```{code-cell} python

```

### Analyse du modèle
Analyser les résidus du modèle retenu et interpréter les
  coefficients
  \[`plt.plot`, `get_influence`, `resid_studentized_external`,
  `sm.qqplot`\]


```{code-cell} python

```

# ANOVA à deux facteurs
Nous disposons de la hauteur moyenne de 8 provenances d'eucalyptus
camaldulensis: les graines de ces eucalyptus ont été récoltées dans
huit endroits du monde (ie les 8 provenances) et plantées aux environs
de Pointe-Noire (Congo). Au même âge sont mesurées les hauteurs
moyennes pour ces 8 provenances. Ces provenances sont plantées dans
une très grande parcelle que l'on soupçonne de ne pas être homogène
du simple fait de sa taille. Cette parcelle est donc divisée en sous 
parcelles appelées bloc que l'on espère être homogène. Les données
propose les hauteurs moyennes des arbres par bloc-provenance.

Nous souhaitons savoir si ces huit provenances sont identiques.

1. Importer les données qui sont dans le fichier `eucalyptus_camaldulensis.txt` (dans Fun Campus les données sont dans le répertoire `data/`) et
   résumer les de façon numérique.
2. Représenter graphiquement les données utilisées pour la réponse à la question.
3. Répondre à la question posée (les huit provenances sont elles identiques ?).
   Où intervient (indirectement) la variable `bloc` dans la statistique de test utilisée ?
4. Analyser les résidus du modèle retenu. 
   Tracer les résidus en fonction de la variable `bloc`.


```{code-cell} python

```
