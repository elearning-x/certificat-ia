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
  title: 'TP comparaison de méthodes de régression logistique'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa BEDIN<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

Le jeu de données que nous voulons traiter est `spambase.data`. Ce fichier contient 4601 observations et 58 variables ont été mesurées. Il correspond à l'analyse d'emails.

La dernière colonne de `spambase.data` indique si l'e-mail a été considéré comme du spam (1) ou non (0), c'est-à-dire comme un e-mail commercial non sollicité. C'est la variable à expliquer. La plupart des attributs indiquent si un mot ou un caractère particulier apparaît fréquemment dans l'e-mail. Les attributs de longueur (55-57) mesurent la longueur des séquences de lettres majuscules consécutives. Voici les définitions des attributs (voir aussi le site de l'[UCI](http://archive.ics.uci.edu/dataset/94/spambase)) :

-   48 attributs réels continus qui représentent le pourcentage de mots de l'e-mail qui correspondent au MOT, c'est-à-dire 100 \* (nombre de fois où le MOT apparaît dans l'e-mail) / nombre total de mots dans l'e-mail.
-   6 attributs réels continus qui représentent le pourcentage de caractères de l'e-mail qui correspondent à CHAR, c'est-à-dire 100 \* (nombre d'occurrences de CHAR) / nombre total de caractères dans l'e-mail.
-   1 continu réel qui représente la longueur moyenne des séquences ininterrompues de lettres majuscules
-   1 entier qui représente la longueur de la plus longue séquence ininterrompue de lettres capitales
-   1 entier qui représente la longueur des séquences ininterrompues de lettres capitales ou du nombre total de majuscules dans l'e-mail
-   1 nominal {0,1} attribut de classe de type spam ie indique si l'e-mail est considéré comme du spam (1) ou non (0), c'est-à-dire un courriel commercial non sollicité.

En utilisant les données de `spambase.data`, sélectionnez le meilleur modèle de régression logistique parmi toutes les modélisations déjà présentées (régression logistique, sélection de variables sur la régression logistique, ridge, lasso et elastic-net). Le seuil sera choisi à 0,5 (sans être optimisé).L'ensemble de données sera séparé en un ensemble d'entraînement (3/4) et un ensemble de validation (1/4). Cette séparation pourra être obtenue avec `sklearn.model_selection.GroupShuffleSplit`.

L'ensemble de validation sera utilisé pour sélectionner les meilleures modélisations. Le critère sera le pourcentage d'erreurs de classification.

Un data-frame contenant les différentes prédictions (une colonne par modèle et autant de lignes qu'il y a d'observations dans le jeu de validation) est obligatoire.

\[`logit` de statsmodels.formula.api, `StandardScaler`, de `sklearn.preprocessing`, `LogisticRegressionCV` de `sklearn.linear_model`, et `Pipeline` de `sklearn.pipeline` \]

```{code-cell} python

```