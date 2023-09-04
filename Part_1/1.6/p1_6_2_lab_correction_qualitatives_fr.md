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
  title: 'Correction du TP variables qualitatives'
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Lisa Bedin<br />Pierre André CORNILLON<br />Eric MATZNER-LOBER</span>
<span>Licence CC BY-NC-ND</span>
</div>

## Modules python



Importer les modules pandas (comme `pd`) numpy (commme `np`)
matplotlib.pyplot (comme `plt`), statsmodels.formula.api (comme `smf`)
et statsmodels.api (comme `sm`)




```{code-cell} python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
```

## Variables qualitatives et quantitatives pour l&rsquo;ozone



### Importation



Importer les données `ozonecomplet.csv` et transformer les deux
dernières variables en variables qualitatives et faites un résumé
numérique par variable [ méthode `astype` sur la colonne du DataFrame et
méthode `describe` sur l&rsquo;instance DataFrame ]




```{code-cell} python
ozone = pd.read_csv("data/ozonecomplet.csv", header=0, sep=";")
ozone = ozone.drop(['nomligne'], axis=1)
ozone.Ne = ozone.Ne.astype("category")
ozone.Dv = ozone.Dv.astype("category")
ozone.describe(include="all")
```

### Premier modèle



Effectuer une régression avec comme variables explicatives `T12`,=Ne= et
`Dv`, combien estime t on de paramètres ?




```{code-cell} python
reg = smf.ols("O3~T12+Ne+Dv", data=ozone).fit()
reg.summary()
```

### Résumé du modèle



Où sont passés les coefficients associés au vent d&rsquo;Est et au temps
nuageux dans le résumé du modèle ?

Il s&rsquo;agit des modalités de références (première par ordre alphabétique).
La constante (intercept) correspond à une température à midi (`T12`) de
0 degré avec un vent d&rsquo;Est et un temps nuageux. Quand le temps
correspond à cette définition alors on prévoit -20 microgramme par m3
d&rsquo;ozone. Le modèle dans cette plage n&rsquo;est pas adapté (car nous n&rsquo;avons
pas de données)



### Changement de modalité de référence



Changer la modalité de référence du vent pour le vent du Nord, [ fonction
`C` dans la formule de la régression, voir
[https://www.statsmodels.org/stable/example_formulas.html](https://www.statsmodels.org/stable/example_formulas.html)) `Treatment`
option `reference` ]. - Vérifier que la valeur de l&rsquo;intercept a changé
ainsi que toutes les valeurs des estimateurs des paramètres associés au
vent. - Vérifier que les $Y$ ajustés sont les mêmes.




```{code-cell} python
reg2 = smf.ols("O3~T12+Ne+C(Dv, Treatment(reference=1))", data=ozone).fit()
reg2.summary()
```


```{code-cell} python
np.all(np.abs(reg.predict() -reg2.predict())<1e-10)
```

### Regroupement de modalité



-   Regroupez Est et Nord et faites un nouveau modèle. [ méthode `map` sur
    la colonne puis `astype` ]
-   Quel est le modèle retenu entre celui-ci et le précédent ? Proposez
    deux tests pour répondre à cette question. [ `sm.stats.anova_lm` ]




```{code-cell} python
Dv2 = ozone.Dv.map({"E": "E+N", "N": "E+N", "O": "O", "S": "S"}).astype("category")
ozone["Dv2"] = Dv2
reg3 = smf.ols("O3~T12+Ne+Dv2", data=ozone).fit()
reg3.summary()
```


```{code-cell} python
sm.stats.anova_lm(reg3, reg)
```

La ligne du résumé de `reg` correspondant à `Nord` donne le test de
nullité du coefficient correspondant, ce qui correspond ici à la nullité
de l&rsquo;écart de cette modalité `Nord` par rapport à celle de référence
`Est` et donc à avoir les deux modalités fusionnées.



## Teneur en folate dans les globules rouges



Nous disposons de la mesure de concentration (en $\mu\mathrm{g/l}$) de
folate (nom de variable `folate`) dans les globules rouge durant une
anesthésie chez $n=22$ patients. L&rsquo;anesthésie est utilise une
ventilation choisie parmi trois méthodes: - le gaz utilisé est un
mélange 50-50 de $\mathrm{N}<sub>2</sub>$O (oxyde nitreux ou gaz hilarant) et
d&rsquo;$\mathrm{O}_2$ pendant une durée 24h (codé `N2O+O2,24h`); - le gaz
utilisé est un mélange 50-50 de $\mathrm{N}<sub>2</sub>$O (oxyde nitreux ou gaz
hilarant) et d&rsquo;$\mathrm{O}_2$ uniquement pendant la durée de l&rsquo;opération
(codé `N2O+O2,op`); - pas d&rsquo;oxyde nitreux, uniquement de l&rsquo;oxygène
pendant 24h (codé `O2,24h`). Nous cherchons à savoir si ces trois
méthodes de ventilations sont équivalentes.



### Importation



Importer les données qui sont dans le fichier `gr.csv` et résumer les de
façon numérique. [ méthode `astype` sur la colonne du DataFrame et
méthode `describe` sur l&rsquo;instance DataFrame ]




```{code-cell} python
gr = pd.read_csv("data/gr.csv", header=0, sep=";")
gr["ventilation"]=gr["ventilation"].astype("category")
gr.describe(include="all")
```

### Représentation graphique



Représenter graphiquement les données. [ `plt.plot` ou méthode `groupby`
sur l&rsquo;instance de DataFrame et méthode `boxplot` sur l&rsquo;instance
DataFrame groupé ]

Le plus simple est de faire soit des points par ventilation




```{code-cell} python
plt.plot(gr.ventilation, gr.folate, "o")
```

Nous constatons que les effectifs dans chaque groupe sont faibles, que
les moyennes par groupe semblent différentes et que les variabilités
semblent comparables.

Mais un boxplot est aussi intéressant quoique moins adapté pour ces
faibles effectifs par groupe.




```{code-cell} python
gr.groupby(by='ventilation').boxplot(False)
plt.show()
```

### Méthode de ventilation



Répondre à la question suivante: les trois méthodes de ventilation sont
elles équivalentes ?

Faisons un test $F$ entre les deux modèles emboités
$\mathrm{H}_0: \ y_{ij}=\mu + \varepsilon_{ij}$ et
$\mathrm{H}_1: \ y_{ij}=\mu + \alpha_i + \varepsilon_{ij}$ avec l&rsquo;erreur
de première espèce de $\alpha=5\%$.




```{code-cell} python
modele1 = smf.ols("folate ~ 1 + ventilation", data=gr).fit()
modele0 = smf.ols("folate ~ 1", data=gr).fit()
sm.stats.anova_lm(modele0, modele1)
```

La valeur de la statistique de test vaut $3.71$ et sa probabilité
critique vaut 0.04, plus petite que $\alpha$ nous repoussons donc
$\mathrm{H}_0$. Le type de ventilation a un effet.



### Analyse du modèle



Analyser les résidus du modèle retenu et interpréter les coefficients
[ `plt.plot`, `get_influence`, `resid_studentized_external`, `sm.qqplot` ]

Les erreurs du modèle sont sensées être iid de loi normale de moyenne 0
et de variance $&sigma;<sup>2</sup>. Les résidus studentisés (par VC) peuvent être
tracés en fonction de $\hat Y$ (qui est la moyenne du groupe de
ventilation.)




```{code-cell} python
infl = modele1.get_influence()
plt.plot(modele1.predict(), infl.resid_studentized_external, "o")
```

Nous retrouvons que la variabilité semble être plus élevé dans un groupe
mais compte tenu du faible nombre d&rsquo;observations on ne peut franchement
parler de problème rédibitoire.

Pour envisager la normalité il est classique de regarder un QQ-plot -
Les résidus studentisés sont ordonnés (ordre croissant):
$t^*_{(1)},\dotsc t^*_{(n)}$ - Soit $Z_{(1)},\dotsc,Z_{(n)}$ un
$n$-échantillon tiré selon une loi $\mathcal{N}(0,1)$ puis ordonné dans
l&rsquo;ordre croissant. On estime alors la valeur moyenne des $Z_{(i)}$
(estimation notée $\bar Z_{(i)}$) - On trace alors les $n$ couples
$\bar Z_{(i)},t^*_{(i)}$




```{code-cell} python
sm.qqplot(infl.resid_studentized_external, line ='s')
```

La normalité des résidus semble correcte. Notre conclusion sur un effet
du type de ventilation n&rsquo;est pas amoindrie car le modèle de régression
(ici nommé ANOVA à un facteur) semble correct.



## ANOVA à deux facteurs



Nous disposons de la hauteur moyenne de 8 provenances d&rsquo;eucalyptus
camaldulensis: les graines de ces eucalyptus ont été récoltées dans huit
endroits du monde (ie les 8 provenances) et plantées aux environs de
Pointe-Noire (Congo). Au même âge sont mesurées les hauteurs moyennes
pour ces 8 provenances. Ces provenances sont plantées dans une très
grande parcelle que l&rsquo;on soupçonne de ne pas être homogène du simple
fait de sa taille. Cette parcelle est donc divisée en sous parcelles
appelées bloc que l&rsquo;on espère être homogène. Les données propose les
hauteurs moyennes des arbres par bloc-provenance.

Nous souhaitons savoir si ces huit provenances sont identiques.

1.  Importer les données qui sont dans le fichier
    `eucalyptus_camaldulensis.txt` et résumer les de façon numérique.




```{code-cell} python
camal = pd.read_csv("data/eucalyptus_camaldulensis.txt", header=0, sep=" ", decimal=",")
camal.bloc = camal.bloc.astype("category")
camal.provenance = camal.provenance.astype("category")
camal.describe(include="all")
```

1.  Représenter graphiquement les données utilisées pour la réponse à la
    question.




```{code-cell} python
camal.groupby(by="provenance").boxplot(False)
plt.show()
```

Les provenances 2 et 4 semblent nettement supérieures.

1.  Répondre à la question posée (les huit provenances sont elles
    identiques ?). Où intervient (indirectement) la variable `bloc` dans
    la statistique de test utilisée ?




```{code-cell} python
modele0 = smf.ols("hauteur~bloc", data=camal).fit()
modele1 = smf.ols("hauteur~bloc+provenance", data=camal).fit()
sm.stats.anova_lm(modele0, modele1)
```

La valeur de la statistique de test vaut $26.65$ et sa probabilité
critique est quasi nulles, plus petite que $\alpha=1\%$ nous repoussons
donc $\mathrm{H}_0$. La provenance a un effet (confirmant le graphique
de la question précédente)

La statistique $F$ compare la variabilité entre provenance (numérateur)
et la variabilité résiduelle ($\hat\sigma^2$ au dénominateur). Pour
améliorer la sensibilité du test il est important d&rsquo;avoir une petite
variabilité résiduelle et donc d&rsquo;inclure les variables explicatives même
si elles ne sont pas dans le questionnement initial, ici la variable
`bloc`.

1.  Analyser les résidus du modèle retenu. Tracer les résidus en fonction
    de la variable `bloc`.




```{code-cell} python
camal["rstudent"] = modele1.get_influence().resid_studentized_external
plt.plot(modele1.predict(), camal.rstudent, "*")
#.boxplot()
```


```{code-cell} python
camal.loc[:,["rstudent", "bloc"]].groupby(by="bloc").boxplot(False)
```

Les résidus semblent corrects, le test est très significatif nous sommes
donc assez certains de notre conclusion: la provenance a bien un effet
sur la hauteur.


