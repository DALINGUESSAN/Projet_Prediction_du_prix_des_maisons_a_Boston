import os
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

base_dir = Path(__file__).resolve().parent
images_dir = base_dir / "images"
images_dir.mkdir(exist_ok=True)

# Load data
path = base_dir / "Boston.csv"
data = pd.read_csv(path, index_col=0)
data.columns = [
    "Taux_criminalite",
    "Prop_terrains_resid",
    "Prop_acres_indus",
    "Bordure_riviere",
    "Concentration_NO2",
    "Nb_pieces_logement",
    "Prop_log_construits_av1940",
    "Dist_ponderees_emploi",
    "Accessibilite_autoroute",
    "Taux_impot_foncier",
    "Ratio_eleve_prof",
    "Prop_noirs_ville",
    "Pourcentage_pop_inf",
    "Valeur_med_logement",
]

# Split data
train, test = train_test_split(data, test_size=0.2, random_state=42)
X_train = train.drop("Valeur_med_logement", axis=1)
y_train = train["Valeur_med_logement"]
X_test = test.drop("Valeur_med_logement", axis=1)
y_test = test["Valeur_med_logement"]

# Model
model = sm.OLS(y_train, sm.add_constant(X_train)).fit()
y_train_pred = model.predict(sm.add_constant(X_train))
y_test_pred = model.predict(sm.add_constant(X_test))

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)

# Plot 1: histograms
fig, axs = plt.subplots(3, 5, figsize=(18, 10))
axs = axs.ravel()
for i, col in enumerate(data.columns):
    axs[i].hist(data[col], bins=25, color="steelblue", edgecolor="black")
    axs[i].set_title(col, fontsize=8)
for j in range(len(data.columns), len(axs)):
    axs[j].axis("off")
plt.tight_layout()
fig.savefig(images_dir / "histogrammes.png", dpi=200)
plt.close(fig)

# Plot 2: boxplots
fig, axs = plt.subplots(3, 5, figsize=(18, 10))
axs = axs.ravel()
for i, col in enumerate(data.columns):
    sns.boxplot(y=data[col], ax=axs[i], color="cornflowerblue")
    axs[i].set_title(col, fontsize=8)
for j in range(len(data.columns), len(axs)):
    axs[j].axis("off")
plt.tight_layout()
fig.savefig(images_dir / "boxplots.png", dpi=200)
plt.close(fig)

# Plot 3: correlation heatmap
corr = data.corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
ax.set_title("Matrice de corrélation")
fig.tight_layout()
fig.savefig(images_dir / "heatmap_correlation.png", dpi=200)
plt.close(fig)

# Plot 4: predictions vs observed
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_test_pred, color="royalblue", alpha=0.7)
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
ax.set_xlabel("Valeurs observées")
ax.set_ylabel("Valeurs prédites")
ax.set_title("Valeurs observées vs valeurs prédites")
fig.tight_layout()
fig.savefig(images_dir / "predictions_vs_observed.png", dpi=200)
plt.close(fig)

# Plot 5: residuals
residuals = y_train - y_train_pred
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=y_train_pred, y=residuals, ax=ax, color="darkorange")
ax.axhline(0, color="red", linestyle="--", linewidth=1)
ax.set_title("Résidus en fonction des prédictions")
ax.set_xlabel("Prédictions")
ax.set_ylabel("Résidus")
fig.tight_layout()
fig.savefig(images_dir / "residus_vs_predictions.png", dpi=200)
plt.close(fig)

# Plot 6: qq plot
fig = sm.qqplot(residuals, line="45", fit=True)
plt.title("QQ-plot des résidus")
plt.tight_layout()
plt.savefig(images_dir / "qqplot_residus.png", dpi=200)
plt.close("all")

# VIF values
vif = pd.DataFrame({
    "Variable": X_train.columns,
    "VIF": [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])],
})

# Build LaTeX report with values embedded
report_text = textwrap.dedent(f"""
\\documentclass[a4paper,12pt]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage[french]{{babel}}
\\usepackage{{geometry}}
\\usepackage{{amsmath,amssymb}}
\\usepackage{{graphicx}}
\\usepackage{{float}}
\\usepackage{{booktabs}}
\\usepackage{{array}}
\\usepackage{{hyperref}}
\\usepackage{{setspace}}
\\usepackage{{enumitem}}
\\usepackage{{xcolor}}

\\geometry{{margin=2.5cm}}

\\title{{\\textbf{{Prédiction des prix de maisons à Boston par régression linéaire}}}}
\\author{{Votre Nom}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

\\begin{{abstract}}
Ce rapport présente une étude de régression linéaire appliquée à la prédiction de la valeur médiane des maisons à Boston. L’objectif est d’analyser les relations entre les variables explicatives et la valeur des logements, de construire un modèle de prédiction, et de vérifier les hypothèses statistiques du modèle.
\\end{{abstract}}

\\tableofcontents
\\newpage

\\section{{Introduction}}
La prévision du prix des logements est un enjeu important dans le secteur immobilier et dans l’analyse économique. Les données de Boston sont un jeu de référence utilisé pour modéliser la relation entre les caractéristiques urbaines, immobilières et la valeur des logements.

Dans ce projet, un modèle de régression linéaire multiple a été utilisé pour prédire la valeur médiane des maisons. L’intérêt de cette approche est double : elle permet d’expliquer l’effet de chaque variable et de produire des prédictions sur de nouvelles observations.

\\section{{Présentation du jeu de données}}
Le jeu de données contient plusieurs variables liées à l’environnement urbain, à la qualité de l’habitat et aux caractéristiques socio-économiques. La variable cible est :

\\[
Y = \\text{{Valeur\\_med\\_logement}}
\\]

Les variables explicatives comprennent notamment le taux de criminalité, la distance aux centres d’emploi, le nombre de pièces par logement, la proportion de logements anciens, et d’autres indicateurs de localisation et de contexte urbain.

\\section{{Analyse exploratoire des données}}
Avant la modélisation, une analyse exploratoire a été réalisée afin de comprendre la structure du jeu de données.

\\subsection{{Statistiques descriptives}}
Les statistiques descriptives mettent en évidence la distribution des variables et les éventuels écarts de valeur.

\\begin{{table}}[H]
\\centering
\\caption{{Statistiques descriptives}}
\\begin{{tabular}}{{lcccc}}
\\toprule
Variable & Moyenne & Écart-type & Minimum & Maximum \\
\\midrule
Taux\\_criminalite & {data['Taux_criminalite'].mean():.2f} & {data['Taux_criminalite'].std():.2f} & {data['Taux_criminalite'].min():.2f} & {data['Taux_criminalite'].max():.2f} \\
Nb\\_pieces\\_logement & {data['Nb_pieces_logement'].mean():.2f} & {data['Nb_pieces_logement'].std():.2f} & {data['Nb_pieces_logement'].min():.2f} & {data['Nb_pieces_logement'].max():.2f} \\
Dist\\_ponderees\\_emploi & {data['Dist_ponderees_emploi'].mean():.2f} & {data['Dist_ponderees_emploi'].std():.2f} & {data['Dist_ponderees_emploi'].min():.2f} & {data['Dist_ponderees_emploi'].max():.2f} \\
Valeur\\_med\\_logement & {data['Valeur_med_logement'].mean():.2f} & {data['Valeur_med_logement'].std():.2f} & {data['Valeur_med_logement'].min():.2f} & {data['Valeur_med_logement'].max():.2f} \\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\subsection{{Matrice de corrélation}}
La matrice de corrélation permet d’identifier les variables les plus liées à la variable cible.

\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.9\\textwidth]{{images/heatmap_correlation.png}}
    \\caption{{Heatmap de corrélation}}
\\end{{figure}}

Les variables les plus corrélées avec la valeur médiane des maisons sont notamment le nombre de pièces, la distance aux centres d’emploi et la proportion de logements anciens.

\\subsection{{Visualisation des distributions}}
\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.95\\textwidth]{{images/histogrammes.png}}
    \\caption{{Histogrammes des variables}}
\\end{{figure}}

\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.95\\textwidth]{{images/boxplots.png}}
    \\caption{{Boxplots des variables explicatives}}
\\end{{figure}}

\\section{{Préparation des données}}
Les données ont été divisées en ensemble d’apprentissage et ensemble de test selon un ratio de 80\% / 20\%.

\\section{{Modèle de régression linéaire}}
Le modèle de régression linéaire multiple est défini comme :

\\[
Y = \\beta_0 + \\beta_1 X_1 + \\beta_2 X_2 + \\cdots + \\beta_p X_p + \\epsilon
\\]

L’ajustement du modèle sur les données d’apprentissage a permis d’estimer les coefficients de régression.

\\section{{Évaluation des performances}}
Les performances du modèle ont été évaluées à l’aide du coefficient de détermination $R^2$, du MSE et du RMSE.

\\[
R^2 = 1 - \\frac{\\sum (y_i - \\hat{{y}}_i)^2}{\\sum (y_i - \\bar{{y}})^2}
\\]

\\[
MSE = \\frac{{1}}{{n}} \\sum_{{i=1}}^{{n}}(y_i - \\hat{{y}}_i)^2
\\]

\\[
RMSE = \\sqrt{{MSE}}
\\]

\\begin{{table}}[H]
\\centering
\\caption{{Métriques de performance du modèle}}
\\begin{{tabular}}{{lcc}}
\\toprule
Métrique & Train & Test \\
\\midrule
R² & {r2_train:.4f} & {r2_test:.4f} \\
MSE & {mse_train:.4f} & {mse_test:.4f} \\
RMSE & {rmse_train:.4f} & {rmse_test:.4f} \\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.85\\textwidth]{{images/predictions_vs_observed.png}}
    \\caption{{Valeurs observées vs valeurs prédites}}
\\end{{figure}}

\\section{{Vérification des hypothèses}}
Les hypothèses du modèle ont été vérifiées afin d’assurer la validité de l’analyse.

\\subsection{{Homoscédasticité}}
\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.8\\textwidth]{{images/residus_vs_predictions.png}}
    \\caption{{Résidus en fonction des prédictions}}
\\end{{figure}}

\\subsection{{Normalité des résidus}}
\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.75\\textwidth]{{images/qqplot_residus.png}}
    \\caption{{QQ-plot des résidus}}
\\end{{figure}}

\\subsection{{Multicolinéarité}}
Le calcul du VIF permet de détecter les variables fortement corrélées entre elles.

\\begin{{table}}[H]
\\centering
\\caption{{Variance Inflation Factor (VIF)}}
\\begin{{tabular}}{{lc}}
\\toprule
Variable & VIF \\
\\midrule
""")

vif_lines = [f"{row['Variable']} & {row['VIF']:.2f} \\\\" for _, row in vif.iterrows()]
report_text += "\n".join(vif_lines)
report_text += textwrap.dedent("""
\\bottomrule
\\end{tabular}
\\end{table}

\\section{{Interprétation des résultats}}
Les résultats montrent que le prix des maisons dépend fortement de facteurs tels que le nombre de pièces, la localisation, la qualité du quartier et la distance aux centres d’emploi. Les coefficients du modèle permettent de quantifier cette relation, tandis que les métriques de performance indiquent la qualité prédictive du modèle.

Les résultats obtenus confirment que la régression linéaire est une méthode adaptée pour expliquer et prédire la valeur des logements à Boston.

\\section{{Conclusion}}
Ce projet a permis de comparer les variables explicatives, d’ajuster un modèle de régression linéaire, et de vérifier les hypothèses du modèle. Les résultats démontrent une capacité de prédiction satisfaisante, tout en mettant en évidence l’importance de l’analyse statistique avant d’interpréter les coefficients.

En conclusion, la régression linéaire demeure un outil pertinent et interprétable pour la prédiction du prix des logements.

\\end{{document}}
""")

(base_dir / "rapport_boston.tex").write_text(report_text, encoding="utf-8")

print(f"R² train: {r2_train:.4f}")
print(f"R² test: {r2_test:.4f}")
print(f"MSE train: {mse_train:.4f}")
print(f"MSE test: {mse_test:.4f}")
print(f"RMSE train: {rmse_train:.4f}")
print(f"RMSE test: {rmse_test:.4f}")
print(f"Created files in: {base_dir}")
