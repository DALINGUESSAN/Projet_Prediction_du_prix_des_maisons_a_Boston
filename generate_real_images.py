import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

base = r"C:\Users\EMPEREUR DALIKAN\Desktop\Base de Donnees ML&DL\PRIX MAISON BOSTON"
img_dir = os.path.join(base, 'images')
os.makedirs(img_dir, exist_ok=True)

# Charger les données
csv_path = os.path.join(base, 'Boston.csv')
data = pd.read_csv(csv_path, index_col=0)
data.columns = [
    'Taux_criminalite', 'Prop_terrains_resid', 'Prop_acres_indus', 'Bordure_riviere', 'Concentration_NO2',
    'Nb_pieces_logement', 'Prop_log_construits_av1940', 'Dist_ponderees_emploi', 'Accessibilite_autoroute',
    'Taux_impot_foncier', 'Ratio_eleve_prof', 'Prop_noirs_ville', 'Pourcentage_pop_inf', 'Valeur_med_logement'
]

# Modèle simple pour générer les résidus
train, test = train_test_split(data, test_size=0.2, random_state=42)
X_train = train.drop('Valeur_med_logement', axis=1)
y_train = train['Valeur_med_logement']
X_test = test.drop('Valeur_med_logement', axis=1)
y_test = test['Valeur_med_logement']
model = sm.OLS(y_train, sm.add_constant(X_train)).fit()
y_train_pred = model.predict(sm.add_constant(X_train))
y_test_pred = model.predict(sm.add_constant(X_test))
residuals = y_train - y_train_pred

# 1. Histogrammes
fig, axs = plt.subplots(3, 5, figsize=(18, 10))
axs = axs.ravel()
for i, col in enumerate(data.columns):
    axs[i].hist(data[col], bins=25, color='steelblue', edgecolor='black')
    axs[i].set_title(col, fontsize=9)
for j in range(len(data.columns), len(axs)):
    axs[j].axis('off')
plt.tight_layout()
fig.savefig(os.path.join(img_dir, 'histogrammes.png'), dpi=200)
plt.close(fig)

# 2. Boxplots
fig, axs = plt.subplots(3, 5, figsize=(18, 10))
axs = axs.ravel()
for i, col in enumerate(data.columns):
    sns.boxplot(y=data[col], ax=axs[i], color='cornflowerblue')
    axs[i].set_title(col, fontsize=9)
for j in range(len(data.columns), len(axs)):
    axs[j].axis('off')
plt.tight_layout()
fig.savefig(os.path.join(img_dir, 'boxplots.png'), dpi=200)
plt.close(fig)

# 3. Heatmap corrélation
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(data.corr(), annot=False, cmap='coolwarm', ax=ax)
ax.set_title('Matrice de corrélation')
fig.tight_layout()
fig.savefig(os.path.join(img_dir, 'heatmap_correlation.png'), dpi=200)
plt.close(fig)

# 4. Observed vs predicted
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_test_pred, color='royalblue', alpha=0.7)
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
ax.set_xlabel('Valeurs observées')
ax.set_ylabel('Valeurs prédites')
ax.set_title('Valeurs observées vs valeurs prédites')
fig.tight_layout()
fig.savefig(os.path.join(img_dir, 'predictions_vs_observed.png'), dpi=200)
plt.close(fig)

# 5. Résidus vs prédictions
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=y_train_pred, y=residuals, ax=ax, color='darkorange')
ax.axhline(0, color='red', linestyle='--', linewidth=1)
ax.set_title('Résidus en fonction des prédictions')
ax.set_xlabel('Prédictions')
ax.set_ylabel('Résidus')
fig.tight_layout()
fig.savefig(os.path.join(img_dir, 'residus_vs_predictions.png'), dpi=200)
plt.close(fig)

# 6. QQ plot
fig = sm.qqplot(residuals, line='45', fit=True)
plt.title('QQ-plot des résidus')
plt.tight_layout()
fig.savefig(os.path.join(img_dir, 'qqplot_residus.png'), dpi=200)
plt.close('all')

print('Images generated:')
for name in sorted(os.listdir(img_dir)):
    print(' -', name)
