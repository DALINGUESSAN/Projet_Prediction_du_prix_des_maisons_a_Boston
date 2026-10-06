# Projet : Prédiction du prix des maisons à Boston

Ce projet porte sur la prédiction de la valeur médiane des logements à Boston à l’aide d’un modèle de régression linéaire.

## Objectif

- explorer le jeu de données Boston Housing ;
- nettoyer et préparer les variables ;
- entraîner un modèle de régression linéaire ;
- évaluer les performances avec $R^2$, MSE et RMSE ;
- analyser les hypothèses du modèle ;
- produire un rapport final avec les résultats et les visualisations.

## Fichiers du projet

- `Boston.csv` : jeu de données utilisé pour le projet.
- `Reg_lin_prix_maison_boston.ipynb` : notebook principal de traitement, modélisation et diagnostics.
- `rapport_boston.tex` : rapport LaTeX du projet.
- `rapport_boston.pdf` : version PDF du rapport compilé.
- `images/` : figures exportées pour le rapport.
- `.gitignore` : fichiers à ignorer dans le dépôt Git.

## Environnement

Le projet a été développé avec Python et les bibliothèques suivantes :

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels
- scipy

## Résultats clés

- $R^2$ train : 0.750886
- $R^2$ test : 0.668759
- MSE train : 21.641413
- MSE test : 24.291119
- RMSE train : 4.652033
- RMSE test : 4.928602

## Commandes pour publier sur GitHub

```bash
git init
git add .
git commit -m "Initial commit: Boston Housing regression project"
git branch -M main
git remote add origin https://github.com/DALINGUESSAN/prix-maison-boston.git
git push -u origin main
```

## Note

Le rapport LaTeX et les images sont prêts à être utilisés et intégrés dans un dépôt GitHub.
