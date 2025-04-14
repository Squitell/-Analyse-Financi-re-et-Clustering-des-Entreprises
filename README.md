Projet : Analyse Financière et Clustering des Entreprises

Contexte du projet

Ce projet vise à réaliser une analyse approfondie des données financières d'entreprises afin d'identifier des segments distincts en fonction de leurs performances financières. Il inclut les phases de nettoyage, d'analyse exploratoire (EDA), et d'application d'algorithmes de clustering (K-Means et PCA).

Objectifs

Nettoyer et préparer les données financières pour l'analyse.

Réaliser une analyse exploratoire pour comprendre les structures et tendances.

Calculer des ratios financiers pertinents : ROA, ROE, marge nette, endettement, liquidité, rotation des actifs.

Identifier des profils d'entreprises à l'aide de techniques de clustering.

Structure du projet

projet-analyse-financiere/
│
├── data/
│   ├── raw/
│   │   └── Financials.csv
│   └── processed/
│       ├── Financials_cleaned.csv
│       └── Financials_clustered.csv
│
├── notebooks/
│   ├── 01_Cleaning.ipynb
│   ├── 02_EDA.ipynb
│   └── 03_Clustering.ipynb
│
├── scripts/
│   ├── cleaning.py
│   ├── eda.py
│   └── clustering.py
│
├── reports/
│   ├── Rapport_final.pdf
│   └── Rapport_final.tex
│
├── photos/
│   └── (images générées pour le rapport)
│
└── README.md

Étapes réalisées

1. Nettoyage des données

Normalisation et standardisation des données

Gestion des valeurs manquantes et des doublons

Sauvegarde du jeu de données nettoyé : Financials_cleaned.csv

2. Analyse Exploratoire (EDA)

Distribution des variables catégorielles et numériques

Visualisations : histogrammes, scatter plots, heatmaps, boxplots

Calcul des ratios financiers clés (ROA, ROE, etc.)

3. Clustering

Sélection des variables pertinentes (ratios financiers)

Standardisation et réduction dimensionnelle (PCA)

Application de l'algorithme K-Means

Analyse et interprétation des clusters

Technologies utilisées

Python (pandas, numpy, matplotlib, scikit-learn)

Jupyter Notebook

LaTeX pour le rapport final

Résultats principaux

Deux profils distincts d'entreprises identifiés via clustering

Insights clairs sur la rentabilité, l'endettement, la liquidité et l'efficacité opérationnelle

Recommandations stratégiques pour améliorer les performances financières

Perspectives

Collecter plus de données pour affiner l'analyse

Utiliser des méthodes alternatives de clustering et de machine learning

Intégrer des données externes pour enrichir l'analyse sectorielle

Auteurs

OUZAHRA Hamza

TERRAF Ahmed

TOUILE Samia

ELHATHOUT Mohammed

JADLAOUI Marwen