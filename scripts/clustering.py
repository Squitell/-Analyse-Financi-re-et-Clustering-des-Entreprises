import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def main():
    # 1. Charger le dataset nettoyé
    data_path = os.path.join('data', 'processed', 'Financials_cleaned.csv')
    df = pd.read_csv(data_path)
    print("Data loaded:", df.shape)
    
    # 2. Vérifier (et créer si nécessaire) les colonnes pour les ratios financiers
    # On considère que 'profit' est le résultat net si 'net_income' n'existe pas.
    if 'net_income' not in df.columns:
        df['net_income'] = df['profit']
    
    # Estimation simplifiée du total des actifs : 10 fois le chiffre d’affaires net
    if 'total_assets' not in df.columns:
        df['total_assets'] = df['sales'] * 10
    
    # Estimation des capitaux propres comme 20% du total des actifs
    if 'equity' not in df.columns:
        df['equity'] = df['total_assets'] * 0.2
    
    # Estimation des dettes comme 80% du total des actifs
    if 'liabilities' not in df.columns:
        df['liabilities'] = df['total_assets'] * 0.8
    
    # Estimation des actifs circulants et passifs circulants (pour le ratio de liquidité)
    if 'current_assets' not in df.columns:
        df['current_assets'] = df['total_assets'] * 0.5
    if 'current_liabilities' not in df.columns:
        df['current_liabilities'] = df['total_assets'] * 0.3

    # 3. Calcul des ratios financiers
    df['roa'] = df['net_income'] / df['total_assets']
    df['roe'] = df['net_income'] / df['equity']
    df['marge_nette'] = df['net_income'] / df['sales']
    df['endettement'] = df['liabilities'] / df['equity']
    df['liquidite'] = df['current_assets'] / df['current_liabilities']
    df['rotation_actifs'] = df['sales'] / df['total_assets']
    
    new_ratio_cols = ['roa', 'roe', 'marge_nette', 'endettement', 'liquidite', 'rotation_actifs']
    print("New ratios (first 5 rows):")
    print(df[new_ratio_cols].head())
    
    # 4. Clustering basé sur les nouveaux ratios
    features = new_ratio_cols
    X = df[features].copy()

    # Standardisation des variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Réduction de dimension avec PCA pour faciliter la visualisation
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print("Variance expliquée par les 2 composantes PCA:", pca.explained_variance_ratio_)
    
    # 5. Détermination du nombre optimal de clusters avec la méthode du coude
    K_candidates = range(1, 11)
    inertias = []
    for k_ in K_candidates:
        km_test = KMeans(n_clusters=k_, random_state=42)
        km_test.fit(X_pca)
        inertias.append(km_test.inertia_)
    
    plt.figure(figsize=(6, 4))
    plt.plot(K_candidates, inertias, marker='o')
    plt.title('Elbow Method (Inertia vs k)')
    plt.xlabel('k')
    plt.ylabel('Inertie')
    plt.xticks(K_candidates)
    plt.tight_layout()
    plt.show()
    
    # Méthode naïve : calcul du "drop" maximum
    drops = []
    for i in range(1, len(inertias)):
        drop = inertias[i-1] - inertias[i]
        drops.append(drop)
    best_drop_index = np.argmax(drops)
    best_k = best_drop_index + 1 + 1   # +1 pour convertir l'indice en k, +1 pour le passage k=1 -> k=2
    print(f"Elbow suggests k = {best_k}")
    
    # 6. Application finale de K-Means avec le meilleur k
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    kmeans.fit(X_pca)
    labels = kmeans.labels_
    
    plt.figure(figsize=(7, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='rainbow')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title(f'K-Means Clustering (k={best_k}) sur PCA (2D)')
    plt.colorbar(label='Cluster')
    plt.tight_layout()
    plt.show()
    
    # 7. Analyse des clusters
    df['Cluster'] = labels
    cluster_summary = df.groupby('Cluster')[features].mean()
    print("\nMean of each ratio by cluster:")
    print(cluster_summary)
    
    # 8. Sauvegarder le DataFrame enrichi
    output_path = os.path.join('data', 'processed', 'Financials_clustered_new.csv')
    df.to_csv(output_path, index=False)
    print(f"\nFichier '{output_path}' enregistré avec la colonne 'Cluster'.")

if __name__ == "__main__":
    main()
