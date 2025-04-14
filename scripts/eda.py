import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# ============================
# 1. Charger le dataset nettoyé
# ============================
data_path = os.path.join('data', 'processed', 'Financials_cleaned.csv')
df = pd.read_csv(data_path)

# Juste pour vérifier
print("DataFrame loaded with shape:", df.shape)
print(df.info())

# ============================
# 2. Variables catégorielles
# ============================
cat_cols = ['segment', 'country', 'product', 'discount_band']

for col in cat_cols:
    if col in df.columns:
        plt.figure(figsize=(8, 5))
        df[col].value_counts(dropna=False).plot(
            kind='bar', 
            edgecolor='k'
        )
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# ============================
# 3. Distribution des variables numériques
# ============================
numeric_columns = [
    'units_sold', 
    'manufacturing_price', 
    'sale_price', 
    'gross_sales',
    'discounts', 
    'sales', 
    'cogs', 
    'profit'
]

for col in numeric_columns:
    if col in df.columns:
        plt.figure(figsize=(8, 5))
        plt.hist(df[col].dropna(), bins=30, edgecolor='k')
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

# ============================
# 4. Bar chart : profit moyen par segment
# ============================
if 'segment' in df.columns and 'profit' in df.columns:
    mean_profit_by_segment = df.groupby('segment')['profit'].mean().sort_values()
    plt.figure(figsize=(8, 5))
    mean_profit_by_segment.plot(kind='bar', edgecolor='k')
    plt.title('Average Profit by Segment')
    plt.xlabel('Segment')
    plt.ylabel('Average Profit')
    plt.tight_layout()
    plt.show()

# ============================
# 5. Bar chart : total sales par pays (top 10 pays)
# ============================
if 'country' in df.columns and 'sales' in df.columns:
    total_sales_by_country = df.groupby('country')['sales'].sum().sort_values(ascending=False)
    top_10 = total_sales_by_country.head(10)
    plt.figure(figsize=(8, 5))
    top_10.plot(kind='bar', edgecolor='k')
    plt.title('Top 10 Countries by Total Sales')
    plt.xlabel('Country')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ============================
# 6. Évolution des ventes / profit au fil du temps (bar charts)
# ============================
if 'year' in df.columns and 'sales' in df.columns:
    sales_by_year = df.groupby('year')['sales'].sum().sort_index()
    plt.figure(figsize=(8, 5))
    sales_by_year.plot(kind='bar', edgecolor='k')
    plt.title('Total Sales by Year')
    plt.xlabel('Year')
    plt.ylabel('Sales')
    plt.tight_layout()
    plt.show()

if 'year' in df.columns and 'profit' in df.columns:
    profit_by_year = df.groupby('year')['profit'].sum().sort_index()
    plt.figure(figsize=(8, 5))
    profit_by_year.plot(kind='bar', edgecolor='k')
    plt.title('Total Profit by Year')
    plt.xlabel('Year')
    plt.ylabel('Profit')
    plt.tight_layout()
    plt.show()

# ============================
# 7. Scatter plots
# ============================
if 'sale_price' in df.columns and 'units_sold' in df.columns:
    plt.figure(figsize=(8, 5))
    plt.scatter(df['sale_price'], df['units_sold'])
    plt.title('Sale Price vs. Units Sold')
    plt.xlabel('Sale Price')
    plt.ylabel('Units Sold')
    plt.tight_layout()
    plt.show()

if 'cogs' in df.columns and 'profit' in df.columns:
    plt.figure(figsize=(8, 5))
    plt.scatter(df['cogs'], df['profit'])
    plt.title('COGS vs. Profit')
    plt.xlabel('COGS')
    plt.ylabel('Profit')
    plt.tight_layout()
    plt.show()

# ============================
# 8. Boxplot : profit par segment
# ============================
if 'segment' in df.columns and 'profit' in df.columns:
    plt.figure(figsize=(8, 5))
    df.boxplot(column='profit', by='segment', vert=False)
    plt.title('Profit by Segment')
    plt.suptitle('')
    plt.xlabel('Profit')
    plt.ylabel('Segment')
    plt.tight_layout()
    plt.show()

# ============================
# 9. Matrice de corrélation
# ============================
corr_matrix = df[numeric_columns].corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

plt.figure(figsize=(10, 8))
im = plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar(im, label='Correlation')
plt.title('Correlation Matrix with Values')
plt.xticks(ticks=range(len(corr_matrix.columns)), labels=corr_matrix.columns, rotation=90)
plt.yticks(ticks=range(len(corr_matrix.index)), labels=corr_matrix.index)

threshold = np.nanmax(corr_matrix.values) / 2.0
for i in range(corr_matrix.shape[0]):
    for j in range(corr_matrix.shape[1]):
        value = corr_matrix.iloc[i, j]
        color = "white" if value > threshold else "black"
        plt.text(j, i, f"{value:.2f}", ha='center', va='center', color=color)
plt.tight_layout()
plt.show()

# ============================
# 10. Pie chart : répartition par segment
# ============================
if 'segment' in df.columns:
    segment_counts = df['segment'].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(segment_counts, labels=segment_counts.index, autopct='%1.1f%%')
    plt.title('Distribution by Segment')
    plt.tight_layout()
    plt.show()


# =====================================================================
# ===================== PLOTS AJOUTÉS SELON TA DEMANDE ================
# =====================================================================

# ============ A. Sales et Profit sur la même courbe (over time) ============
# On va créer un line chart combiné (monthly data si possible).
if 'year' in df.columns and 'month_number' in df.columns and 'sales' in df.columns and 'profit' in df.columns:
    # Agrégation par (year, month_number)
    monthly_data = df.groupby(['year', 'month_number'])[['sales', 'profit']].sum().reset_index()
    # Trie par année, puis par mois
    monthly_data.sort_values(['year', 'month_number'], inplace=True)
    # On crée une string "AAAA-MM" pour l'axe x
    monthly_data['year_month'] = monthly_data['year'].astype(str) + '-' + monthly_data['month_number'].astype(str)

    plt.figure(figsize=(10, 5))
    plt.plot(monthly_data['year_month'], monthly_data['sales'], marker='o', label='Sales')
    plt.plot(monthly_data['year_month'], monthly_data['profit'], marker='x', label='Profit')
    plt.title('Sales and Profit Over Time (Monthly)')
    plt.xlabel('Year-Month')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============ B. Heatmap of Units Sold : Products vs. Segments ============
if 'product' in df.columns and 'segment' in df.columns and 'units_sold' in df.columns:
    pivot_units_ps = df.pivot_table(values='units_sold', index='product', columns='segment', aggfunc='sum')
    pivot_units_ps = pivot_units_ps.fillna(0)  # Remplace les NaN par 0

    plt.figure(figsize=(8, 6))
    im = plt.imshow(pivot_units_ps, aspect='auto', cmap='viridis')
    plt.colorbar(im, label='Units Sold')
    plt.title('Heatmap of Units Sold (Product vs. Segment)')
    plt.xlabel('Segment')
    plt.ylabel('Product')
    plt.xticks(ticks=range(len(pivot_units_ps.columns)), labels=pivot_units_ps.columns, rotation=45)
    plt.yticks(ticks=range(len(pivot_units_ps.index)), labels=pivot_units_ps.index)

    # Calcul du seuil pour choisir la couleur du texte
    threshold = pivot_units_ps.values.max() / 2.0
    for i in range(pivot_units_ps.shape[0]):
        for j in range(pivot_units_ps.shape[1]):
            value = pivot_units_ps.iloc[i, j]
            color = "white" if value > threshold else "black"
            plt.text(j, i, f'{value:.0f}', ha='center', va='center', color=color)
    plt.tight_layout()
    plt.show()



# ============ C. Heatmap of Units Sold : Products vs. Discount Bands ============
if 'product' in df.columns and 'discount_band' in df.columns and 'units_sold' in df.columns:
    pivot_units_pd = df.pivot_table(values='units_sold', index='product', columns='discount_band', aggfunc='sum')
    pivot_units_pd = pivot_units_pd.fillna(0)

    plt.figure(figsize=(8, 6))
    im = plt.imshow(pivot_units_pd, aspect='auto', cmap='viridis')
    plt.colorbar(im, label='Units Sold')
    plt.title('Heatmap of Units Sold (Product vs. Discount Band)')
    plt.xlabel('Discount Band')
    plt.ylabel('Product')
    plt.xticks(ticks=range(len(pivot_units_pd.columns)), labels=pivot_units_pd.columns, rotation=45)
    plt.yticks(ticks=range(len(pivot_units_pd.index)), labels=pivot_units_pd.index)

    threshold = pivot_units_pd.values.max() / 2.0
    for i in range(pivot_units_pd.shape[0]):
        for j in range(pivot_units_pd.shape[1]):
            value = pivot_units_pd.iloc[i, j]
            color = "white" if value > threshold else "black"
            plt.text(j, i, f'{value:.0f}', ha='center', va='center', color=color)
    plt.tight_layout()
    plt.show()



# ============ D. Heatmap of Profit : Segments vs. Discount Bands ============
if 'segment' in df.columns and 'discount_band' in df.columns and 'profit' in df.columns:
    pivot_profit_sd = df.pivot_table(values='profit', index='segment', columns='discount_band', aggfunc='sum')
    pivot_profit_sd = pivot_profit_sd.fillna(0)

    plt.figure(figsize=(8, 6))
    im = plt.imshow(pivot_profit_sd, aspect='auto', cmap='plasma')
    plt.colorbar(im, label='Profit')
    plt.title('Heatmap of Profit (Segment vs. Discount Band)')
    plt.xlabel('Discount Band')
    plt.ylabel('Segment')
    plt.xticks(ticks=range(len(pivot_profit_sd.columns)), labels=pivot_profit_sd.columns, rotation=45)
    plt.yticks(ticks=range(len(pivot_profit_sd.index)), labels=pivot_profit_sd.index)

    threshold = pivot_profit_sd.values.max() / 2.0
    for i in range(pivot_profit_sd.shape[0]):
        for j in range(pivot_profit_sd.shape[1]):
            value = pivot_profit_sd.iloc[i, j]
            color = "white" if value > threshold else "black"
            plt.text(j, i, f'{value:.0f}', ha='center', va='center', color=color)
    plt.tight_layout()
    plt.show()



# ============ E. Heatmap of Profit : Products vs. Discount Bands ============
if 'product' in df.columns and 'discount_band' in df.columns and 'profit' in df.columns:
    pivot_profit_pd = df.pivot_table(values='profit', index='product', columns='discount_band', aggfunc='sum')
    pivot_profit_pd = pivot_profit_pd.fillna(0)

    plt.figure(figsize=(8, 6))
    im = plt.imshow(pivot_profit_pd, aspect='auto', cmap='plasma')
    plt.colorbar(im, label='Profit')
    plt.title('Heatmap of Profit (Product vs. Discount Band)')
    plt.xlabel('Discount Band')
    plt.ylabel('Product')
    plt.xticks(ticks=range(len(pivot_profit_pd.columns)), labels=pivot_profit_pd.columns, rotation=45)
    plt.yticks(ticks=range(len(pivot_profit_pd.index)), labels=pivot_profit_pd.index)

    threshold = pivot_profit_pd.values.max() / 2.0
    for i in range(pivot_profit_pd.shape[0]):
        for j in range(pivot_profit_pd.shape[1]):
            value = pivot_profit_pd.iloc[i, j]
            color = "white" if value > threshold else "black"
            plt.text(j, i, f'{value:.0f}', ha='center', va='center', color=color)
    plt.tight_layout()
    plt.show()


    # Vérification des colonnes nécessaires. Si elles ne sont pas présentes, nous les créons avec des hypothèses.
# On suppose ici, par simplicité, que :
#   - Le "résultat net" est équivalent à la colonne 'profit' déjà présente.
#   - Le "total des actifs" est estimé comme 10 fois le chiffre d'affaires net ('sales'). (Hypothèse simplifiée)
#   - Les "capitaux propres" représentent 20% du total des actifs.
#   - Les "dettes" représentent les 80% restants.
#   - Les "actifs circulants" représentent 50% du total des actifs.
#   - Les "passifs circulants" représentent 30% du total des actifs.

# Si les colonnes n'existent pas dans le DataFrame, nous les ajoutons.

if 'net_income' not in df.columns:
    df['net_income'] = df['profit']  # On utilise 'profit' comme résultat net

if 'total_assets' not in df.columns:
    df['total_assets'] = df['sales'] * 10  # Hypothèse simplifiée : 10 fois le chiffre d'affaires net

if 'equity' not in df.columns:
    df['equity'] = df['total_assets'] * 0.2

if 'liabilities' not in df.columns:
    df['liabilities'] = df['total_assets'] * 0.8

if 'current_assets' not in df.columns:
    df['current_assets'] = df['total_assets'] * 0.5

if 'current_liabilities' not in df.columns:
    df['current_liabilities'] = df['total_assets'] * 0.3

# Maintenant, calculons les principaux ratios financiers :

# 1. Rentabilité : ROA, ROE, Marge nette
df['roa'] = df['net_income'] / df['total_assets']
df['roe'] = df['net_income'] / df['equity']
df['marge_nette'] = df['net_income'] / df['sales']

# 2. Endettement
df['endettement'] = df['liabilities'] / df['equity']

# 3. Liquidité
df['liquidite'] = df['current_assets'] / df['current_liabilities']

# 4. Efficacité : Rotation des actifs
df['rotation_actifs'] = df['sales'] / df['total_assets']

# Affichage rapide des nouveaux ratios pour vérification

new_ratio_cols = ['roa', 'roe', 'marge_nette', 'endettement', 'liquidite', 'rotation_actifs']

for col in new_ratio_cols:
    if col in df.columns:
        data = df[col].dropna()
        plt.figure(figsize=(8, 5))
        rng = data.max() - data.min()
        # Si le nombre de valeurs uniques est inférieur à 5 ou la plage est très faible,
        # on affiche un diagramme à barres pour mieux représenter la distribution.
        if data.nunique() < 5 or rng < 1e-4:
            counts = data.value_counts().sort_index()
            plt.bar(counts.index.astype(str), counts.values, color='skyblue', edgecolor='k')
            plt.title(f'Bar Plot de {col} (données quasi-constantes)')
            plt.xlabel(col)
            plt.ylabel("Nombre d'observations")
        else:
            plt.hist(data, bins='auto', edgecolor='k')
            plt.title(f'Histogramme de {col}')
            plt.xlabel(col)
            plt.ylabel('Fréquence')
        plt.tight_layout()
        plt.show()
