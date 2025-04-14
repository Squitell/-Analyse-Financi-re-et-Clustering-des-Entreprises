import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os

# ============================================
# 1. Auxiliary cleaning functions
# ============================================

def clean_column_names(df):
    """
    Remove leading/trailing spaces from column names,
    convert to lowercase, and replace spaces with underscores.
    """
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    return df

def convert_parentheses(value):
    """
    Convert numbers enclosed in parentheses to negative numbers.
    
    This function uses a regular expression to capture numbers inside parentheses,
    handling possible extra spaces. For example, "( 4533.75 )" becomes "-4533.75".
    """
    if isinstance(value, str):
        pattern = r"^\(\s*([0-9.,-]+)\s*\)$"
        match = re.match(pattern, value.strip())
        if match:
            inner = match.group(1)
            return "-" + inner
    return value

def clean_numeric_value(x):
    """
    Clean numeric values by:
      - Converting values in parentheses to negative (using the above function).
      - Removing symbols like '$' and ','.
      - Converting empty strings or '-' to NaN.
      - Finally converting the result to float.
    """
    if isinstance(x, str):
        x = convert_parentheses(x)
        x = x.replace('$', '').replace(',', '')
        if x.strip() == '' or x.strip() == '-':
            return np.nan
        try:
            return float(x)
        except ValueError:
            return np.nan
    elif pd.isna(x):
        return np.nan
    else:
        return x

def clean_numeric_columns(df, cols):
    """
    Apply the cleaning function to specified numeric columns.
    """
    for col in cols:
        df[col] = df[col].apply(clean_numeric_value)
    return df

def remove_duplicates(df):
    """
    Remove duplicate rows.
    """
    return df.drop_duplicates()

def clean_string_columns(df, cols):
    """
    Remove extra spaces in string-type columns.
    """
    for col in cols:
        df[col] = df[col].astype(str).str.strip()
    return df

def standardize_categorical(df, cols):
    """
    Standardize categorical values: convert to lowercase and trim spaces.
    """
    for col in cols:
        df[col] = df[col].astype(str).str.lower().str.strip()
    return df

def fill_missing_numeric(df, method='median'):
    """
    Fill missing values in numeric columns using the specified method (median or mean).
    Avoids inplace modification warnings.
    """
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        if method == 'median':
            df[col] = df[col].fillna(df[col].median())
        elif method == 'mean':
            df[col] = df[col].fillna(df[col].mean())
    return df


def convert_date_column(df, date_col, date_format='%Y-%m-%d'):
    """
    Convert a column to datetime format.
    """
    df[date_col] = pd.to_datetime(df[date_col], format=date_format, errors='coerce')
    return df

# ============================================
# 2. Full cleaning pipeline
# ============================================

if __name__ == "__main__":
    # Load the raw dataset
    df = pd.read_csv('data/raw/Financials.csv')
    
    # --- Step 1: Clean column names ---
    df = clean_column_names(df)
    print("Columns after cleaning:", df.columns.tolist())
    
    # --- Step 2: Remove duplicates ---
    df = remove_duplicates(df)
    
    # --- Step 3: Clean numeric columns ---
    numeric_cols = [
        'units_sold', 
        'manufacturing_price', 
        'sale_price', 
        'gross_sales', 
        'discounts', 
        'sales', 
        'cogs', 
        'profit'
    ]
    df = clean_numeric_columns(df, numeric_cols)
    
    # --- Step 4: Clean string columns ---
    string_cols = ['segment', 'country', 'product', 'discount_band']
    df = clean_string_columns(df, string_cols)
    
    # --- Step 5: Convert the 'date' column to datetime ---
    if 'date' in df.columns:
        df = convert_date_column(df, 'date', date_format='%Y-%m-%d')
    
    # --- Step 6: Standardize categorical variables ---
    categorical_cols = ['discount_band']  # Adjust as needed
    df = standardize_categorical(df, categorical_cols)
    
    # --- Step 7: Fill missing values for numeric columns ---
    df = fill_missing_numeric(df, method='median')
    
    # (Optional) --- Output DataFrame info and summary statistics
    print(df.info())
    print(df.describe())
    
    # Create the output directory if it does not exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Save the cleaned dataset
    df.to_csv('data/processed/Financials_cleaned.csv', index=False)
    
    # (Optional) --- Visual inspection: simple boxplot for 'sale_price'
    plt.boxplot(df['sale_price'].dropna())
    plt.title('Boxplot of Sale Price')
    plt.show()
