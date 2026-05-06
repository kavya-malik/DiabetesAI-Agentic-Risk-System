"""
Diabetes Dataset EDA Analysis
Loads diabetes.csv and performs comprehensive exploratory data analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def load_diabetes_data():
    """Load diabetes dataset, create sample if not exists"""
    data_path = Path('data/raw/diabetes.csv')
    
    if data_path.exists():
        print(f"✓ Loading dataset from {data_path}")
        df = pd.read_csv(data_path)
    else:
        print(f"⚠️  File not found at {data_path}")
        print("Creating sample diabetes dataset...")
        
        # Create sample data matching typical diabetes datasets (UCI Diabetes dataset format)
        np.random.seed(42)
        n_samples = 768
        
        df = pd.DataFrame({
            'Pregnancies': np.random.randint(0, 17, n_samples),
            'Glucose': np.random.randint(0, 200, n_samples),
            'BloodPressure': np.random.randint(0, 130, n_samples),
            'SkinThickness': np.random.randint(0, 100, n_samples),
            'Insulin': np.random.randint(0, 900, n_samples),
            'BMI': np.random.uniform(18, 50, n_samples),
            'DiabetesPedigreeFunction': np.random.uniform(0, 2.5, n_samples),
            'Age': np.random.randint(21, 82, n_samples),
            'Outcome': np.random.randint(0, 2, n_samples)
        })
        
        # Introduce some invalid zeros (common in medical datasets)
        for col in ['Glucose', 'BloodPressure', 'BMI', 'SkinThickness', 'Insulin']:
            zero_indices = np.random.choice(df.index, size=int(len(df) * 0.05), replace=False)
            df.loc[zero_indices, col] = 0
        
        # Save to CSV
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"✓ Sample dataset created and saved to {data_path}\n")
    
    return df


def perform_eda(df):
    """Perform comprehensive EDA"""
    
    print("\n" + "="*80)
    print("📊 DIABETES DATASET - EXPLORATORY DATA ANALYSIS")
    print("="*80 + "\n")
    
    # 1. FIRST 5 ROWS
    print("1️⃣  FIRST 5 ROWS")
    print("-" * 80)
    print(df.head())
    print()
    
    # 2. DATASET INFO
    print("\n2️⃣  DATASET INFO")
    print("-" * 80)
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    print(f"\nData Types:")
    print(df.dtypes)
    print(f"\nInfo:")
    df.info()
    print()
    
    # 3. MISSING VALUES BEFORE CLEANING
    print("\n3️⃣  MISSING VALUES ANALYSIS (BEFORE CLEANING)")
    print("-" * 80)
    missing_before = df.isnull().sum()
    if missing_before.sum() == 0:
        print("✓ No missing values detected")
    else:
        print(missing_before[missing_before > 0])
    print()
    
    # 4. CHECK FOR INVALID ZEROS
    print("\n4️⃣  INVALID ZERO VALUES (BEFORE CLEANING)")
    print("-" * 80)
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    zero_counts = {}
    for col in zero_columns:
        if col in df.columns:
            zero_count = (df[col] == 0).sum()
            zero_pct = (zero_count / len(df)) * 100
            zero_counts[col] = (zero_count, zero_pct)
            print(f"  {col:20s}: {zero_count:4d} zeros ({zero_pct:5.1f}%)")
    print()
    
    # 5. REPLACE INVALID ZEROS WITH NaN
    print("\n5️⃣  CLEANING DATA - REPLACING INVALID ZEROS WITH NaN")
    print("-" * 80)
    
    df_cleaned = df.copy()
    
    for col in zero_columns:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].replace(0, np.nan)
    
    print("✓ Invalid zeros replaced with NaN\n")
    
    # 6. MISSING VALUES AFTER CLEANING
    print("6️⃣  MISSING VALUES ANALYSIS (AFTER CLEANING)")
    print("-" * 80)
    missing_after = df_cleaned.isnull().sum()
    missing_pct = (missing_after / len(df_cleaned)) * 100
    
    print("Columns with missing values:")
    for col in df_cleaned.columns:
        if missing_after[col] > 0:
            print(f"  {col:20s}: {missing_after[col]:4d} missing ({missing_pct[col]:5.1f}%)")
    print()
    
    # 7. STATISTICAL SUMMARY (BEFORE CLEANING)
    print("\n7️⃣  STATISTICAL SUMMARY (BEFORE CLEANING)")
    print("-" * 80)
    print(df.describe())
    print()
    
    # 8. STATISTICAL SUMMARY (AFTER CLEANING)
    print("\n8️⃣  STATISTICAL SUMMARY (AFTER CLEANING)")
    print("-" * 80)
    print(df_cleaned.describe())
    print()
    
    # 9. OUTCOME DISTRIBUTION
    if 'Outcome' in df.columns:
        print("\n9️⃣  OUTCOME DISTRIBUTION")
        print("-" * 80)
        outcome_counts = df['Outcome'].value_counts()
        print(outcome_counts)
        print(f"\nPercentage:")
        print(df['Outcome'].value_counts(normalize=True) * 100)
        print()
    
    # 10. COMPARISON BEFORE/AFTER
    print("\n🔟 COMPARISON: BEFORE vs AFTER CLEANING")
    print("-" * 80)
    print(f"{'Column':<20} {'Before Mean':<15} {'After Mean':<15} {'Change':<15}")
    print("-" * 65)
    
    for col in zero_columns:
        if col in df.columns:
            before_mean = df[col].mean()
            after_mean = df_cleaned[col].mean()
            # Calculate mean excluding NaN
            after_mean_valid = df_cleaned[col][df_cleaned[col].notna()].mean()
            change = after_mean_valid - before_mean if pd.notna(after_mean_valid) else 0
            
            print(f"{col:<20} {before_mean:<15.3f} {after_mean_valid:<15.3f} {change:+.3f}")
    
    print("\n" + "="*80)
    print("✅ EDA ANALYSIS COMPLETE")
    print("="*80 + "\n")
    
    return df, df_cleaned


def main():
    """Main execution"""
    
    # Load data
    df = load_diabetes_data()
    
    # Perform EDA
    df_original, df_cleaned = perform_eda(df)
    
    # Save cleaned dataset
    cleaned_path = Path('data/processed/diabetes_cleaned.csv')
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_csv(cleaned_path, index=False)
    print(f"✓ Cleaned dataset saved to {cleaned_path}\n")
    
    # Summary statistics
    print("\n📈 SUMMARY")
    print("-" * 80)
    print(f"Original dataset shape:  {df_original.shape}")
    print(f"Cleaned dataset shape:   {df_cleaned.shape}")
    print(f"Total missing values:    {df_cleaned.isnull().sum().sum()}")
    print(f"Missing data %:          {(df_cleaned.isnull().sum().sum() / (df_cleaned.shape[0] * df_cleaned.shape[1])) * 100:.2f}%")
    
    return df_original, df_cleaned


if __name__ == "__main__":
    df_orig, df_clean = main()
