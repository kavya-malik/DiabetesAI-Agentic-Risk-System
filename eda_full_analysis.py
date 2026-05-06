#!/usr/bin/env python3
"""
DIABETES DATASET EDA WITH INVALID ZERO VALUE REPLACEMENT
=========================================================

This script:
1. Loads diabetes dataset from data/raw/diabetes.csv
2. Creates sample data if file doesn't exist
3. Performs comprehensive EDA analysis
4. Identifies and replaces invalid zeros (0 values in medical measurements)
5. Saves cleaned dataset to data/processed/diabetes_cleaned.csv

HOW TO RUN:
-----------
1. Install dependencies:
   pip install pandas numpy

2. Run script:
   python3 eda_full_analysis.py

WHAT IT DOES:
-------------
✓ Loads data from CSV
✓ Shows first 5 rows
✓ Displays dataset info (shape, dtypes, memory)
✓ Lists missing values
✓ Identifies invalid zeros in medical columns
✓ Replaces invalid zeros with NaN
✓ Shows before/after statistics
✓ Saves cleaned data to CSV
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys


class DiabetesEDA:
    """Diabetes Dataset EDA Analysis"""
    
    def __init__(self):
        self.df_original = None
        self.df_cleaned = None
        self.invalid_zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    def load_or_create_data(self):
        """Load diabetes dataset or create sample"""
        data_path = Path('data/raw/diabetes.csv')
        
        if data_path.exists():
            print(f"✓ Loading dataset from {data_path}")
            self.df_original = pd.read_csv(data_path)
        else:
            print(f"⚠️  File not found at {data_path}")
            print("Creating sample diabetes dataset (UCI Diabetes Dataset format)...\n")
            
            np.random.seed(42)
            n_samples = 768
            
            self.df_original = pd.DataFrame({
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
            
            # Introduce some invalid zeros (5% of medical measurement columns)
            for col in self.invalid_zero_columns:
                if col != 'BMI':  # BMI is float, handle differently
                    zero_indices = np.random.choice(
                        self.df_original.index, 
                        size=int(len(self.df_original) * 0.05), 
                        replace=False
                    )
                    self.df_original.loc[zero_indices, col] = 0
            
            # Save sample data
            data_path.parent.mkdir(parents=True, exist_ok=True)
            self.df_original.to_csv(data_path, index=False)
            print(f"✓ Sample dataset created: {data_path}\n")
    
    def section_header(self, title, number):
        """Print section header"""
        print(f"\n{'='*90}")
        print(f"{number}️⃣  {title}")
        print(f"{'='*90}\n")
    
    def analyze_first_rows(self):
        """Section 1: Show first 5 rows"""
        self.section_header("FIRST 5 ROWS", "1")
        print(self.df_original.head().to_string())
    
    def analyze_dataset_info(self):
        """Section 2: Dataset Info"""
        self.section_header("DATASET INFORMATION", "2")
        print(f"Shape: {self.df_original.shape[0]} rows × {self.df_original.shape[1]} columns")
        print(f"Memory: {self.df_original.memory_usage(deep=True).sum() / 1024:.2f} KB\n")
        
        print("DATA TYPES:")
        print(self.df_original.dtypes.to_string())
        print("\n" + "-"*90)
        print("DETAILED INFO:")
        print(self.df_original.info())
    
    def analyze_missing_before(self):
        """Section 3: Missing values before cleaning"""
        self.section_header("MISSING VALUES (BEFORE CLEANING)", "3")
        
        missing = self.df_original.isnull().sum()
        missing_pct = (missing / len(self.df_original)) * 100
        
        print(f"Total missing values: {missing.sum()}")
        
        if missing.sum() == 0:
            print("✓ No missing values detected in original data\n")
        else:
            print("\nMissing values by column:")
            for col in self.df_original.columns:
                if missing[col] > 0:
                    print(f"  {col:25s}: {missing[col]:4d} ({missing_pct[col]:5.1f}%)")
    
    def analyze_invalid_zeros(self):
        """Section 4: Identify invalid zeros"""
        self.section_header("INVALID ZERO VALUES (BEFORE CLEANING)", "4")
        
        print("Medical measurements should not be zero. Found:\n")
        print(f"{'Column':<25} {'Count':<10} {'Percentage':<12}")
        print("-" * 50)
        
        self.zero_counts = {}
        for col in self.invalid_zero_columns:
            if col in self.df_original.columns:
                zero_count = (self.df_original[col] == 0).sum()
                zero_pct = (zero_count / len(self.df_original)) * 100
                self.zero_counts[col] = (zero_count, zero_pct)
                
                marker = "⚠️ " if zero_count > 0 else "✓"
                print(f"{marker} {col:<23} {zero_count:<10} {zero_pct:>6.1f}%")
    
    def clean_data(self):
        """Section 5: Replace invalid zeros with NaN"""
        self.section_header("CLEANING DATA - REPLACING INVALID ZEROS WITH NaN", "5")
        
        self.df_cleaned = self.df_original.copy()
        
        print("Replacing zero values with NaN in:")
        for col in self.invalid_zero_columns:
            if col in self.df_cleaned.columns:
                zeros_replaced = (self.df_cleaned[col] == 0).sum()
                self.df_cleaned[col] = self.df_cleaned[col].replace(0, np.nan)
                if zeros_replaced > 0:
                    print(f"  ✓ {col:25s}: {zeros_replaced} zeros → NaN")
        
        print("\n✓ Data cleaning complete!")
    
    def analyze_missing_after(self):
        """Section 6: Missing values after cleaning"""
        self.section_header("MISSING VALUES (AFTER CLEANING)", "6")
        
        missing = self.df_cleaned.isnull().sum()
        missing_pct = (missing / len(self.df_cleaned)) * 100
        
        total_missing = missing.sum()
        total_cells = self.df_cleaned.shape[0] * self.df_cleaned.shape[1]
        total_pct = (total_missing / total_cells) * 100
        
        print(f"Total missing values: {total_missing} ({total_pct:.2f}% of all data)\n")
        
        if total_missing > 0:
            print("Missing values by column:")
            print(f"{'Column':<25} {'Count':<10} {'Percentage':<12}")
            print("-" * 50)
            for col in self.df_cleaned.columns:
                if missing[col] > 0:
                    print(f"  {col:<23} {missing[col]:<10} {missing_pct[col]:>6.1f}%")
    
    def analyze_statistics_before(self):
        """Section 7: Statistics before cleaning"""
        self.section_header("STATISTICAL SUMMARY (BEFORE CLEANING)", "7")
        print(self.df_original.describe().to_string())
    
    def analyze_statistics_after(self):
        """Section 8: Statistics after cleaning"""
        self.section_header("STATISTICAL SUMMARY (AFTER CLEANING)", "8")
        print(self.df_cleaned.describe().to_string())
    
    def analyze_outcome(self):
        """Section 9: Outcome distribution"""
        self.section_header("OUTCOME DISTRIBUTION", "9")
        
        if 'Outcome' in self.df_original.columns:
            print("Outcome value counts:")
            print(self.df_original['Outcome'].value_counts().sort_index().to_string())
            
            print("\nOutcome percentages:")
            pct = self.df_original['Outcome'].value_counts(normalize=True).sort_index() * 100
            print(pct.to_string())
    
    def compare_before_after(self):
        """Section 10: Comparison before/after"""
        self.section_header("IMPACT COMPARISON: BEFORE vs AFTER CLEANING", "10")
        
        print(f"{'Column':<20} {'Before Mean':<18} {'After Mean':<18} {'Change':<15}")
        print("-" * 72)
        
        for col in self.invalid_zero_columns:
            if col in self.df_original.columns:
                before_mean = self.df_original[col].mean()
                
                # After mean (excluding NaN)
                after_mean = self.df_cleaned[col][self.df_cleaned[col].notna()].mean()
                
                change = after_mean - before_mean if pd.notna(after_mean) else 0
                
                print(f"{col:<20} {before_mean:<18.4f} {after_mean:<18.4f} {change:+.4f}")
    
    def save_cleaned_data(self):
        """Save cleaned dataset"""
        cleaned_path = Path('data/processed/diabetes_cleaned.csv')
        cleaned_path.parent.mkdir(parents=True, exist_ok=True)
        self.df_cleaned.to_csv(cleaned_path, index=False)
        print(f"\n✓ Cleaned dataset saved to: {cleaned_path}")
    
    def final_summary(self):
        """Print final summary"""
        print(f"\n\n{'='*90}")
        print("📊 FINAL SUMMARY")
        print(f"{'='*90}\n")
        
        print(f"Original Dataset:")
        print(f"  • Rows: {self.df_original.shape[0]}")
        print(f"  • Columns: {self.df_original.shape[1]}")
        print(f"  • Missing Values: {self.df_original.isnull().sum().sum()}")
        
        print(f"\nCleaned Dataset:")
        print(f"  • Rows: {self.df_cleaned.shape[0]}")
        print(f"  • Columns: {self.df_cleaned.shape[1]}")
        print(f"  • Missing Values: {self.df_cleaned.isnull().sum().sum()}")
        
        print(f"\nInvalid Zeros Replaced:")
        total_zeros = sum(count for count, _ in self.zero_counts.values())
        print(f"  • Total zeros converted to NaN: {total_zeros}")
        
        print(f"\n✅ EDA ANALYSIS COMPLETE!\n")
    
    def run_full_analysis(self):
        """Run complete analysis"""
        print("\n" + "="*90)
        print("🏥 DIABETES DATASET EXPLORATORY DATA ANALYSIS (EDA)")
        print("="*90)
        
        # Execute analysis steps
        self.load_or_create_data()
        self.analyze_first_rows()
        self.analyze_dataset_info()
        self.analyze_missing_before()
        self.analyze_invalid_zeros()
        self.clean_data()
        self.analyze_missing_after()
        self.analyze_statistics_before()
        self.analyze_statistics_after()
        self.analyze_outcome()
        self.compare_before_after()
        self.save_cleaned_data()
        self.final_summary()
        
        return self.df_original, self.df_cleaned


def main():
    """Main entry point"""
    eda = DiabetesEDA()
    df_orig, df_clean = eda.run_full_analysis()
    return df_orig, df_clean


if __name__ == "__main__":
    main()
