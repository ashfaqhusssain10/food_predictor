# Create: food_predictor/scripts/prepare_df5_for_tonic.py
import pandas as pd
from food_predictor.data.data_loader import DataLoader
from food_predictor.data.df5_analyzer import DF5Analyzer
from food_predictor.utils.quantity_utils import extract_quantity_value, extract_unit

def prepare_df5_for_tonic():
    """Prepare DF5 data with business logic for Tonic AI"""
    
    # Load data
    data_loader = DataLoader()
    df5_raw = data_loader.load_df5()
    
    # Analyze
    analyzer = DF5Analyzer(df5_raw)
    df5_analyzed = analyzer.analyze_data_quality()
    constraints = analyzer.identify_constraints()
    
    # Clean and prepare
    df5_prep = df5_analyzed.copy()
    
    # 1. Handle missing values
    df5_prep['Veg_Count'] = df5_prep['Veg_Count'].fillna(
        df5_prep['total_guest_count'] * 0.4
    )
    
    # 2. Add derived columns for constraints
    df5_prep['veg_ratio'] = df5_prep['Veg_Count'] / df5_prep['total_guest_count']
    df5_prep['total_quantity_calc'] = df5_prep['quantity_numeric'] * df5_prep['total_guest_count']
    
    # 3. Add validation flags
    df5_prep['is_valid_quantity'] = df5_prep['quantity_numeric'] > 0
    df5_prep['is_valid_guest_count'] = df5_prep['total_guest_count'] > 0
    df5_prep['is_valid_veg_ratio'] = (df5_prep['veg_ratio'] >= 0) & (df5_prep['veg_ratio'] <= 1)
    
    # 4. Remove invalid records
    valid_mask = (
        df5_prep['is_valid_quantity'] & 
        df5_prep['is_valid_guest_count'] & 
        df5_prep['is_valid_veg_ratio']
    )
    df5_clean = df5_prep[valid_mask].copy()
    
    print(f"\nData cleaned: {len(df5_raw)} -> {len(df5_clean)} rows")
    
    # 5. Save prepared data
    prepared_path = data_loader.save_synthetic_data(df5_clean, 'df5_prepared_for_tonic.xlsx')
    
    return df5_clean, constraints, prepared_path

if __name__ == "__main__":
    df5_clean, constraints, path = prepare_df5_for_tonic()
    print(f"Prepared data saved to: {path}")