# Create: food_predictor/scripts/generate_df5_synthetic.py
import pandas as pd
import numpy as np
from food_predictor.data.data_loader import DataLoader
from food_predictor.synthetic.tonic_generator import TonicAIGenerator, TonicSynthesisEngine
from food_predictor.scripts.prepare_df5_for_tonic import prepare_df5_for_tonic
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_df5_synthetic_data(
    multiplier=3,
    api_key=None,
    validation_split=0.2
):
    """
    Complete pipeline to generate synthetic DF5 data using Tonic AI approach
    
    Args:
        multiplier: How many times to multiply the original data
        api_key: Tonic AI API key (optional)
        validation_split: Split for validation
    """
    
    print("=== Starting DF5 Synthetic Data Generation ===")
    
    # Step 1: Prepare data
    print("\n1. Preparing DF5 data...")
    df5_clean, constraints, prep_path = prepare_df5_for_tonic()
    
    original_size = len(df5_clean)
    target_size = original_size * multiplier
    
    print(f"Original data: {original_size} records")
    print(f"Target synthetic: {target_size} records")
    
    # Step 2: Split for validation
    from sklearn.model_selection import train_test_split
    
    train_data, validation_data = train_test_split(
        df5_clean,
        test_size=validation_split,
        stratify=df5_clean['Category'],
        random_state=42
    )
    
    print(f"Training data: {len(train_data)} records")
    print(f"Validation data: {len(validation_data)} records")
    
    # Step 3: Configure Tonic AI
    print("\n2. Configuring Tonic AI synthesis...")
    generator = TonicAIGenerator(api_key=api_key)
    config = generator.configure_df5_synthesis(train_data)
    
    # Step 4: Train synthesis model
    print("\n3. Training synthesis model...")
    synthesis_engine = TonicSynthesisEngine(config)
    model = synthesis_engine.train_synthesis_model(train_data)
    
    # Step 5: Generate synthetic data
    print("\n4. Generating synthetic data...")
    num_orders = int(target_size / (len(train_data) / train_data['Order_Number'].nunique()))
    synthetic_data = synthesis_engine.generate_synthetic_data(num_orders=num_orders)
    
    print(f"Generated {len(synthetic_data)} synthetic records")
    
    # Step 6: Validate synthetic data
    print("\n5. Validating synthetic data...")
    validation_results = validate_synthetic_data(
        original_data=train_data,
        synthetic_data=synthetic_data,
        validation_data=validation_data
    )
    
    # Step 7: Save results
    print("\n6. Saving results...")
    data_loader = DataLoader()
    
    synthetic_path = data_loader.save_synthetic_data(
        synthetic_data, 
        f'df5_synthetic_tonic_x{multiplier}.xlsx'
    )
    
    validation_path = data_loader.save_synthetic_data(
        pd.DataFrame(validation_results),
        f'df5_synthetic_validation_x{multiplier}.xlsx'
    )
    
    # Step 8: Create combined dataset
    combined_data = pd.concat([train_data, synthetic_data], ignore_index=True)
    combined_path = data_loader.save_synthetic_data(
        combined_data,
        f'df5_combined_real_synthetic_x{multiplier}.xlsx'
    )
    
    print(f"\n=== Generation Complete ===")
    print(f"Synthetic data: {synthetic_path}")
    print(f"Validation results: {validation_path}")
    print(f"Combined dataset: {combined_path}")
    print(f"Total records in combined: {len(combined_data)}")
    
    return {
        'synthetic_data': synthetic_data,
        'validation_results': validation_results,
        'paths': {
            'synthetic': synthetic_path,
            'validation': validation_path,
            'combined': combined_path
        }
    }

def validate_synthetic_data(original_data, synthetic_data, validation_data):
    """Validate quality of synthetic data"""
    
    validation_results = {}
    
    # 1. Distribution comparison
    print("   Validating distributions...")
    
    categorical_cols = ['Category', 'Event_Type', 'Meal_Time']
    for col in categorical_cols:
        orig_dist = original_data[col].value_counts(normalize=True)
        synth_dist = synthetic_data[col].value_counts(normalize=True)
        
        # Calculate similarity (simplified)
        common_values = set(orig_dist.index) & set(synth_dist.index)
        similarity = sum([min(orig_dist.get(v, 0), synth_dist.get(v, 0)) for v in common_values])
        
        validation_results[f'{col}_distribution_similarity'] = similarity
    
    # 2. Numerical validation
    print("   Validating numerical ranges...")
    
    numerical_cols = ['quantity_numeric', 'total_guest_count', 'veg_ratio']
    for col in numerical_cols:
        orig_stats = original_data[col].describe()
        synth_stats = synthetic_data[col].describe()
        
        validation_results[f'{col}_mean_diff'] = abs(orig_stats['mean'] - synth_stats['mean'])
        validation_results[f'{col}_std_diff'] = abs(orig_stats['std'] - synth_stats['std'])
    
    # 3. Business logic validation
    print("   Validating business constraints...")
    
    # Veg ratio constraint
    valid_veg_ratio = (
        (synthetic_data['veg_ratio'] >= 0) & 
        (synthetic_data['veg_ratio'] <= 1)
    ).mean()
    validation_results['valid_veg_ratio_percentage'] = valid_veg_ratio
    
    # Positive quantities
    valid_quantities = (synthetic_data['quantity_numeric'] > 0).mean()
    validation_results['valid_quantities_percentage'] = valid_quantities
    
    # 4. Uniqueness
    print("   Checking uniqueness...")
    unique_orders = synthetic_data['Order_Number'].nunique()
    total_orders = len(synthetic_data['Order_Number'].unique())
    validation_results['order_uniqueness'] = unique_orders / total_orders if total_orders > 0 else 0
    
    print(f"   Validation complete. Valid veg ratios: {valid_veg_ratio:.2%}")
    print(f"   Valid quantities: {valid_quantities:.2%}")
    
    return validation_results

if __name__ == "__main__":
    # Generate synthetic data
    results = generate_df5_synthetic_data(
        multiplier=3,  # Generate 3x the original data
        api_key=None,  # Set your Tonic AI API key here
        validation_split=0.2
    )
    
    print("\n=== Summary ===")
    print(f"Synthetic records generated: {len(results['synthetic_data'])}")
    print("Files saved:")
    for name, path in results['paths'].items():
        print(f"  {name}: {path}")