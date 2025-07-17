# Create: food_predictor/scripts/train_with_synthetic.py
from food_predictor.data.data_loader import DataLoader
from food_predictor.core.model_manager import ModelManager
from food_predictor.core.feature_engineering import FeatureEngineer
from food_predictor.core.category_rules import FoodCategoryRules
from food_predictor.data.item_service import ItemService
from food_predictor.core.menu_analyzer import MenuAnalyzer
import os
import pandas as pd

def train_model_with_synthetic_data(synthetic_multiplier=3):
    """Train model using synthetic + real data"""
    
    # Initialize components
    food_rules = FoodCategoryRules()
    item_service = ItemService(food_rules)
    feature_engineer = FeatureEngineer(food_rules)
    menu_analyzer = MenuAnalyzer(food_rules, item_service)
    model_manager = ModelManager(feature_engineer, item_service, food_rules, menu_analyzer)
    
    # Load combined dataset
    data_loader = DataLoader()
    combined_data = pd.read_excel(
        os.path.join(data_loader.synthetic_dir, f'df5_combined_real_synthetic_x{synthetic_multiplier}.xlsx')
    )
    
    print(f"Training with {len(combined_data)} records (real + synthetic)")
    
    # Build metadata
    item_service.build_item_metadata(combined_data)
    
    # Train model
    model_manager.fit(combined_data)
    
    # Save model
    model_path = model_manager.save_models('./models_with_synthetic')
    
    print(f"Model trained and saved to: {model_path}")
    
    return model_manager

if __name__ == "__main__":
    model_manager = train_model_with_synthetic_data(synthetic_multiplier=3)