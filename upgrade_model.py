"""
Model Metadata Upgrade Script - Adds item_metadata persistence to existing model artifacts

Usage: python upgrade_model.py --model_path models/food_prediction_model.joblib --data_path DB38.xlsx --output_path models/upgraded_model.joblib
"""

import os
import sys
import argparse
import pandas as pd
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure food_predictor package is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from food_predictor.core.model_manager import ModelManager
from food_predictor.core.feature_engineering import FeatureEngineer
from food_predictor.core.category_rules import FoodCategoryRules
from food_predictor.data.item_service import ItemService
from food_predictor.core.menu_analyzer import MenuAnalyzer

def upgrade_model_serialization(model_path, data_path, output_path):
    """
    Upgrade an existing model by adding item_metadata to the serialized artifact.
    
    Args:
        model_path: Path to existing model file
        data_path: Path to training data (Excel file)
        output_path: Path to save the upgraded model
    """
    logger.info(f"Starting model metadata upgrade process")
    
    # 1. Initialize the component chain
    logger.info("Initializing component chain")
    food_rules = FoodCategoryRules()
    item_service = ItemService(food_rules)
    feature_engineer = FeatureEngineer(food_rules)
    menu_analyzer = MenuAnalyzer(food_rules, item_service)
    model_manager = ModelManager(feature_engineer, item_service, food_rules, menu_analyzer)
    
    # 2. Load existing trained model (weights and parameters)
    logger.info(f"Loading existing model from {model_path}")
    try:
        model_manager.load_models(model_path)
        logger.info(f"Successfully loaded model with {len(model_manager.category_models)} category models")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False
    
    # 3. Rebuild metadata dictionary
    logger.info(f"Rebuilding item metadata from {data_path}")
    try:
        data = pd.read_excel(data_path)
        logger.info(f"Loaded training data with {len(data)} rows")
        
        # Build item metadata and matcher
        item_service.build_item_metadata(data)
        logger.info(f"Built metadata for {len(item_service.item_metadata)} unique items")
    except Exception as e:
        logger.error(f"Failed to rebuild metadata: {str(e)}")
        return False
    
    # 4. Re-serialize with enhanced metadata persistence
    logger.info(f"Saving upgraded model to {output_path}")
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        model_manager.save_models(output_path)
        
        # Verify metadata is included
        saved_model = joblib.load(output_path)
        if 'item_metadata' in saved_model:
            metadata_count = len(saved_model['item_metadata'])
            logger.info(f"Verification successful: item_metadata with {metadata_count} entries included in saved model")
        else:
            logger.error("Verification failed: item_metadata not found in saved model")
            return False
            
        logger.info(f"Model upgrade completed successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to save upgraded model: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upgrade existing model with item metadata persistence")
    parser.add_argument("--model_path", required=True, help="Path to existing model file")
    parser.add_argument("--data_path", required=True, help="Path to training data Excel file")
    parser.add_argument("--output_path", required=True, help="Path to save the upgraded model")
    
    args = parser.parse_args()
    
    success = upgrade_model_serialization(args.model_path, args.data_path, args.output_path)
    
    if success:
        logger.info("✅ Model upgrade completed successfully")
        sys.exit(0)
    else:
        logger.error("❌ Model upgrade failed")
        sys.exit(1)