"""
AWS Lambda handler for the Food Quantity Prediction service.
Handles initialization, model loading, and request processing.
"""

import os
import json
import logging
import time
import pandas as pd
from typing import Dict, Any

from food_predictor.api.request_processor import RequestProcessor
from food_predictor.core.model_manager import ModelManager
from food_predictor.core.feature_engineering import FeatureEngineer
from food_predictor.core.category_rules import FoodCategoryRules
from food_predictor.data.item_service import ItemService
from food_predictor.core.menu_analyzer import MenuAnalyzer
from food_predictor.pricing.price_calculator import PriceCalculator

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize components once, outside the handler
def initialize_components():
    """Initialize model components - done once on cold start."""
    start_time = time.time()
    logger.info("Initializing model components...")
    
    # Create base directory paths
    base_dir = os.environ.get('LAMBDA_TASK_ROOT', os.path.dirname(os.path.dirname(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    data_path = os.path.join(base_dir, 'DB38.xlsx')
    
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Models directory: {models_dir}")
    logger.info(f"Data path: {data_path}")
    
    # Initialize component chain
    food_rules = FoodCategoryRules()
    item_service = ItemService(food_rules)
    feature_engineer = FeatureEngineer(food_rules)
    menu_analyzer = MenuAnalyzer(food_rules, item_service)
    model_manager = ModelManager(feature_engineer, item_service, food_rules, menu_analyzer)
    price_calculator = PriceCalculator(food_rules, item_service)
    
    # Check if data file exists for item metadata
    if os.path.exists(data_path):
        logger.info(f"Loading item metadata from {data_path}")
        try:
            data = pd.read_excel(data_path)
            item_service.build_item_metadata(data)
            logger.info(f"Loaded metadata for {len(item_service.item_metadata)} items")
        except Exception as e:
            logger.error(f"Error loading item metadata: {str(e)}")
    else:
        logger.warning(f"Data file not found at {data_path}, continuing without item metadata")
    
    # Check if models directory exists
    if os.path.exists(models_dir):
        logger.info(f"Loading models from {models_dir}")
        try:
            model_manager.load_models(models_dir)
            logger.info(f"Models loaded successfully with {len(model_manager.category_models)} category models and {len(model_manager.item_models)} item models")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
    else:
        logger.error(f"Models directory not found at {models_dir}")
    
    logger.info(f"Component initialization completed in {time.time() - start_time:.2f} seconds")
    
    return model_manager, price_calculator, item_service

# Initialize components once on cold start
request_processor = RequestProcessor()
model_manager, price_calculator, item_service = initialize_components()

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler function for food quantity prediction.
    
    Args:
        event: Lambda event containing the request data
        context: Lambda context
        
    Returns:
        API Gateway response
    """
    logger.info(f"Received event: {json.dumps(event)}")
    start_time = time.time()
    
    # Process request
    request_data, is_valid, error_message = request_processor.process_request(event)
    
    if not is_valid:
        logger.error(f"Invalid request: {error_message}")
        return request_processor.format_response({}, is_error=True, error_message=error_message)
    
    try:
        # Extract parameters from validated request
        event_time = request_data['event_time']
        meal_type = request_data['meal_type']
        event_type = request_data['event_type']
        total_guest_count = request_data['total_guest_count']
        veg_guest_count = request_data['veg_guest_count']
        selected_items = request_data['selected_items']
        
        # Log request parameters
        logger.info(f"Prediction request: {event_time}, {meal_type}, {event_type}, {total_guest_count} guests ({veg_guest_count} veg), {len(selected_items)} items")
        
        # Generate predictions
        predictions = model_manager.predict(
            event_time, meal_type, event_type, 
            total_guest_count, veg_guest_count, selected_items
        )
        
        # Add price information if possible
        for item, pred in predictions.items():
            try:
                qty = float(pred['total'].split(pred['unit'])[0])
                unit = pred['unit']
                category = pred['category']
                
                # Calculate price
                total_price, base_price_per_unit, per_person_price = price_calculator.calculate_price(
                    converted_qty=qty,
                    category=category,
                    total_guest_count=total_guest_count,
                    item_name=item,
                    unit=unit
                )
                
                # Add price information to prediction
                predictions[item]['price'] = {
                    'total': round(total_price, 2),
                    'per_person': round(per_person_price, 2),
                    'base_per_unit': round(base_price_per_unit, 2)
                }
            except Exception as e:
                # Just log the error but continue without price info
                logger.warning(f"Error calculating price for {item}: {str(e)}")
        
        # Add metadata to prediction results
        predictions['total_guest_count'] = total_guest_count
        predictions['veg_guest_count'] = veg_guest_count
        predictions['event_type'] = event_type
        predictions['meal_type'] = meal_type
        predictions['event_time'] = event_time
        
        # Format response
        response = request_processor.format_response(predictions)
        
        # Log timing
        exec_time = time.time() - start_time
        logger.info(f"Prediction completed in {exec_time:.2f} seconds for {len(selected_items)} items")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return request_processor.format_response({}, is_error=True, error_message=f"Internal server error: {str(e)}")