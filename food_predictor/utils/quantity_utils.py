import re
import logging
import pandas as pd
from typing import Optional

# Initialize module-level logger
logger = logging.getLogger(__name__)

def extract_quantity_value(quantity_str: str) -> float:
    """Extract numerical value from a quantity string."""
    if not quantity_str or pd.isna(quantity_str):
        return 0.0
    match = re.search(r'(\d+\.?\d*)', str(quantity_str))
    if match:
        return float(match.group(1))
    return 0.0

def extract_unit(quantity_str: str) -> str:
    """Extract unit from a quantity string."""
    if not quantity_str or pd.isna(quantity_str):
        return ''
    match = re.search(r'[a-zA-Z]+', str(quantity_str))
    if match:
        return match.group(0)
    return ''

def infer_default_unit(category: str) -> str:
    """Determine default unit based on food category."""
    liquid_categories = ["Liquids(Less_Dense)", "Liquids(High_Dense)", "Soups", "Welcome_Drinks", "Hot_Beverages"]
    piece_categories = ["Breakfast_pcs","Sides_pcs"]
    
    if category in liquid_categories:
        return "ml"
    elif category in piece_categories:
        return "pcs"
    else:
        return "g"

def validate_unit(category: str, unit: str) -> str:
    """Validate and normalize unit for a specific food category."""
    liquid_categories = ["Liquids(Less_Dense)", "Liquids(High_Dense)", "Soups", "Welcome_Drinks", "Hot_Beverages"]
    piece_categories = ["Breakfast_pcs","Sides_pcs"]
    
    if not unit:
        # Direct function call instead of self.method()
        inferred_unit = infer_default_unit(category)
        return inferred_unit
        
    if category in liquid_categories and unit not in ["ml", "l"]:
        logger.warning(f"Invalid unit '{unit}' for liquid category '{category}', defaulting to 'ml'")
        return "ml"
    elif category in piece_categories and unit != "pcs":
        logger.warning(f"Invalid unit '{unit}' for piece category '{category}', defaulting to 'pcs'")
        return "pcs"
    elif unit not in ["g", "kg", "ml", "l", "pcs"]:
        logger.warning(f"Invalid unit '{unit}' for category '{category}', defaulting to 'g'")
        return "g"
    
    return unit