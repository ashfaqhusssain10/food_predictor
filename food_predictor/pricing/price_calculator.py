import math
import logging
from typing import Dict, Tuple, Optional, Union, Any
from food_predictor.core.category_rules import FoodCategoryRules
from food_predictor.data.item_service import ItemService
from food_predictor.data.item_matcher import FoodItemMatcher
from food_predictor.core.model_manager import ModelManager
from food_predictor.utils.quantity_utils import extract_quantity_value,extract_unit,infer_default_unit,validate_unit

logger = logging.getLogger("PriceCalculator")

class PriceCalculator:
    """
    Implements food item pricing algorithms with quantity-based scaling
    and category-specific parameters.
    
    The pricing model uses a sophisticated power-law based calculation
    that accounts for economies of scale, minimum viable quantities,
    and category-specific base rates.
    """
    
    def __init__(self, food_rules, item_service):
        """
        Initialize price calculator with required dependencies.
        
        Args:
            food_rules: FoodCategoryRules instance for category rules and units
            item_service: ItemService for item metadata and VC mappings
        """
        self.food_rules = food_rules
        self.item_service = item_service
        
        # Default pricing parameters
        self.default_parameters = {
            "FC": 6000,       # Base fixed cost
            "Qmin_static": 50,  # Minimum static quantity
            "beta": 0.5,      # Power exponent for dynamic minimum quantity
            "default_vc": 220  # Default variable cost when no category specific VC found
        }
        
        # Category-specific default VCs
        self.category_defaults = {
            "Desserts": 45,
            "Appetizers": 60,
            "Biryani": 320,
            "Curries": 300,
            "Breads": 10
        }
    
    def calculate_price(self, converted_qty: float, category: str, 
                        total_guest_count: int, item_name: str, unit: str) -> Tuple[float, float, float]:
        """
        Calculate price using comprehensive unit handling and dynamic scaling.
        
        Implements a sophisticated pricing model with power-law scaling, unit
        conversion, and per-category optimizations.
        
        Args:
            converted_qty: The total quantity (e.g., 120g, 2pcs)
            category: Category of the food item
            total_guest_count: Number of guests
            item_name: Name of the food item
            unit: Unit of measurement (g, kg, ml, l, pcs)
            
        Returns:
            Tuple of (total_price, base_price_per_unit, per_person_price)
        """
        # Normalize category name for rule lookup
        normalized_category = self.food_rules.normalize_category_name(category)
        rule = self.food_rules.category_rules.get(normalized_category, {})
        
        # Initialize pricing parameters
        FC = self.default_parameters["FC"]
        Qmin_static = self.default_parameters["Qmin_static"]
        beta = self.default_parameters["beta"]
        
        # Get item-specific data with proper fallbacks
        std_item_name = self.item_service.standardize_item_name(item_name)
        item_data = self.item_service.item_specific_data.get(std_item_name, {})
        
        # Try alternative standardization if not found
        if not item_data and '>' in item_name:
            alt_std_name = self.item_service.standardize_item_name(item_name.replace('>', '').strip())
            item_data = self.item_service.item_specific_data.get(alt_std_name, {})
        
        # Extract core pricing parameters
        preferred_unit = item_data.get('preferred_unit', None)
        base_price_per_piece = item_data.get('base_price_per_piece', None)
        
        # Determine appropriate VC (variable cost) value with fallback chain
        vc_data = self.item_service.item_vc_mapping.get(std_item_name, {})
        if not vc_data and '>' in item_name:
            alt_std_name = self.item_service.standardize_item_name(item_name.replace('>', '').strip())
            vc_data = self.item_service.item_vc_mapping.get(alt_std_name, {})
        
        # Get category-appropriate VC with fallback chain
        category_vc = rule.get("vc_price", self.default_parameters["default_vc"])
        
        # Override with category defaults if available
        if normalized_category in self.category_defaults:
            category_vc = self.category_defaults[normalized_category]
        
        # Final VC determination
        VC = vc_data.get('VC', category_vc)
        
        # Get p-value with fallback to default
        p_value = vc_data.get('p_value', 0.19)
        
        # Constrain p-value to realistic bounds
        p_value = max(0.05, min(0.35, p_value))
        
        # Determine unit for pricing with fallbacks
        default_qty, default_unit = self.food_rules.get_default_quantity(normalized_category)
        unit = unit if unit else default_unit
        unit = validate_unit(normalized_category, unit)
        
        # Use preferred unit if specified in item data
        if preferred_unit:
            unit = preferred_unit
        
        # Ensure positive quantity
        converted_qty = max(1, converted_qty)
        
        # Calculate per-person quantity
        per_person_qty = converted_qty / total_guest_count if total_guest_count > 0 else 0
        
        # Initialize conversion factor
        display_to_pricing_factor = 1.0
        
        # Handle different unit types for pricing
        if unit == "pcs":
            # Piece-based items use direct pricing
            Q_threshold = 45
            Q_in_threshold_unit = converted_qty
            base_price_per_unit = base_price_per_piece if base_price_per_piece is not None else VC
        else:
            # Weight or volume-based items need conversion factors
            if unit == "kg":
                base_threshold = 50
                Q_in_threshold_unit = converted_qty
                display_to_pricing_factor = 1.0  # Already in kg
            elif unit == "g":
                base_threshold = 50000  # 50 kg in g
                Q_in_threshold_unit = converted_qty / 1000  # Convert g to kg
                display_to_pricing_factor = 1000.0
            elif unit == "l":
                base_threshold = 60
                Q_in_threshold_unit = converted_qty
                display_to_pricing_factor = 1.0  # Already in L
            elif unit == "ml":
                base_threshold = 60000  # 60 L in ml
                Q_in_threshold_unit = converted_qty / 1000
                display_to_pricing_factor = 1000.0
            else:
                base_threshold = 50
                Q_in_threshold_unit = converted_qty / 1000
                display_to_pricing_factor = 1000.0
            
            # Scale threshold based on guest count
            Q_threshold = base_threshold * (1 + (total_guest_count / 1000))
            base_price_per_unit = item_data.get('base_price_per_kg', VC)
        
        # Calculate dynamic minimum quantity
        Qmin = min(max(Qmin_static, Q_in_threshold_unit ** beta), Q_threshold)
        
        # Calculate base rate with protection against division by zero
        base_rate = FC / max(Qmin, 0.1) + VC
        
        # Calculate final price based on unit type
        if unit == "pcs":
            # Piece-based pricing 
            price_per_piece = base_price_per_unit
            total_price = price_per_piece * Q_in_threshold_unit
            price_per_person = price_per_piece * per_person_qty
        else:
            # Weight/volume-based pricing with dynamic scaling
            if Q_in_threshold_unit <= 0.1:
                # Very small quantity premium
                unit_price = base_rate
            elif Q_in_threshold_unit <= Q_threshold:
                # Normal scale pricing with power-law scaling
                unit_price = base_rate * (Q_in_threshold_unit ** (-p_value))
            else:
                # Large scale pricing - fixed unit price after threshold
                unit_price = base_rate * (Q_threshold ** (-p_value))
            
            # Store base price per unit for reference
            base_price_per_unit = unit_price
            
            # Calculate total price based on pricing unit
            total_price = unit_price * Q_in_threshold_unit
            
            # Calculate per-person price in original display unit
            price_per_person = (unit_price / display_to_pricing_factor) * per_person_qty
        
        # Sanity check and fallback for invalid prices
        if total_price < 0 or math.isnan(total_price) or math.isinf(total_price):
            logger.warning(f"Invalid price calculation for {item_name}: {total_price}. Using fallback.")
            total_price = VC * Q_in_threshold_unit  # Simple fallback
        
        # Ensure reasonable per-person pricing
        if price_per_person < 0 or math.isnan(price_per_person) or math.isinf(price_per_person):
            price_per_person = total_price / total_guest_count if total_guest_count > 0 else 0
        
        # Return results with float conversion for consistency
        return float(total_price), float(base_price_per_unit), float(price_per_person)

def predict_and_price(model_manager, price_calculator, event_time, meal_type, event_type, total_guest_count, veg_guest_count, menu_analyzer, selected_items):
    """
    Utility function to get predicted quantities and prices for each menu item.
    """
    # Get predictions
    predictions = model_manager.predict(
        event_time, meal_type, event_type, total_guest_count, veg_guest_count, menu_analyzer
    )

    results = {}
    from food_predictor.utils.quantity_utils import extract_quantity_value
    for item, pred in predictions.items():
        qty = extract_quantity_value(pred['total'])
        unit = pred['unit']
        category = pred['category']
        total_price, base_price_per_unit, per_person_price = price_calculator.calculate_price(
            converted_qty=qty,
            category=category,
            total_guest_count=total_guest_count,
            item_name=item,
            unit=unit
        )
        results[item] = {
            'predicted_quantity': pred['total'],
            'unit': unit,
            'total_price': total_price,
            'per_person_price': per_person_price
        }
    return results

# Example usage (uncomment to run as a script):
# if __name__ == "__main__":
#     from food_predictor.core.model_manager import ModelManager
#     from food_predictor.core.feature_engineering import FeatureEngineer
#     from food_predictor.core.category_rules import FoodCategoryRules
#     from food_predictor.data.item_service import ItemService
#     from food_predictor.core.menu_analyzer import MenuAnalyzer
#
#     food_rules = FoodCategoryRules()
#     item_service = ItemService(food_rules)
#     feature_engineer = FeatureEngineer(food_rules)
#     menu_analyzer = MenuAnalyzer(food_rules, item_service)
#     model_manager = ModelManager(feature_engineer, item_service, food_rules)
#     model_manager.load_models('./models')
#     price_calculator = PriceCalculator(food_rules, item_service)
#
#     event_time = "Evening"
#     meal_type = "Dinner"
#     event_type = "Wedding"
#     total_guest_count = 100
#     veg_guest_count = 40
#     selected_items = ["Paneer Butter Masala", "Biryani", "Curd", "Papad"]
#
#     results = predict_and_price(
#         model_manager, price_calculator, event_time, meal_type, event_type,
#         total_guest_count, veg_guest_count, menu_analyzer, selected_items
#     )
#     for item, info in results.items():
#         print(f"{item}: {info['predicted_quantity']} ({info['unit']}) | Total Price: ₹{info['total_price']:.2f} | Per Person: ₹{info['per_person_price']:.2f}")