# food_predictor/core/category_rules.py
from typing import Dict, List, Optional, Union, Tuple, Any
import re
import logging
logger = logging.getLogger(__name__)
#from food_predictor.core.menu_analyzer import MenuAnalyzer
# Import previously extracted utility functions
from food_predictor.utils.quantity_utils import (
    extract_quantity_value,
    extract_unit,
    infer_default_unit,
    validate_unit
)

class FoodCategoryRules:
    def __init__(self):
        logger.info("Initializing FoodCategoryRules")
        #self.menu_context = {}
        self.category_rules = self._initialize_category_rules()
        self.category_dependencies = self._initialize_category_dependencies()
        self.meal_type_modifiers = self._initialize_meal_type_modifiers()

    def _initialize_category_rules(self) -> Dict[str, Dict[str, Any]]:
        return{
            "Biryani": {
                "min_quantity": "250g",
                "max_quantity": "550g",
                "default_quantity": "320g",
                "vc_price": 260,
                "adjustments": {
                    "per_person": lambda total_items: "550g" if total_items == 1 else ("250g" if total_items > 3 else "300g"),
                    #"per_person": lambda total_guest_count: "250g" if total_guest_count > 100 else "300g",
                    #"variety_count": lambda count: "280g" if count > 2 else "350g",
                    "total_items": lambda total_items: ("550g" if total_items <= 3 else
                                                        "400g" if total_items <= 9 else
                                                        "350g")
                }},
            "Welcome_Drinks": {
                "min_quantity": "100ml",
                "max_quantity": "120ml",
                "default_quantity": "120ml",
                "vc_price": 180,
            },
            "Paan": {
                "min_quantity": "1pcs",
                "max_quantity": "2pcs",
                "default_quantity": "1pcs",
                "vc_price": 150,
            },  
             "Sides_pcs": {
                "min_quantity": "1pcs",
                "max_quantity": "2pcs",
                "default_quantity": "1pcs",
                "vc_price": 150,
            },  
             "Sides": {
                "min_quantity": "15g",
                "max_quantity": "40g",
                "default_quantity": "30g",
                "vc_price": 70,
            },
            "Salad": {
                "min_quantity": "50g",
                "max_quantity": "50g",
                "default_quantity": "50g",
                "vc_price": 70,
            },
            "Podi": {
                "min_quantity": "10g",
                "max_quantity": "50g",
                "default_quantity": "10g",
                "vc_price": 100,
            },
            "Ghee": {
                "min_quantity": "5g",
                "max_quantity": "10g",
                "default_quantity": "5g",
                "vc_price": 150,
            },
            "Pickle": {
                "min_quantity": "10g",
                "max_quantity": "10g",
                "default_quantity": "10g",
                "vc_price": 180,
            },
            "Flavored_Rice": {
                "min_quantity": "80g",
                "max_quantity": "100g",
                "default_quantity": "100g",
                "vc_price": 150,
                "adjustments": {
                    "variety_count": lambda count: {
                        1: "100g",
                        2: "80g",
                        3: "80g",
                        4: "80g",
                        5: "80g",
                    }.get(count, "80g")
                }
            },
            "Soups": {
                "min_quantity": "60ml",
                "max_quantity": "120ml",
                "default_quantity": "100ml",
                "vc_price": 150,
                "adjustments": {
                    "variety_count": lambda count: {
                        1: "120ml",
                        2: "100ml",
                        3: "80ml",
                        4: "70ml",
                        5: "60ml"
                    }.get(count, "60ml")
                }
            },
            "Crispers": {
                "min_quantity": "10g",
                "max_quantity": "20g",
                "default_quantity": "15g",
                "vc_price": 40,
            },
            "Fried_Rice": {
                "min_quantity": "80g",
                "max_quantity": "120g",
                "default_quantity": "80g",
                "vc_price": 70,
                "adjustments": {
                    "per_person": lambda total_guest_count: "100g" if total_guest_count > 100 else "100g",
                    "variety_count": lambda count: "80g" if count > 3 else "100g"
                }
            },
            "fry": {
                "min_quantity": "30g",
                "max_quantity": "40g",
                "default_quantity": "30g",
                "vc_price": 60,
            },
            "Fryums": {
                "min_quantity": "5g",
                "max_quantity": "10g",
                "default_quantity": "5g",
                "vc_price": 10,
            },
            "Salan": {
                "min_quantity": "40g",
                "max_quantity": "40g",
                "default_quantity": "40g",
                "vc_price": 80,
            },
            "Bakery": {
                "min_quantity": "40g",
                "max_quantity": "80g",
                "default_quantity": "40g",
                "vc_price": 80,
            },
            
            "Cakes": {
                "min_quantity": "100g",
                "max_quantity": "200g",
                "default_quantity": "120g",
                "vc_price": 400,
                "adjustments": {
                    "per_person": lambda total_guest_count: "150g" if total_guest_count > 30 else "150g"
                }
            },
            "Cup_Cakes": {
                "min_quantity": "50g",
                "max_quantity": "100g",
                "default_quantity": "50g",
                "vc_price": 200,
            },
            "Hot_Beverages": {
                "min_quantity": "100ml",
                "max_quantity": "120ml",
                "default_quantity": "120ml",
                "vc_price": 150,
                "adjustments": {
                "variety_count": lambda count: {
                    1: "120ml",
                    2: "100ml",
                    3: "80ml",
                    4: "70ml",
                    5: "60ml"
                }.get(count, "60ml")
    }
            },
            "Appetizers": {
                "min_quantity": "50g",
                "max_quantity": "120g",
                "default_quantity": "120g",
                "vc_price": 270,
                #"piece_weight":40,
                "adjustments": {
                    "variety_count": lambda count: {
                        1: "120g",
                        2: "100g",
                        3: "80g",
                        4: "60g",
                        5: "50g",
                        6: "50g",
                        7: "50g",
                        8: "50g",
                        9: "50g",
                        10: "50g"
                    }.get(count, "80g")
                }
        },
            "Roti_Pachadi": {
                "min_quantity": "15g",
                "max_quantity": "20g",
                "default_quantity": "15g",
                "vc_price": 80,
            },
            "Curries": {
                "min_quantity": "80g",
                "max_quantity": "180g",
                "default_quantity": "120g",
                "vc_price": 270,
                "adjustments": {
                    "variety_count": lambda count: {
                        1: "150g",
                        2: "120g",
                        3: "80g",
                        4: "80g",
                        5: "80g",
                        6: "80g",
                    }.get(count, "100g")
                }
            },
            "Rice": {
                "min_quantity": "150g",
                "max_quantity": "250g",
                "default_quantity": "200g",
                "vc_price": 70,
                "adjustments": {
                    "with_curry": lambda: "200g",
                    "standalone": lambda: "250g"
                }
            },
            "Liquids(Less_Dense)": {
                "min_quantity": "60ml",
                "max_quantity": "100ml",
                "default_quantity": "70ml",
                "vc_price": 80,
                "adjustments": {
                    "with_dal": lambda: "60ml",
                    "standalone": lambda: "100ml"
                }
            },
            "Liquids(High_Dense)": {
                "min_quantity": "30ml",
                "max_quantity": "30ml",
                "default_quantity": "30ml",
                "vc_price": 160,
            },
            "Dal": {
                "min_quantity": "60g",
                "max_quantity": "80g",
                "default_quantity": "70g",
                "vc_price": 120,
            },
            "Desserts": {
                "min_quantity": "80g",
                "max_quantity": "120g",
                "default_quantity": "100g",
                #"piece_weight":40,
                "vc_price": 170,
                "adjustments": {
                    "variety_count": lambda count: "80g" if count > 2 else "100g"
                }
            },
            "Curd": {
                "min_quantity": "50g",
                "max_quantity": "60g",
                "default_quantity": "50g",        
                "vc_price": 50,
            },
            "Fruits": {
                "min_quantity": "100g",
                "max_quantity": "120g",
                "default_quantity": "100g",
                "vc_price": 150,
            },
            
            "Chutneys": {
                "min_quantity": "50g",
                "max_quantity": "70g",
                "default_quantity": "50g",
                "vc_price": 150,
            },
            "French_Fries": {
                "min_quantity": "50g",
                "max_quantity": "70g",
                "default_quantity": "50g",
                "vc_price": 150,
            },
            "Samosa": {
                "min_quantity": "50g",
                "max_quantity": "70g",
                "default_quantity": "50g",
                "vc_price": 150,
            },
            "Dips": {
                "min_quantity": "10g",
                "max_quantity": "20g",
                "default_quantity": "15g",
                "vc_price": 50,
            },
            
            "Papad": {
                "min_quantity": "1pcs",
                "max_quantity": "3pcs",
                "default_quantity": "2pcs",
                "vc_price": 10,
            },
            "Breads": {
                "min_quantity": "40g",
                "max_quantity": "100g",
                "default_quantity": "80g",
                "vc_price": 40,
                "adjustments": {
                    "with_curry": lambda: "60g",
                }
            },
            "Italian": {
                "min_quantity": "100g",
                "max_quantity": "120g",
                "default_quantity": "100g",
                "vc_price": 150,
                "adjustments": {
                    "with_pasta": lambda: "120g",
                    "variety_count": lambda count: {
                        1: "120g",
                        2: "100g",
                        3: "80g",
                        4: "70g",
                        5: "60g"
                    }.get(count, "60g")
                    }
                
            },
            "Breakfast_pcs": {
                "min_quantity": "2pcs",
                "max_quantity": "4pcs",
                "default_quantity": "2pcs",
                "vc_price": 200,
            },
            "Raitha": {
                "min_quantity": "50g",
                "max_quantity": "60g",
                "default_quantity": "60g",
                "vc_price": 60,
                "adjustments": {
                    "with_biryani": lambda: "70g",
                    "standalone": lambda: "50g"
                }
            },
            "Breakfast": {
                "min_quantity": "100g",
                "max_quantity": "150g",
                "default_quantity": "100g",
                "vc_price": 150,
                #"piece_weight": 50,
            },
            "Sandwich": {
                "min_quantity": "100g",
                "max_quantity": "150g",
                "default_quantity": "100g",
                "vc_price": 150,
            }
        }

    def _initialize_category_dependencies(self) -> Dict[str, Dict[str, float]]:
        return  {
    "Rice": {"Curries": 0.9, "Dal": 0.7, "Curd": 0.6, "Liquids(Less_Dense)": 0.6, "Chutneys": 0.5, "Raitha": 0.5, "Pickle": 0.5, "Roti_Pachadi": 0.5},
    "Biryani": {"Salan": 0.85, "Raitha": 0.8, "Salad": 0.4},
    "Flavored_Rice": {"Raitha": 0.7, "Pickle": 0.6, "Salan": 0.5},
    "Fried_Rice": {"Curries": 0.7, "Chutneys": 0.5, "Raitha": 0.4},
    "Breads": {"Curries": 0.9, "Dal": 0.7},
    "Breakfast": {"Chutneys": 0.8, "Hot_Beverages": 0.9},
    "Breakfast_pcs": {"Chutneys": 0.8, "Hot_Beverages": 0.9},
    "Sandwich": {"Soups": 0.6, "Welcome_Drinks": 0.5},
    "Italian": {"Salad": 0.7, "Welcome_Drinks": 0.5},
    "Samosa": {"Chutneys": 0.8},
    "Curries": {"Rice": 0.8, "Breads": 0.75},
    "Dal": {"Rice": 0.8, "Breads": 0.7},
    "Appetizers": {"Welcome_Drinks": 0.7}
}

    def _initialize_meal_type_modifiers(self) -> Dict[str, float]:
        return {
            "Breakfast": 1.0,
            "Lunch": 1.0,
            "Hi-Tea": 1.0,
            "Dinner": 1.0
        }

   

    def normalize_category_name(self, category: str) -> str:
        normalized = category.replace(" ", "_").strip()
        if normalized == "Bread":
            return "Breads"
        return normalized

    

    def get_default_quantity(self, category: str) -> Tuple[float, str]:
        normalized_category = self.normalize_category_name(category)
        if normalized_category in self.category_rules:
            rule = self.category_rules[normalized_category]
            default_qty_str = rule["default_quantity"]
            default_qty = extract_quantity_value(default_qty_str)
            unit = extract_unit(default_qty_str)
            unit = validate_unit(normalized_category, unit)
            return default_qty, unit
        inferred_unit =infer_default_unit(normalized_category)
        return 0.0, inferred_unit

    def get_item_specific_quantity(self, item_name, total_guest_count, item_specific_data):
        """
        Get item-specific quantity based on per-guest ratio.

        Args:
            item_name: Name of the item (string or tuple)
            guest_count: Number of guests
            item_specific_data: Dictionary of item-specific data

        Returns:
            Tuple of (quantity, unit) or (None, None) if not available
        """
        # Handle if item_name is a tuple (output from FoodItemMatcher)
        if isinstance(item_name, tuple) and len(item_name) > 0:
            item_name = item_name[0]  # Extract the string from the tuple

        # Now proceed with the standard processing
        std_item_name = item_name.lower().strip() if isinstance(item_name, str) else ""
        if std_item_name not in item_specific_data:
            return None, None

        item_data = item_specific_data[std_item_name]
        preferred_unit = item_data['preferred_unit']
        per_guest_ratio = item_data['per_guest_ratio']

        if preferred_unit == "pcs" and per_guest_ratio is not None:
            quantity = total_guest_count * per_guest_ratio
            return quantity, "pcs"
        return None, None
        #need to remove the special Categories and the unit_type from the function
    def apply_category_rules(self, item_name, item_properties, menu_context, main_items_info):
        """
        Apply category-specific quantity rules with comprehensive context considerations.
        
        Args:
            item_name (str): Name of the food item
            item_properties (dict): Comprehensive item properties
            menu_context (dict): Detailed menu context
            main_items_info (dict): Information about main menu items
        
        Returns:
            dict: Quantity prediction with detailed breakdown
        """
        # Extract key information
        category = item_properties['category']
        quantity_rule = item_properties.get('quantity_rule')
        # Ensure quantity_rule is not None - ADD THIS CHECK
        if not quantity_rule \
           or quantity_rule.get("default_quantity") in (None, "", "100g"):
            quantity_rule = None
        
        
        normalized_category = self.normalize_category_name(category)
        
        
        
        if quantity_rule is None or 'default_quantity' not in quantity_rule:
    # Lookup category rule
            if normalized_category in self.category_rules:
                quantity_rule = self.category_rules[normalized_category]
                logger.debug(f"Using category rule for {item_name} (category: {category})")
            else:
                # Fallback to safe default
                quantity_rule = {
                    "default_quantity": "100g",
                    "min_quantity": "50g",
                    "max_quantity": "150g",
                    "vc_price": 100,
                }
                logger.warning(f"No category rule found for {normalized_category}, using generic defaults for {item_name}")
        
        
        # Determine base quantity from category rules
        default_qty_str = quantity_rule.get('default_quantity', '100g')
        base_quantity = extract_quantity_value(default_qty_str)  # FIXED
        base_unit = extract_unit(default_qty_str)  # FIXED
        
        # Retrieve category context
        category_context = menu_context['category_context'].get(category, {})
        total_items_in_category = category_context.get('total_items', 1)
        
        # Apply variety count adjustments
        if 'adjustments' in quantity_rule and 'variety_count' in quantity_rule['adjustments']:
            adjusted_qty_str = quantity_rule['adjustments']['variety_count'](total_items_in_category)
            base_quantity = extract_quantity_value(adjusted_qty_str)  # FIXED
        
        # Main item context adjustments
        if main_items_info.get('has_main_item', False):
            primary_main_category = main_items_info.get('primary_main_category')
            
            # Contextual adjustments based on main item
            if category == 'Appetizers' and primary_main_category in ['Biryani', 'Rice']:
                base_quantity *= 1.0  # Slightly reduce if main item is substantial
            elif category == 'Sides' and primary_main_category in ['Biryani', 'Pulav']:
                base_quantity *= 1.0 # Slightly increase complementary sides
        
        # Veg/Non-veg guest considerations
        veg_guest_ratio = menu_context['veg_guest_count'] / menu_context['total_guest_count']
        is_veg = item_properties.get('is_veg', 'veg')
        
        if is_veg == 'veg':
            # Boost vegetarian items slightly in mixed groups
            base_quantity *= (1 + (0.2 * (1 - veg_guest_ratio)))
        elif is_veg == 'non-veg':
            # Adjust non-veg items based on non-veg guest ratio
            base_quantity *= (1 + (0.2 * veg_guest_ratio))
        
        # Additional per-person adjustments if available
        if 'adjustments' in quantity_rule and 'per_person' in quantity_rule['adjustments']:
            total_guest_count = menu_context['total_guest_count']
            adjusted_qty_str = quantity_rule['adjustments']['per_person'](total_guest_count)
            base_quantity = extract_quantity_value(adjusted_qty_str)  # FIXED
        
        # Meal type modifier
        meal_type = menu_context['meal_type']
        if meal_type in self.meal_type_modifiers:  # FIXED
            base_quantity *= self.meal_type_modifiers[meal_type]  # FIXED
        
        # Bounds check
        if 'min_quantity' in quantity_rule:
            min_qty = extract_quantity_value(quantity_rule['min_quantity'])  # FIXED
            base_quantity = max(base_quantity, min_qty)
        
        if 'max_quantity' in quantity_rule:
            max_qty =extract_quantity_value(quantity_rule['max_quantity'])  # FIXED
            base_quantity = min(base_quantity, max_qty)
        if category == 'Biryani' and 'total_items' in quantity_rule.get('adjustments', {}):
            total_items = menu_context['total_items']
            adjusted_qty_str = quantity_rule['adjustments']['total_items'](total_items)
            base_quantity = extract_quantity_value(adjusted_qty_str)
        return {
            'quantity': base_quantity,
            'unit': base_unit,
            'category': category,
            'is_veg': is_veg,
            'adjustments_applied': {
                'variety_count': total_items_in_category,
                'veg_guest_ratio': veg_guest_ratio,
                'meal_type': meal_type
            }
        }
    def apply_dependency_rules(self, category: str, dependent_category: str, current_qty: float, unit: str) -> float:
        normalized_category = self.normalize_category_name(category)
        if normalized_category not in self.category_rules:
            return current_qty
        rule = self.category_rules[normalized_category]
        if "adjustments" not in rule:
            return current_qty
        adjustments = rule["adjustments"]
        dependent_normalized = dependent_category.lower()

        if dependent_normalized == "breads" and "with_breads" in adjustments:
            adjusted_qty_str = adjustments["with_breads"]()
            adjusted_qty =extract_quantity_value(adjusted_qty_str)
            logger.info(f"Category {category}: Adjusted for {dependent_category} from {current_qty:.2f} to {adjusted_qty}")
            return adjusted_qty
        elif dependent_normalized == "rice" and "with_rice" in adjustments:
            adjusted_qty_str = adjustments["with_rice"]()
            adjusted_qty = extract_quantity_value(adjusted_qty_str)
            logger.info(f"Category {category}: Adjusted for {dependent_category} from {current_qty:.2f} to {adjusted_qty}")
            return adjusted_qty
        elif dependent_normalized == "curries" and "with_curry" in adjustments:
            adjusted_qty_str = adjustments["with_curry"]()
            adjusted_qty =extract_quantity_value(adjusted_qty_str)
            logger.info(f"Category {category}: Adjusted for {dependent_category} from {current_qty:.2f} to {adjusted_qty}")
            return adjusted_qty
        elif dependent_normalized == "biryani" and "with_biryani" in adjustments:
            adjusted_qty_str = adjustments["with_biryani"]()
            adjusted_qty = extract_quantity_value(adjusted_qty_str)
            logger.info(f"Category {category}: Adjusted for {dependent_category} from {current_qty:.2f} to {adjusted_qty}")
            return adjusted_qty
        elif dependent_normalized == "dal" and "with_dal" in adjustments:
            adjusted_qty_str = adjustments["with_dal"]()
            adjusted_qty = extract_quantity_value(adjusted_qty_str)
            logger.info(f"Category {category}: Adjusted for {dependent_category} from {current_qty:.2f} to {adjusted_qty}")
            return adjusted_qty
        elif dependent_normalized == "rasam" and "with_rasam" in adjustments:
            adjusted_qty_str = adjustments["with_rasam"]()
            adjusted_qty = extract_quantity_value(adjusted_qty_str)
            logger.info(f"Category {category}: Adjusted for {dependent_category} from {current_qty:.2f} to {adjusted_qty}")
            return adjusted_qty

        return current_qty

    def apply_meal_type_modifier(self, meal_type: str, qty: float) -> float:
        if meal_type in self.meal_type_modifiers:
            modifier = self.meal_type_modifiers[meal_type]
            return qty * modifier
        return qty

    def get_dependent_categories(self, category: str) -> List[str]:
        normalized_category = self.normalize_category_name(category)
        if normalized_category in self.category_dependencies:
            return self.category_dependencies[normalized_category]
        return []