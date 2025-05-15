#identify the main items and Menu context
import logging
import requests
import json
from typing import Dict,List,Any,Optional,Tuple
from food_predictor.data.item_service import ItemService,ItemMetadata
from food_predictor.core.category_rules import FoodCategoryRules

logger = logging.getLogger("MenuAnalyzer")
import time
from threading import Lock

class GeminiRateLimiter:
    def __init__(self, max_calls_per_second):
        self.max_calls = max_calls_per_second
        self.min_interval = 1.0 / max_calls_per_second
        self.last_call = 0.0
        self.lock = Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            wait_time = self.min_interval - (now - self.last_call)
            if wait_time > 0:
                time.sleep(wait_time)
            self.last_call = time.time()

class MenuAnalyzer:
    """
    Analyzes menu compositions, builds comprehensive menu contexts,
    and identifies main items using semantic analysis.s
    """
    
    def __init__(self, food_rules, item_service):
        """
        Initialize the menu analyzer with required dependencies.
        
        Args:
            food_rules: FoodCategoryRules instance for category-specific rules
            item_service: ItemService for item metadata and property resolution
        """
        self.food_rules = food_rules
        self.item_service = item_service
        self.gemini_rate_limiter = GeminiRateLimiter(max_calls_per_second=10)  # You can reduce to 5 if needed

        # API credentials for main item identification
        self.GEMINI_API_KEY = "AIzaSyCtSNFgv474qPIJPHipmip0Lavi3T-k8rw"
        self.GEMINI_API_URL = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-1.5-flash-latest:generateContent?key={self.GEMINI_API_KEY}")
        
    def build_menu_context(self, event_time, meal_type, event_type, total_guest_count, veg_guest_count, selected_items):
        """
        Build comprehensive menu context.
        
        Args:
            event_time (str): Time of the event
            meal_type (str): Type of meal
            event_type (str): Type of event
            total_guest_count (int): Total number of guests
            veg_guest_count (int): Number of vegetarian guests
            selected_items (list): List of selected menu items
        
        Returns:
            dict: Comprehensive menu context
        """
        non_veg_guest_count = total_guest_count - veg_guest_count 
        
        # Initialize context
        menu_context = {
            'categories': [],
            'items': selected_items,
            'total_items': len(selected_items),
            'meal_type': meal_type,
            'event_time': event_time,
            'event_type': event_type,
            'total_guest_count': total_guest_count,
            'veg_guest_count': veg_guest_count,
            'non_veg_guest_count': non_veg_guest_count,
            'items_by_category': {},
            'veg_items_by_category': {},
            'non_veg_items_by_category': {},
            'item_properties': {},
            'category_context': {}
        }
        
        # Process each item
        for item in selected_items:
            # Get comprehensive item properties
            properties = self.item_service.determine_item_properties(item)
            category = properties["category"]
            is_veg = properties.get("is_veg", "veg")
            
            # Store item properties
            menu_context['item_properties'][item] = properties
            
            # Organize by category
            if category not in menu_context['items_by_category']:
                menu_context['items_by_category'][category] = []
                menu_context['categories'].append(category)
            menu_context['items_by_category'][category].append(item)
            
            # Veg/Non-veg categorization
            if is_veg == "veg":
                if category not in menu_context['veg_items_by_category']:
                    menu_context['veg_items_by_category'][category] = []
                menu_context['veg_items_by_category'][category].append(item)
            else:
                if category not in menu_context['non_veg_items_by_category']:
                    menu_context['non_veg_items_by_category'][category] = []
                menu_context['non_veg_items_by_category'][category].append(item)
            
            # Build category-level context
            if category not in menu_context['category_context']:
                menu_context['category_context'][category] = {
                    'total_items': 0,
                    'veg_items': 0,
                    'non_veg_items': 0,
                    'category_rules': properties.get('quantity_rule', {}),
                    'default_quantity': properties.get('category_default_qty', '100g')
                }
            
            # Update category context
            cat_context = menu_context['category_context'][category]
            cat_context['total_items'] += 1
            cat_context['veg_items'] += 1 if is_veg == "veg" else 0
            cat_context['non_veg_items'] += 1 if is_veg == "non-veg" else 0
        self.menu_context = menu_context
        return menu_context
    
    def identify_main_items(self, menu_context,meal_type,categories):
        """
        Identify main items using the Gemini API, with sophisticated veg/non-veg segregation.

        Args:
            menu_context: Comprehensive menu context dictionary

        Returns:
            Detailed dictionary of main items across vegetarian and non-vegetarian categories
        """
        # Initialize result structure
        if getattr(self, '_evaluation_mode', False):
            logger.info("Evaluation mode: using fallback instead of Gemini")
            return self._identify_main_items_fallback(menu_context, meal_type, categories)

                
        
        
        main_items_info = {
            'has_main_item': False,
            'primary_main_category': None,
            'primary_main_item': None,
            'categories_found': [],
            'veg': {
                'categories_found': [],
                'items_by_category': {},
                'primary_main_category': None,
                'primary_main_item': None
            },
            'non_veg': {
                'categories_found': [],
                'items_by_category': {},
                'primary_main_category': None,
                'primary_main_item': None
            }
        }

        # Only proceed with Gemini API if credentials are available
        if self.GEMINI_API_KEY and self.GEMINI_API_URL:
            try:
                #
                # Prepare the prompt
                selected_items = menu_context['items']
                prompt =    (
                    "You are a culinary expert. A main item is defined as any dish that guests "
                    "typically take in full-plate or ladle portions,and that contributes significantly to the total food volume."
                    "Do not limit your output to just one main item — multiple dishes may qualify if they each contribute significantly to food volume.\n\n"
                    "only pick items that are major dishes -meaning large/high-volume/full-plate consumption items.\n\n"
                    "Ignore the condiments like salads, pickles, raita,salan,desserts, "
                    "and beverages—they are not main items under this definition. \n\n"
                    f"The Menu is for a  {meal_type} event .\n"
                    f"The following broad categories are present in the menu: {', '.join(categories)}.\n"
                    "Respond *strictly* as a JSON array.  Each element must have: \n"
                    "  • 'main_item' - the item name exactly as given\n"
                    "If no main items fit the high-consumption definition, return an empty list []. \n\n"
                    "Menu items:\n"
                    f"{', '.join(selected_items)}"
                    
                    )   
               # Make API request
                headers = {
                    "Content-Type": "application/json"
                }
                payload = {
                    "contents": [{
                        "parts": [{
                            "text": prompt
                        }]
                    }]
                }
                self.gemini_rate_limiter.wait()
                response = requests.post(
                    self.GEMINI_API_URL,
                    headers={"Content-Type": "application/json"},
                    json=payload
                )
                response.raise_for_status()

                # Parse response
                response_data = response.json()
                if isinstance(response_data, list):
                    response_data = response_data[0]  
                cand = response_data.get("candidates", [{}])[0]

                # Prefer the "output" field (where Gemini puts your JSON), fallback to old content/parts
                if "output" in cand:
                    generated_text = cand["output"].strip()
                else:
                    generated_text = (
                        cand
                        .get("content", {})
                        .get("parts", [{}])[0]
                        .get("text", "")
                    ).strip()

                # Clean and parse JSON response
                if generated_text.startswith("```"):
                    lines=generated_text.splitlines()
                    if lines [0].startswith("```"):
                        lines=lines[1:]
                    # drop the last line if it's a fence
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    generated_text = "\n".join(lines).strip()
               
                try:
                    api_result = json.loads(generated_text)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse Gemini API response as JSON: {generated_text}, error: {e}")
                    api_result = []

                # Process API results
                if api_result:
                    main_items_info['has_main_item'] = True
                    for item in api_result:
                        main_item = item['main_item']
            
                        
                        # Validate that the main item exists in selected_items
                        if main_item in selected_items:
                            # Determine veg/non-veg status
                            item_properties = menu_context['item_properties'].get(main_item, {})
                            is_veg = item_properties.get('is_veg', 'veg')
                            category = item_properties.get('category')
                            # Update main items info
                            main_items_info['categories_found'].append(category)
                            
                            if is_veg == 'veg':
                                main_items_info['veg']['categories_found'].append(category)
                                if category not in main_items_info['veg']['items_by_category']:
                                    main_items_info['veg']['items_by_category'][category] = []
                                
                                main_items_info['veg']['items_by_category'][category].append(main_item)
                                if not main_items_info['veg']['primary_main_category']:
                                    main_items_info['veg']['primary_main_category'] = category
                                    main_items_info['veg']['primary_main_item'] = main_item
                            else:
                                main_items_info['non_veg']['categories_found'].append(category)
                                if category not in main_items_info['non_veg']['items_by_category']:
                                    main_items_info['non_veg']['items_by_category'][category] = []
                                main_items_info['non_veg']['items_by_category'][category].append(main_item)
                                if not main_items_info['non_veg']['primary_main_category']:
                                    main_items_info['non_veg']['primary_main_category'] = category
                                    main_items_info['non_veg']['primary_main_item'] = main_item

                            # Set overall primary if not set
                            if not main_items_info['primary_main_category']:
                                main_items_info['primary_main_category'] = category
                                main_items_info['primary_main_item'] = main_item

                            logger.info(f"Gemini API identified main item: {main_item} (category: {category}, is_veg: {is_veg})")
                        else:
                            logger.warning(f"Gemini API returned main item '{main_item}' not in selected_items: {selected_items}")

            except requests.RequestException as e:
                logger.error(f"Gemini API request failed: {e}")
                # Fall back to default logic
                main_items_info = self._identify_main_items_fallback(menu_context,meal_type,categories)
        else:
            # Use fallback if no API credentials
            main_items_info = self._identify_main_items_fallback(menu_context,meal_type,categories)

        return main_items_info

    def _identify_main_items_fallback(self, menu_context,meal_type,categories):
        """
        Fallback method for identifying main items when the API is unavailable.
        Uses hardcoded category list to identify main items.
        """
        # Start with the same structure as the main method
        main_items_info = {
            'has_main_item': False,
            'primary_main_category': None,
            'primary_main_item': None,
            'categories_found': [],
            'veg': {
                'categories_found': [],
                'items_by_category': {},
                'primary_main_category': None,
                'primary_main_item': None
            },
            'non_veg': {
                'categories_found': [],
                'items_by_category': {},
                'primary_main_category': None,
                'primary_main_item': None
            }
        }
        
        # List of categories that are likely to contain main items
        main_item_categories = [
            "Biryani", "Rice", "Flavored_Rice", "Curries", 
            "Fried_Rice", "Breads", "Italian", "Breakfast", 
            "Breakfast_pcs"
        ]
        
        # Iterate through menu items to find main items
        for category in main_item_categories:
            if category in menu_context['items_by_category']:
                main_items_info['has_main_item'] = True
                main_items_info['categories_found'].append(category)
                
                # Process items in this category
                for item in menu_context['items_by_category'][category]:
                    # Get item properties 
                    item_properties = menu_context['item_properties'].get(item, {})
                    is_veg = item_properties.get('is_veg', 'veg')
                    
                    # Store in appropriate veg/non-veg section
                    if is_veg == 'veg':
                        if category not in main_items_info['veg']['categories_found']:
                            main_items_info['veg']['categories_found'].append(category)
                        
                        if category not in main_items_info['veg']['items_by_category']:
                            main_items_info['veg']['items_by_category'][category] = []
                        
                        main_items_info['veg']['items_by_category'][category].append(item)
                        
                        # Set as primary if not already set
                        if not main_items_info['veg']['primary_main_category']:
                            main_items_info['veg']['primary_main_category'] = category
                            main_items_info['veg']['primary_main_item'] = item
                    else:
                        if category not in main_items_info['non_veg']['categories_found']:
                            main_items_info['non_veg']['categories_found'].append(category)
                            
                        if category not in main_items_info['non_veg']['items_by_category']:
                            main_items_info['non_veg']['items_by_category'][category] = []
                            
                        main_items_info['non_veg']['items_by_category'][category].append(item)
                        
                        # Set as primary if not already set
                        if not main_items_info['non_veg']['primary_main_category']:
                            main_items_info['non_veg']['primary_main_category'] = category
                            main_items_info['non_veg']['primary_main_item'] = item
                    
                    # Set overall primary if not set
                    if not main_items_info['primary_main_category']:
                        main_items_info['primary_main_category'] = category
                        main_items_info['primary_main_item'] = item
                    
                    logger.debug(f"Fallback identified main item: {item} (category: {category}, is_veg: {is_veg})")
        
        return main_items_info