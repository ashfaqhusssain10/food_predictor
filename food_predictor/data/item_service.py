import pandas as pd
import logging
from dataclasses import dataclass
from typing import Dict,Optional,Tuple,Any
from food_predictor.core.category_rules import FoodCategoryRules
from food_predictor.data.item_matcher import FoodItemMatcher
from food_predictor.utils.quantity_utils import extract_unit, infer_default_unit, validate_unit

logger = logging.getLogger("ItemService")

@dataclass
class ItemMetadata:
    """Container for item-specific metadata"""
    category: str
    unit: str
    is_veg: str
    conversion_factor: float = 1.0

class ItemService:
    """Centralizes item metadata management and lookup operations"""
    
    def __init__(self, food_rules):
        """Initialize the item service with required dependencies
        
        Args:
            food_rules: FoodCategoryRules instance for category inference and unit validation
        """
        self.food_rules = food_rules
        self.item_metadata = {}         # Primary metadata store
        self.item_name_mapping = {}     # Standardized name mapper
        self.item_matcher = None        # Will be initialized when metadata is available
        self.item_vc_mapping = {}       # Item-specific VC data
        self.item_specific_data = {}
        self.item_vc_file = r"C:\Users\Syed Ashfaque Hussai\OneDrive\Desktop\CraftMyplate Machine Learning Task\Book1 (1).xlsx"
        self.item_data_file = r"C:\Users\Syed Ashfaque Hussai\OneDrive\Desktop\CraftMyplate Machine Learning Task\item2.csv"
        
    def standardize_item_name(self, item_name: str) -> str:
        """Convert an item name to a standardized representation for consistent matching
        
        Preserves exact functionality of the original implementation while providing
        a single point of maintenance for name standardization logic.
        """
        if pd.isna(item_name):
            return ""
        item_name = str(item_name).strip()
        standardized = item_name.lower()
        standardized = " ".join(standardized.split())  # Normalize whitespace
        return standardized
    
    def build_item_name_mapping(self, data):
        """Construct mapping between standardized and original item names
        
        Args:
            data: DataFrame containing 'Item_name' column
        """
        logger.info("Building item_name mapping")
        self.item_name_mapping = {}
        item_groups = data.groupby(['Item_name'])
        
        for item_name, _ in item_groups:
            # Handle tuple-based item names (preserves original functionality)
            if isinstance(item_name, tuple):
                item_name = item_name[0] if len(item_name) > 0 else ""
            
            std_name = self.standardize_item_name(item_name)
            if std_name:
                self.item_name_mapping[std_name] = item_name
                self.item_name_mapping[item_name] = item_name

        logger.info(f"Built mapping for {len(self.item_name_mapping)} item variations")
    
    def build_item_metadata(self, data):
        """Extract and build comprehensive item metadata from training data
        
        Args:
            data: DataFrame with item information
        """
        logger.info("Building item metadata from training data")
        if not self.item_name_mapping:
            self.build_item_name_mapping(data)
            
        # Process each unique item in the dataset
        item_groups = data.groupby(['Item_name'])
        for item_name, group in item_groups:
            # Normalize tuple-based item names
            if isinstance(item_name, tuple) and len(item_name) > 0:
                item_name = item_name[0]
            item_name = str(item_name)
            
            # Extract veg/non-veg classification with original logic
            is_veg = group['Veg_Classification'].iloc[0]
            if is_veg in ['veg', 'vegetarian']:
                is_veg = "veg"
            elif is_veg in ['non-veg', 'nonveg', 'non veg']:
                is_veg = "non-veg"
            else:
                # Fallback detection based on item name keywords
                item_lower = item_name.lower()
                non_veg_indicators = ["chicken", "mutton", "fish", "prawn", "beef", "pork", "egg", "meat",
                                    "non veg", "kodi", "murg", "lamb", "goat", "seafood", "keema", "crab"]
                is_veg = "non-veg" if any(indicator in item_lower for indicator in non_veg_indicators) else "veg"

            # Extract category and unit information
            category = group['Category'].iloc[0]
            unit = extract_unit(group['Per_person_quantity'].iloc[0])
            
            # Use category-specific rules for default units
            if not unit:
                unit = infer_default_unit(category)
            unit = validate_unit(category, unit)
            
            # Store metadata for both original and standardized names
            self.item_metadata[item_name] = ItemMetadata(category=category, unit=unit, is_veg=is_veg)
            std_name = self.standardize_item_name(item_name)
            if std_name != item_name:
                self.item_metadata[std_name] = ItemMetadata(category=category, unit=unit, is_veg=is_veg)

        logger.info(f"Built metadata for {len(self.item_metadata)} items")
        
        # Initialize the matcher with built metadata
        self.initialize_matcher()
    
    def initialize_matcher(self):
        """Initialize the two-tier hash table matcher with current metadata"""
        self.item_matcher = FoodItemMatcher(self.item_metadata)
        
        # Add direct lookups from the name mapping
        for std_name, item_name in self.item_name_mapping.items():
            self.item_matcher.direct_lookup[std_name] = item_name
            
        logger.info("Initialized FoodItemMatcher with item metadata")
    
    def find_item(self, item_name):
        """Find an item using the optimal matching strategy
        
        Tries exact match first, then uses FoodItemMatcher for fuzzy matching.
        
        Args:
            item_name: Item name to look up
            
        Returns:
            Tuple of (matched_item_name, ItemMetadata) or (None, None) if no match
        """
        # Handle '>' prefix automatically
        clean_item_name = item_name.replace('> ', '').replace('>', '') if isinstance(item_name, str) else item_name
        
        # First try FoodItemMatcher if available
        if self.item_matcher:
            matched_item, metadata = self.item_matcher.find_item(clean_item_name, self.item_metadata)
            if matched_item:
                logger.debug(f"FoodItemMatcher matched '{clean_item_name}' to '{matched_item}'")
                return matched_item, metadata

        # Fall back to standard lookup
        std_name = self.standardize_item_name(clean_item_name)
        if std_name in self.item_name_mapping:
            original_name = self.item_name_mapping[std_name]
            if original_name in self.item_metadata:
                return original_name, self.item_metadata[original_name]
            
        # Check for direct match with original name
        if clean_item_name in self.item_metadata:
            return clean_item_name, self.item_metadata[clean_item_name]
            
        # No match found
        return None, None
    def guess_item_category(self, item_name):
        """Infer category for an unknown item based on keywords
        
        Args:
            item_name: Item name to categorize
            
        Returns:
            String category name
        """
        item_lower = item_name.lower()
        category_keywords = {
            "Welcome_Drinks": ["punch", "Packed Juice","fresh fruit juice", "juice", "mojito", "drinks","milk", "tea", "coffee", "juice", "butter milk", "lassi", "soda",  "water melon juice"],
            "Appetizers": ["tikka", "65","Sauteed Grilled Chicken Sausage", "paneer","Fish Fingers","Mashed Potatos","Cheese Fries","french fires","Potato Skins","Pepper Chicken (Oil Fry)","Lemon Chicken","kabab", "hariyali kebab", "tangdi", "drumsticks", "nuggets","majestic","roll", "poori", "masala vada", "alasanda vada", "veg bullets", "veg spring rolls", "hara bara kebab", "kebab", "lollipop", "chicken lollipop", "pakora", "kodi", "cut", "bajji", "vepudu", "roast", "kurkure", "afghani kebab", "corn", "manchuria", "manchurian", "gobi","jalapeno pop up","Chilli Garlic "],
            "Soups": ["soup", "shorba","mutton marag","broth","cream of chicken","paya","hot and sour",],
            "Fried": ["fried rice"],
            "Italian": ["pasta", "noodles", "white pasta", "veg garlic soft noodles","macroni"],
            "Fry": ["fry", "bendi kaju","Dondakaya Fry","Bhindi Fry","Aloo Fry","Cabbage Fry"],
            "Liquids(Less_Dense)": ["rasam","Pachi pulusu","Sambar", "charu", "majjiga pulusu", "Miriyala Rasam","chintapandu rasam","lemon pappucharu","mulakaya pappucharu"],
            "Liquids(High_Dense)": ["ulavacharu"],
            "Curries": ["iguru","Paneer Butter Masala","Chicken Chettinad","gutti vankaya curry","kadai","scrambled egg curry","baigan bartha","bendi do pyaza","boiled egg cury","chana masala","curry", "gravy", "masala", "kurma", "butter","pulusu","mutton rogan josh curry","kadai", "tikka masala", "dal tadka", "boti", "murgh", "methi", "bhurji", "chatapata", "pulsu", "vegetable curry", "dum aloo curry"],
            "Rice": ["steamed rice", "kaju ghee rice", "bagara rice"],
            "Flavored_Rice": ["muddapappu avakai annam","Ragi Sangati", "pudina rice","temple style pulihora", "pappucharu annam","pappu charu annam","cocount rice","cocunut rice","pulihora", "curd rice", "jeera rice", "gongura rice", "Muddapappu Avakaya Annam", "sambar rice", "Muddapappu avakai Annam", "annam"],
            "Pulav": ["pulav","Mutton Fry Piece Pulav","Natukodi Pulav","jeera mutter pulav", "fry piece pulav", "ghee pulav","Green Peas Pulav","Meal Maker Pulav","Paneer Pulav"],
            "Biryani": ["biryani", "biriyani", "Mutton Kheema Biryani","biriani", "panaspottu biryani","egg biryani","chicken chettinad biryani","ulavacharu chicken biryani", "mushroom biryani", "veg biryani", "chicken dum biryani"],
            "Breads": ["naan", "paratha", "kulcha", "pulka", "chapati", "rumali roti","laccha paratha","masala kulcha","panner kulcha","butter garlic naan","roti,pudina naan","tandoori roti"],
            "Dal": ["dal", "lentil", "pappu", "Mamidikaya Pappu (Mango)","Dal Makhani","Dal Tadka","sorakaya pappu", "thotakura pappu", "tomato pappu", "yellow dal""chintakaya papu","palakura pappu","thotakura pappu","tomato pappu","yellow dal tadka"],
            "Chutney":["peanut chutney","allam chutney","green chutney","pudina chutney","dondakay chutney"],
            "Ghee": ["ghee"],
            "Podi": ["podi"],
            "Pickle": ["pickle"],
            "Paan": ["paan", "pan"],
            "Dips": ["dip","Sour cream Dip","jam","Tandoori Mayo","Mayonnaise Dip","Hummus Dip","Garlic Mayonnaise Dip"],
            "Roti_Pachadi": ["Beerakaya Pachadi", "roti pachadi", "Tomato Pachadi","Vankaya Pachadi","Roti Pachadi", "pachadi","gongura onion chutney"],
            "Crispers": ["fryums", "papad", "crispers"],
            "Raitha": ["raitha", "Raitha", "boondi raitha"],
            "Salan": ["salan", "Salan", "Mirchi Ka Salan"],
            "Fruits": ["seaonsal", "mixed", "cut", "fruit", "Fruit"],
            "Salad": ["salad", "Salad", "ceasar", "green salad", "boiled peanut salad","boiled peanut salad","mexican corn salad","Laccha Pyaaz","Cucumber Salad"],
            "Curd": ["curd", "set curd"],
            "Desserts": ["brownie", "walnut brownie", "Gajar Ka Halwa","Chocolate Brownie","Assorted Pastries","halwa","Semiya Payasam (Kheer)","Sabudana Kheer","Kesari Bath","Double Ka Meetha", "carrot halwa", "shahi ka tukda", "gulab jamun", "apricot delight", "baked gulab jamun", "bobbattu", "bobbatlu", "kalajamun", "rasagulla", "laddu", "poornam", "apricot delight", "gulab jamun", "rasammaiah"],
            "Breakfast": ["idly", "dosa", "vada", "upma","Rava Khichdi","Bisi Bela Bath","Sabudana Khichdi","Upma","Millet Upma", "pongal", "mysore bonda", "idly"],
            "Sandwich": ["sandwich","Veg Sandwitch"],
            "Cup_Cakes": ["cup cake", "cupcake"]
        }
        for category, keywords in category_keywords.items():
            if item_lower in keywords:
                #logger.info(f"Categorized '{item_name}' as '{category}' based on exact match")
                return category
                
        # Then check for partial matches
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in item_lower:
                    #logger.info(f"Categorized '{item_name}' as '{category}' based on keyword '{keyword}'")
                    return category
                    
        # Default fallback
        #logger.warning(f"Using default category 'Curries' for '{item_name}'")
        return "Curries"            
    def determine_item_properties(self, item_name):
        """Determine comprehensive properties for a given item
        
        Args:
            item_name: Name of the food item
            
        Returns:
            Dictionary with comprehensive item properties
        """
        # Handle '>' prefix automatically
        clean_item_name = item_name.replace('> ', '').replace('>', '') if isinstance(item_name, str) else item_name

        # Initialize with defaults
        properties = {
            "category": None,
            "unit": "g",
            "is_veg": "veg",
            "quantity_rule": None,
            "category_default_qty": None,
            "category_adjustments": None
        }

        # Try to find the item in metadata
        matched_item, metadata = self.find_item(clean_item_name)
        
        if matched_item and metadata:
            # Item found in metadata
            category = metadata.category
            
            # Retrieve category-specific rules
            normalized_category = self.food_rules.normalize_category_name(category)
            
            if normalized_category in self.food_rules.category_rules:
                rule = self.food_rules.category_rules[normalized_category]
                
                properties.update({
                    "category": category,
                    "unit": metadata.unit,
                    "is_veg": metadata.is_veg,
                    "quantity_rule": rule,
                    "category_default_qty": rule.get("default_quantity", "100g"),
                    "category_adjustments": rule.get("adjustments", {})
                })
                
                return properties
        
        # Fallback: Guess category and infer properties
        item_lower = clean_item_name.lower() if isinstance(clean_item_name, str) else ""
        properties["category"] = self.guess_item_category(clean_item_name)
        
        # Retrieve category-specific rules for guessed category
        normalized_category = self.food_rules.normalize_category_name(properties["category"])
        
        if normalized_category in self.food_rules.category_rules:
            rule = self.food_rules.category_rules[normalized_category]
            
            properties.update({
                "quantity_rule": rule,
                "category_default_qty": rule.get("default_quantity", "100g"),
                "category_adjustments": rule.get("adjustments", {})
            })

        # Veg/Non-veg detection
        non_veg_indicators = ["chicken", "mutton", "fish", "prawn", "beef", "pork", "egg", "meat",
                            "non veg", "kodi", "murg", "lamb", "goat", "seafood", "keema", "crab"]
        if any(indicator in item_lower for indicator in non_veg_indicators):
            properties["is_veg"] = "non-veg"

        return properties

    def load_item_specific_data(self, item_data_file):
        logger.info(f"Loading item-specific data from {item_data_file}")
        try:
            item_data = pd.read_csv(item_data_file)
            required_columns = ['item_name', 'category', 'preferred_unit', 'per_guest_ratio', 'base_price_per_piece',
                                'base_price_per_kg']
            missing_columns = [col for col in required_columns if col not in item_data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in {item_data_file}: {missing_columns}")

            # Debug: Print first few rows of the dataframe
            # logger.info(f"Item data preview: {item_data.head()}")

            # # Debug: Count of items by preferred_unit
            # unit_counts = item_data['preferred_unit'].value_counts()
            # logger.info(f"Units distribution in item_data.csv: {unit_counts}")

            # Debug: Count of non-null values in pricing columns
            # non_null_piece_prices = item_data['base_price_per_piece'].notna().sum()
            # non_null_kg_prices = item_data['base_price_per_kg'].notna().sum()
            # logger.info(
            #     f"Items with base_price_per_piece: {non_null_piece_prices}, Items with base_price_per_kg: {non_null_kg_prices}")

            for _, row in item_data.iterrows():
                item_name = self.standardize_item_name(row['item_name'])

                # Debug: For specific items of interest
                # if 'papad' in item_name.lower() or 'curd' in item_name.lower():
                #     logger.info(f"Found special item in item_data.csv: {item_name}")
                #     logger.info(f"  - preferred_unit: {row['preferred_unit']}")
                #     logger.info(f"  - base_price_per_piece: {row['base_price_per_piece']}")
                #     logger.info(f"  - base_price_per_kg: {row['base_price_per_kg']}")

                self.item_specific_data[item_name] = {
                    'category': row['category'],
                    'preferred_unit': row['preferred_unit'],
                    'per_guest_ratio': float(row['per_guest_ratio']) if pd.notna(row['per_guest_ratio']) else None,
                    'base_price_per_piece': float(row['base_price_per_piece']) if pd.notna(
                        row['base_price_per_piece']) else None,
                    'base_price_per_kg': float(row['base_price_per_kg']) if pd.notna(row['base_price_per_kg']) else None
                }

            logger.info(f"Loaded item-specific data for {len(self.item_specific_data)} items")

            # Debug: Check if key items are in the final dictionary
            # for test_item in ['papad', 'curd', 'chicken oil fry']:
            #     test_item_std = self.standardize_item_name(test_item)
            #     if test_item_std in self.item_specific_data:
            #         item_data = self.item_specific_data[test_item_std]
            #         logger.info(f"Item '{test_item}' found in item_specific_data with:")
            #         logger.info(f"  - preferred_unit: {item_data['preferred_unit']}")
            #         logger.info(f"  - base_price_per_piece: {item_data['base_price_per_piece']}")
            #         logger.info(f"  - base_price_per_kg: {item_data['base_price_per_kg']}")
                # else:
                #     logger.warning(f"Item '{test_item}' NOT found in item_specific_data")

        except Exception as e:
            logger.error(f"Failed to load item-specific data: {e}")
            raise

    def load_item_vc_data(self, item_vc_file):
        """Load item VC (variable cost) data from Excel file
        
        Args:
            item_vc_file: Path to Excel file containing VC data
        """
        logger.info(f"Loading item VC data from {item_vc_file}")
        try:
            vc_data = pd.read_excel(item_vc_file)

            # Process records efficiently
            for _, row in vc_data.iterrows():
                item_name = self.standardize_item_name(row['Item_name'])
                self.item_vc_mapping[item_name] = {
                    'VC': float(row['VC']),
                    'p_value': float(row.get('Power Factor (p)', 0.18))
                }
                
            logger.info(f"Loaded VC and P_value data for {len(self.item_vc_mapping)} items")

        except Exception as e:
            logger.error(f"Failed to load item VC data: {e}")
            raise