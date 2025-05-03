# core/feature_engineering.py
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from food_predictor.core.menu_analyzer import MenuAnalyzer

from food_predictor.core.category_rules import FoodCategoryRules
from food_predictor.data.item_service import ItemService
from food_predictor.utils.quantity_utils import extract_quantity_value,extract_unit,validate_unit,infer_default_unit


logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Handles feature extraction, transformation, and engineering for food prediction models.
    
    This class is responsible for:
    1. Encoding categorical variables
    2. Extracting menu context features
    3. Handling inter-category dependencies
    4. Scaling numerical features
    """
    
    def __init__(self,food_rules= None):
        """Initialize feature engineering components."""
        self.event_time_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.meal_type_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.event_type_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.guest_scaler = None
        self.ratio_feature_scaler = None
        self.std_feature_scaler = None
        self.network_feature_scaler = None
        self.transform_feature_scaler = None
        self.raw_qty_feature_scaler = None
        self.menu_feature_scaler = None
        self.food_rules = food_rules
    def fit_encoders(self, data):
        """
        Fit categorical encoders on training data.
        
        Args:
            data (DataFrame): Training data with categorical columns
        """
        self.event_time_encoder.fit(data[['Event_Time']])
        self.meal_type_encoder.fit(data[['Meal_Time']])
        self.event_type_encoder.fit(data[['Event_Type']])
        
    def prepare_features(self, data, menu_features=None):
        """
        Prepare feature matrix with enhanced dependency features.
        
        Args:
            data (DataFrame): Input data
            menu_features (dict, optional): Menu context features
            
        Returns:
            ndarray: Feature matrix
        """
        # 1) encode the three categorical columns (all return arrays of shape (n_rows, n_encoded))
        event_time_encoded = self.event_time_encoder.transform(data[['Event_Time']])
        meal_type_encoded = self.meal_type_encoder.transform(data[['Meal_Time']])
        event_type_encoded = self.event_type_encoder.transform(data[['Event_Type']])
        
        base_features = np.hstack([
            event_time_encoded,
            meal_type_encoded,
            event_type_encoded
        ])  # shape = (n_rows, sum_of_encoded_dims)

        features = base_features

        # 2) Add guest count if present
        if 'Guest_Count' in data.columns:
            guest_count = data[['Guest_Count']].values.astype(float)
            if not self.guest_scaler:
                self.guest_scaler = RobustScaler()
                self.guest_scaler.fit(guest_count)
            guest_scaled = self.guest_scaler.transform(guest_count)
            features = np.hstack([features, guest_scaled])

        # 3) Add menu context features (broadcasted to every row)
        if menu_features is not None:
            # Determine number of rows for broadcasting
            n_rows = features.shape[0]
            
            # --- 3.1. FEATURE TYPE CATEGORIZATION ---
            # Separate features by type to apply appropriate transformations
            
            # 1. Ratio-based features require minimal transformation to preserve mathematical relationships
            ratio_features = [col for col in menu_features.keys() 
                            if ('ratio' in col or 
                                'multiplier' in col or 
                                'scale_factor' in col)]
            
            # 2. Binary features (0/1) should remain untransformed
            binary_features = [col for col in menu_features.keys() 
                            if (col.startswith('is_') or 
                                col.startswith('has_') or 
                                '_present' in col or 
                                '_both_present' in col)]
            
            # Network complexity features for dependency hubs
            network_features = [col for col in menu_features.keys()
                            if ('network_complexity' in col or 
                                'second_order' in col or 
                                '_presence_ratio' in col)]
            
            # Interaction features for context-sensitive categories
            interaction_features = [col for col in menu_features.keys()
                                if ('_interaction' in col)]
            
            # Transformed quantity features for small-quantity items
            transformed_qty_features = [col for col in menu_features.keys()
                                    if ('log_qty' in col or 
                                        'sqrt_qty' in col)]
            
            # Raw quantity features (preserve without centering)
            raw_qty_features = [col for col in menu_features.keys()
                            if 'raw_qty' in col]
            
            # 3. Count-based features benefit from robust scaling
            count_features = [col for col in menu_features.keys() 
                            if ('count' in col or 
                                'variety' in col or 
                                'items_per' in col)]
            
            # 4. Remaining features get standard robust scaling
            standard_features = [col for col in menu_features.keys() 
                                if col not in ratio_features + 
                                            binary_features + 
                                            count_features]
            
            # Track which feature matrices we'll need to combine
            feature_matrices = []
            
            # --- 3.2. PROCESS RATIO FEATURES ---
            if ratio_features:
                ratio_values = [menu_features.get(col, 0.0) for col in ratio_features]
                ratio_matrix = np.array([ratio_values])
                
                # Apply minimal scaling - no centering to preserve multiplier semantics
                if not self.ratio_feature_scaler:
                    self.ratio_feature_scaler = RobustScaler(with_centering=False, 
                                                            with_scaling=True,
                                                            quantile_range=(10.0, 90.0))
                    self.ratio_feature_scaler.fit(ratio_matrix)
                
                # Transform and broadcast to all rows
                ratio_scaled = self.ratio_feature_scaler.transform(ratio_matrix)
                ratio_repeated = np.repeat(ratio_scaled, n_rows, axis=0)
                feature_matrices.append(ratio_repeated)
                
                # Log ratio feature details for debugging
                logger.debug(f"Processed {len(ratio_features)} ratio features")
            
            # --- 3.3. PROCESS BINARY FEATURES ---
            if binary_features:
                binary_values = [float(menu_features.get(col, 0.0)) for col in binary_features]
                binary_matrix = np.array([binary_values])
                
                # No transformation needed, just broadcast to all rows
                binary_repeated = np.repeat(binary_matrix, n_rows, axis=0)
                feature_matrices.append(binary_repeated)
                
                logger.debug(f"Processed {len(binary_features)} binary features")
            
            # --- 3.4. PROCESS NETWORK FEATURES ---
            if network_features:
                network_values = [menu_features.get(col, 0.0) for col in network_features]
                network_matrix = np.array([network_values])
                
                if not self.network_feature_scaler:
                    # Use robust scaling without centering to preserve network structure
                    self.network_feature_scaler = RobustScaler(with_centering=False, 
                                                            with_scaling=True,
                                                            quantile_range=(5.0, 95.0))
                    self.network_feature_scaler.fit(network_matrix)
                
                network_scaled = self.network_feature_scaler.transform(network_matrix)
                network_repeated = np.repeat(network_scaled, n_rows, axis=0)
                feature_matrices.append(network_repeated)
            
            # --- 3.5. PROCESS INTERACTION FEATURES ---
            if interaction_features:
                interaction_values = [menu_features.get(col, 0.0) for col in interaction_features]
                interaction_matrix = np.array([interaction_values])
                interaction_repeated = np.repeat(interaction_matrix, n_rows, axis=0)
                feature_matrices.append(interaction_repeated)
            
            # --- 3.6. PROCESS TRANSFORMED QUANTITY FEATURES ---
            if transformed_qty_features:
                transform_values = [menu_features.get(col, 0.0) for col in transformed_qty_features]
                transform_matrix = np.array([transform_values])
                
                # These already have non-linear transformation but benefit from light scaling
                if not self.transform_feature_scaler:
                    self.transform_feature_scaler = RobustScaler(with_centering=False,
                                                            with_scaling=True)
                    self.transform_feature_scaler.fit(transform_matrix)
                
                transform_scaled = self.transform_feature_scaler.transform(transform_matrix)
                transform_repeated = np.repeat(transform_scaled, n_rows, axis=0)
                feature_matrices.append(transform_repeated)
            
            # --- 3.7. PROCESS RAW QUANTITY FEATURES ---
            if raw_qty_features:
                raw_qty_values = [menu_features.get(col, 0.0) for col in raw_qty_features]
                raw_qty_matrix = np.array([raw_qty_values])
                
                if not self.raw_qty_feature_scaler:
                    self.raw_qty_feature_scaler = RobustScaler(with_centering=False,
                                                            with_scaling=True)
                    self.raw_qty_feature_scaler.fit(raw_qty_matrix)
                
                raw_qty_scaled = self.raw_qty_feature_scaler.transform(raw_qty_matrix)
                raw_qty_repeated = np.repeat(raw_qty_scaled, n_rows, axis=0)
                feature_matrices.append(raw_qty_repeated)
                
            # --- 3.8. PROCESS COUNT AND STANDARD FEATURES ---
            combined_standard_features = count_features + standard_features
            if combined_standard_features:
                std_values = [menu_features.get(col, 0.0) for col in combined_standard_features]
                std_matrix = np.array([std_values])
                
                # Apply full robust scaling with centering and scaling
                if not self.std_feature_scaler:
                    self.std_feature_scaler = RobustScaler(with_centering=True, 
                                                        with_scaling=True)
                    self.std_feature_scaler.fit(std_matrix)
                
                # Transform and broadcast to all rows
                std_scaled = self.std_feature_scaler.transform(std_matrix)
                std_repeated = np.repeat(std_scaled, n_rows, axis=0)
                feature_matrices.append(std_repeated)
                
                logger.debug(f"Processed {len(combined_standard_features)} standard features " 
                            f"({len(count_features)} count, {len(standard_features)} other)")
            
            # --- 3.9. COMBINE ALL MENU FEATURE MATRICES ---
            if feature_matrices:
                menu_features_combined = np.hstack(feature_matrices)
                
                # Add to our existing feature matrix
                features = np.hstack([features, menu_features_combined])
                
                # Log overall feature dimensions
                logger.debug(f"Added {menu_features_combined.shape[1]} menu features to create "
                            f"final feature matrix of shape {features.shape}")
        
        return features
        
    def extract_menu_features(self, data, menu_context, food_rules=None):
        """
        Extract rich menu complexity features from existing menu context.
        
        Args:
            data (DataFrame): Input data
            menu_context (dict): Menu context information
            food_rules (object, optional): Food category rules
            
        Returns:
            dict: Extracted menu features
        """
        features = {}
        
        # Extract required variables from menu_context
        meal_type = menu_context.get('meal_type', '')
        event_time = menu_context.get('event_time', '')
        
        # Menu structure metrics
        features['category_count'] = len(menu_context['categories'])
        features['total_items'] = len(menu_context['items'])
        features['items_per_category'] = features['total_items'] / max(features['category_count'], 1)
        
        # Category distribution
        category_item_counts = [len(items) for cat, items in menu_context['items_by_category'].items()]
        total_items = sum(category_item_counts)
        category_ratios = [count/total_items for count in category_item_counts if total_items > 0]
        
        # Shannon entropy (menu diversity measure)
        features['category_entropy'] = -sum([r * np.log(r) for r in category_ratios if r > 0])
        
        # Update guest count column handling with column aliasing
        if 'total_guest_count' in data.columns:
            guest_count = data[['total_guest_count']].values.astype(float)
        elif 'Guest_Count' in data.columns:
            guest_count = data[['Guest_Count']].values.astype(float)
        else:
            guest_count = None
        
        if guest_count is not None:
            # Guest-related metrics
            features['total_guests'] = menu_context['total_guest_count']
            features['veg_ratio'] = menu_context['veg_guest_count'] / max(features['total_guests'], 1)
            features['items_per_guest'] = features['total_items'] / max(features['total_guests'], 1)
            
        # Main course presence
        main_categories = ["Biryani", "Rice", "Pulav", "Flavored_Rice"]
        features['has_main_dish'] = int(any(cat in menu_context['categories'] for cat in main_categories))
        features['main_dish_count'] = sum(1 for cat in menu_context['categories'] if cat in main_categories)
        
        # Veg/non-veg composition
        veg_items = sum(1 for item in menu_context['items'] if menu_context['item_properties'][item].get('is_veg', 'veg') == 'veg')
        features['veg_item_ratio'] = veg_items / max(features['total_items'], 1)
        
        # Extract additional features if food_rules is provided
        if food_rules:
            # --- Intra‑category dependencies (variety counts) ---
            for cat, ctx in menu_context['category_context'].items():
                norm_cat = food_rules.normalize_category_name(cat)
                features[f"{norm_cat}__variety_count"] = ctx['total_items']
                if norm_cat in food_rules.category_rules:
                    rule = food_rules.category_rules[norm_cat]
                
                    # Explicit variety-based quantity scaling factor
                    if "adjustments" in rule and "variety_count" in rule["adjustments"]:
                        try:
                            # Base quantity from rule
                            base_qty_str = rule.get("default_quantity", "100g")
                            base_qty = extract_quantity_value(base_qty_str)
                            
                            # Adjusted quantity based on item count
                            item_count = ctx['total_items']
                            adj_qty_str = rule["adjustments"]["variety_count"](item_count)
                            adj_qty = extract_quantity_value(adj_qty_str)
                            
                            # Store the exact scaling factor
                            if base_qty > 0:
                                features[f"{norm_cat}__qty_scale_factor"] = adj_qty / base_qty
                                if item_count > 1:
                                    adj_minus_str = rule["adjustments"]["variety_count"](item_count - 1)
                                    adj_minus = extract_quantity_value(adj_minus_str)
                                    features[f"{norm_cat}__minus1_scale_factor"] = adj_minus / base_qty
                                
                            adj_plus_str = rule["adjustments"]["variety_count"](item_count + 1)
                            adj_plus = extract_quantity_value(adj_plus_str)
                            features[f"{norm_cat}__plus1_scale_factor"] = adj_plus / base_qty
                        except Exception as e:
                            # Fallback for parsing errors
                            features[f"{norm_cat}__qty_scale_factor"] = 1.0

            # --- Inter‑category dependencies ---
            categories_present = set(menu_context['categories'])
            
            for cat, deps in food_rules.category_dependencies.items():
                # Basic dependency count
                present_deps = [d for d in deps if d in categories_present]
                features[f"{cat}__deps_present_count"] = len(present_deps)
                features[f"{cat}__deps_satisfaction"] = len(present_deps) / len(deps) if deps else 0
                
                # Precise counterfactual quantity adjustment multiplier
                expected_multiplier = 1.0
                missing_imp_sum = 0.0
                present_imp_sum = 0.0
                
                for dep, importance in deps.items():
                    if dep in categories_present:
                        present_imp_sum += importance
                    else:
                        # This exactly mirrors the counterfactual generation formula
                        missing_imp_sum += importance
                        expected_multiplier *= (1.0 + (0.2 * importance))
                
                # Store the precise multiplier
                features[f"{cat}__expected_adj_multiplier"] = expected_multiplier
                features[f"{cat}__missing_dep_importance"] = missing_imp_sum
                features[f"{cat}__present_dep_importance"] = present_imp_sum
                
                # Second-order dependencies for key categories
                if cat in ['Biryani', 'Rice', 'Curries', 'Flavored_Rice', 'Dal']:
                    second_order_deps = set()
                    second_order_importance = 0.0
                    
                    for dep, imp in deps.items():
                        if dep in food_rules.category_dependencies:
                            # Get second-order dependencies
                            dep_deps = food_rules.category_dependencies[dep]
                            for d, d_imp in dep_deps.items():
                                if d != cat:  # Avoid self-reference
                                    second_order_deps.add(d)
                                    second_order_importance += d_imp * imp  # Importance cascade
                
                    second_order_deps = second_order_deps - set(deps.keys())
                
                    # Store network metrics
                    features[f"{cat}__second_order_count"] = len(second_order_deps)
                    features[f"{cat}__network_complexity"] = len(deps) * len(second_order_deps)
                    features[f"{cat}__second_order_importance"] = second_order_importance
                    
                    # Calculate presence of second-order dependencies
                    if second_order_deps:
                        present_second = sum(1 for d in second_order_deps if d in categories_present)
                        features[f"{cat}__second_order_presence"] = present_second / len(second_order_deps)
                    else:
                        features[f"{cat}__second_order_presence"] = 0.0
                        
                    # Create dependency network ratio
                    total_deps = len(deps) + len(second_order_deps)
                    if total_deps > 0:
                        features[f"{cat}__network_presence_ratio"] = (
                            (len(present_deps) + present_second) / total_deps
                        )
                    
            # Context sensitive categories
            context_sensitive_categories = [
                'Welcome_Drinks', 'Breakfast', 'Papad', 'Pickle', 'Ghee', 'Sides', 'Sides_pcs'
            ]
        
            for cat in context_sensitive_categories:
                if cat in menu_context['categories']:
                    # 1. Create context-interaction features
                    for meal in ['Breakfast', 'Lunch', 'Hi-Tea', 'Dinner']:
                        is_meal = 1.0 if meal_type == meal else 0.0
                        features[f"{cat}_{meal}_interaction"] = is_meal
                        
                        # Higher-order meal-veg interaction
                        if is_meal > 0 and 'veg_ratio' in features:
                            features[f"{cat}_{meal}_veg_interaction"] = is_meal * features['veg_ratio']
                    
                    # 2. Time-specific interactions
                    for time in ['Morning', 'Afternoon', 'Evening', 'Night']:
                        is_time = 1.0 if event_time == time else 0.0
                        features[f"{cat}_{time}_interaction"] = is_time
                    
                    # 3. Per-guest scaling features
                    cat_items = menu_context['items_by_category'].get(cat, [])
                    if cat_items and 'total_guests' in features:
                        features[f"{cat}_items_per_guest"] = len(cat_items) / max(features['total_guests'], 1)
            
            # Process small-quantity categories
            small_quantity_categories = ['Pickle', 'Papad', 'Ghee', 'Sides', 'Sides_pcs']
            
            for cat in small_quantity_categories:
                if cat in menu_context['categories']:
                    # Extract base quantities from this category
                    cat_items = menu_context['items_by_category'].get(cat, [])
                    if not cat_items:
                        continue
                        
                    # Calculate total category quantity
                    total_qty = 0.0
                    for item in cat_items:
                        if item in menu_context['item_properties']:
                            props = menu_context['item_properties'][item]
                            # Extract quantity from rules
                            if 'quantity_rule' in props:
                                qty_rule = props['quantity_rule']
                                qty_str = qty_rule.get('default_quantity', '10g')
                                qty = extract_quantity_value(qty_str)
                                total_qty += qty
                    
                    # Create transformed quantity features for non-linear scaling
                    if total_qty > 0:
                        # Logarithmic transformation reduces proportional error
                        features[f"{cat}_log_qty"] = np.log1p(total_qty)  # log(1+x) for stability
                        # Square root transformation for intermediate scaling
                        features[f"{cat}_sqrt_qty"] = np.sqrt(total_qty)
                        # Store actual qty for reference
                        features[f"{cat}_raw_qty"] = total_qty
                
            # Process dependency pairs
            dependency_importance_map = {}
            for primary, deps_dict in food_rules.category_dependencies.items():
                for dependent, importance in deps_dict.items():
                    dependency_importance_map[f"{primary}_{dependent}"] = importance
                    
            processed_pairs = set()
            for primary, deps in food_rules.category_dependencies.items():
                for dependent in deps:
                    pair_key = f"{primary}_{dependent}"
                    if pair_key in processed_pairs:
                        continue
                        
                    processed_pairs.add(pair_key)
                    
                    # Only process if both categories exist
                    if primary in menu_context['items_by_category'] and dependent in menu_context['items_by_category']:
                        primary_items = menu_context['items_by_category'][primary]
                        dependent_items = menu_context['items_by_category'][dependent]
                        
                        # Get importance from map
                        importance = dependency_importance_map.get(pair_key, 0.5)
                        
                        # Store relationship features
                        features[f"{pair_key}_both_present"] = 1.0
                        features[f"{pair_key}_importance"] = importance
                        features[f"{pair_key}_count_ratio"] = len(primary_items) / max(len(dependent_items), 1)
                        
                        # Calculate quantity-based features
                        try:
                            primary_qty = 0.0
                            dependent_qty = 0.0
                            
                            # Extract quantities from properties
                            for item in primary_items:
                                if item in menu_context['item_properties']:
                                    props = menu_context['item_properties'][item]
                                    # Try per_person_qty first, fallback to quantity_rule
                                    if 'per_person_qty' in props:
                                        qty_str = props['per_person_qty']
                                    elif 'quantity_rule' in props:
                                        qty_rule = props['quantity_rule']
                                        qty_str = qty_rule.get('default_quantity', '100g')
                                    else:
                                        qty_str = '100g'  # Sensible default
                                    primary_qty += food_rules.extract_quantity_value(qty_str)
                            
                            for item in dependent_items:
                                if item in menu_context['item_properties']:
                                    props = menu_context['item_properties'][item]
                                    if 'per_person_qty' in props:
                                        qty_str = props['per_person_qty']
                                    elif 'quantity_rule' in props:
                                        qty_rule = props['quantity_rule']
                                        qty_str = qty_rule.get('default_quantity', '100g')
                                    else:
                                        qty_str = '100g'
                                    dependent_qty += extract_quantity_value(qty_str)
                            
                            # Store ratios with edge case handling
                            if dependent_qty > 0:
                                features[f"{pair_key}_qty_ratio"] = primary_qty / dependent_qty
                            else:
                                features[f"{pair_key}_qty_ratio"] = -1.0  # Signal missing ratio
                            
                            if primary_qty > 0:
                                features[f"{dependent}_{primary}_qty_ratio"] = dependent_qty / primary_qty
                            else:
                                features[f"{dependent}_{primary}_qty_ratio"] = -1.0
                        except Exception:
                            # Fallback to count-based ratio
                            features[f"{pair_key}_qty_ratio"] = -1.0
        
        # Meal balance metrics
        features['meal_component_balance'] = len(menu_context.get('categories', [])) / 12.0  # Normalize by typical meal
        
        return features
        
    def print_feature_engineering_dataset(self, data, sample_count=10):
        """
        Print and save feature engineering dataset with robust dependency handling.
        
        Args:
            data (DataFrame): Training data
            sample_count (int): Number of orders to sample
            
        Returns:
            DataFrame: Sample feature engineering dataset
        """
        print(f"\n=== Feature Engineering Dataset (First {sample_count} Orders) ===\n")
        
        # Get encoder feature names
        event_time_features = [f"event_time_{c}" for c in self.event_time_encoder.categories_[0]]
        meal_type_features = [f"meal_type_{c}" for c in self.meal_type_encoder.categories_[0]]
        event_type_features = [f"event_type_{c}" for c in self.event_type_encoder.categories_[0]]
        
        # Store all sampled orders' features
        all_orders_features = []
        
        # Sample unique orders 
        unique_orders = data['Order_Number'].unique()
        sample_orders = unique_orders[:min(sample_count, len(unique_orders))]
        
        # Process each order without dependency service recreation
        for order_idx, order_num in enumerate(sample_orders):
            print(f"[Processing Order {order_idx+1}/{len(sample_orders)}: {order_num}]")
            
            # Get data for this order
            order_data = data[data['Order_Number'] == order_num]
            if order_data.empty:
                print(f"  - Warning: No data found for order {order_num}, skipping.")
                continue
            
            # Extract basic order information
            order_items = order_data['Item_name'].unique().tolist()
            evt_time = order_data['Event_Time'].iat[0]
            meal_t = order_data['Meal_Time'].iat[0]
            evt_type = order_data['Event_Type'].iat[0]
            
            # Handle column naming variations
            guests = order_data['total_guest_count'].iat[0] if 'total_guest_count' in order_data.columns else (
                order_data['Guest_Count'].iat[0] if 'Guest_Count' in order_data.columns else 50)
            
            vegs = order_data['Veg_Count'].iat[0] if 'Veg_Count' in order_data.columns else int(0.4 * guests)
            
            # Build order features dictionary
            order_features = {
                'order_number': order_num,
                'event_time': evt_time,
                'meal_type': meal_t,
                'event_type': evt_type,
                'total_guests': guests,
                'vegetarian_guests': vegs,
                'item_count': len(order_items)
            }
            
            # One-hot encode categorical features
            event_time_encoded = self.event_time_encoder.transform([[evt_time]])
            meal_type_encoded = self.meal_type_encoder.transform([[meal_t]])
            event_type_encoded = self.event_type_encoder.transform([[evt_type]])
            
            # Add encoded features to dictionary
            for i, feat in enumerate(event_time_features):
                order_features[feat] = event_time_encoded[0][i]
            for i, feat in enumerate(meal_type_features):
                order_features[feat] = meal_type_encoded[0][i]
            for i, feat in enumerate(event_type_features):
                order_features[feat] = event_type_encoded[0][i]
            
            # NEW: Extract dependency features from main data frame if available
            dep_features = [col for col in data.columns if col.startswith('dep_') or 
                                                        '__' in col or 
                                                        '_interaction' in col or
                                                        '_ratio' in col]
            
            if dep_features:
                dep_row = order_data.iloc[0]
                for feat in dep_features:
                    if feat in dep_row:
                        order_features[feat] = dep_row[feat]
            
            # Add to collection
            all_orders_features.append(order_features)
            
            # Print summary
            print(f"  Order {order_num}: {len(order_items)} items")
            print(f"  Event: {evt_time}, {meal_t}, {evt_type}, {guests} guests ({vegs} vegetarian)")
            print("  ------------")
        
        # Create DataFrame with all orders' features
        all_features_df = pd.DataFrame(all_orders_features)
        
        # Save to CSV with clear naming
        csv_filename = f'feature_engineering_sample_{sample_count}_orders.csv'
        all_features_df.to_csv(csv_filename, index=False)
        print(f"\nFeature engineering data for {len(all_orders_features)} orders saved to '{csv_filename}'")
        
        return all_features_df