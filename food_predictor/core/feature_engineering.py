# core/feature_engineering.py
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from food_predictor.core.menu_analyzer import MenuAnalyzer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from food_predictor.core.category_rules import FoodCategoryRules
from food_predictor.data.item_service import ItemService
from food_predictor.utils.quantity_utils import extract_quantity_value,extract_unit,validate_unit,infer_default_unit


logger = logging.getLogger(__name__)

class FeatureSpaceManager(BaseEstimator, TransformerMixin):
    """Handles dictionary-to-matrix conversion with consistent dimensionality.
    
    This transformer maintains exact sklearn compatibility:
    - Preserves identical feature space between fit/transform
    - Handles dynamic feature dictionaries
    - Produces deterministic output dimensions
    """
    
    def __init__(self):
        # Following sklearn naming conventions
        self.feature_names_in_ = None
        self.n_features_in_ = None
        self._feature_indices = {}
        
    def fit(self, X, y=None):
        """Learn feature space from dictionary or list of dictionaries.
        
        Parameters
        ----------
        X : dict or list of dict
            Training feature dictionaries
        
        Returns
        -------
        self : object
            Returns self for method chaining
        """
        # Handle both single dictionary and list of dictionaries
        if isinstance(X, dict):
            dict_list = [X]
        else:
            dict_list = X
            
        # Build canonical feature space
        all_features = set()
        for feature_dict in dict_list:
            all_features.update(feature_dict.keys())
            
        # Sort for determinism - critical for consistent output
        self.feature_names_in_ = sorted(list(all_features))
        self.n_features_in_ = len(self.feature_names_in_)
        
        # Build fast lookup for O(1) access
        self._feature_indices = {feat: i for i, feat in enumerate(self.feature_names_in_)}
        
        return self
        
    def transform(self, X):
        """Transform dictionaries to fixed-dimension arrays.
        
        Parameters
        ----------
        X : dict or list of dict
            Feature dictionaries to transform
            
        Returns
        -------
        X_transformed : ndarray
            Shape (n_samples, n_features_in_)
        """
        # Handle both single dictionary and list of dictionaries
        if isinstance(X, dict):
            dict_list = [X]
        else:
            dict_list = X
            
        # Initialize output matrix with zeros
        result = np.zeros((len(dict_list), self.n_features_in_))
        
        # Populate with values from dictionaries
        for i, feature_dict in enumerate(dict_list):
            for feat_name, feat_value in feature_dict.items():
                if feat_name in self._feature_indices:
                    idx = self._feature_indices[feat_name]
                    result[i, idx] = feat_value
        
        return result
        
    def get_feature_names_out(self, input_features=None):
        """Get output feature names.
        
        Parameters
        ----------
        input_features : None
            Unused, present for API compatibility
            
        Returns
        -------
        feature_names_out : ndarray
            Output feature names
        """
        return np.array(self.feature_names_in_)



class FeatureTypeSplitter(BaseEstimator, TransformerMixin):
    """Splits feature matrix by type and applies appropriate scaling.
    
    This follows the compositional pattern from sklearn documentation,
    but adapts it for dynamic feature space.
    """
    
    def __init__(self):
        self.feature_types = {
            'ratio': lambda name: 'ratio' in name or 'multiplier' in name or 'scale_factor' in name,
            'binary': lambda name: name.startswith('is_') or name.startswith('has_') or '_present' in name,
            'network': lambda name: 'network_complexity' in name or 'second_order' in name,
            'transformed': lambda name: 'log_qty' in name or 'sqrt_qty' in name,
            'raw_qty': lambda name: 'raw_qty' in name,
            'standard': lambda name: True  # Catch-all for other features
        }
        
        self.type_scalers = {}
        self.feature_indices = {}
        self.feature_names_in_ = None
        
    def fit(self, X, y=None):
        """Fit scalers on appropriate feature subsets.
        
        Parameters
        ----------
        X : array-like
            Feature matrix with columns corresponding to feature_names_in_
            
        Returns
        -------
        self : object
        """
        # Store feature names for validation
        if hasattr(X, 'columns'):
            # DataFrame case
            self.feature_names_in_ = X.columns.tolist()
        else:
            # Get from outer scope - not ideal but practical
            self.feature_names_in_ = self.feature_space_manager.feature_names_in_
            
        # Group indices by feature type
        for type_name, type_predicate in self.feature_types.items():
            # Get indices of features matching this type
            indices = [i for i, name in enumerate(self.feature_names_in_) 
                      if type_predicate(name)]
            
            if not indices:
                continue
                
            self.feature_indices[type_name] = indices
            
            # Create appropriate scaler for this type
            if type_name in ('binary', 'network'):
                # No scaling for binary features
                self.type_scalers[type_name] = None
            elif type_name == 'ratio':
                # No centering for ratio features to preserve semantics
                self.type_scalers[type_name] = RobustScaler(
                    with_centering=False, 
                    with_scaling=True,
                    quantile_range=(10.0, 90.0)
                )
            else:
                # Standard robust scaling for other features
                self.type_scalers[type_name] = RobustScaler()
                
            # Fit scaler if needed
            if self.type_scalers[type_name] is not None:
                # Extract feature subset and fit
                X_subset = X[:, indices]
                self.type_scalers[type_name].fit(X_subset)
                
        return self
        
    def transform(self, X):
        """Transform features by applying appropriate scalers.
        
        Parameters
        ----------
        X : array-like
            Feature matrix with same structure as in fit
            
        Returns
        -------
        X_scaled : ndarray
            Transformed features
        """
        X_scaled = X.copy()
        
        # Apply each type-specific scaler
        for type_name, indices in self.feature_indices.items():
            scaler = self.type_scalers[type_name]
            
            if scaler is not None:
                X_subset = X[:, indices]
                X_scaled[:, indices] = scaler.transform(X_subset)
                
        return X_scaled
    
class FeatureEngineer:
    """
    Handles feature extraction, transformation, and engineering for food prediction models.
    
    This class is responsible for:
    1. Encoding categorical variabless
    2. Extracting menu context features
    3. Handling inter-category dependencies
    4. Scaling numerical features
    """
    
    def __init__(self,food_rules= None,menu_analyzer=None):
        """Initialize feature engineering components."""
        self.event_time_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.meal_type_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.event_type_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.feature_space_manager = FeatureSpaceManager()
        self.feature_type_splitter = FeatureTypeSplitter()
        self.feature_type_splitter.feature_space_manager = self.feature_space_manager
        self.menu_feature_keys = set()
        self.menu_analyzer =  menu_analyzer
        
        
        self.guest_scaler = None
        self.ratio_feature_scaler = None
        self.std_feature_scaler = None
        self.network_feature_scaler = None
        self.transform_feature_scaler = None
        self.raw_qty_feature_scaler = None
        self.menu_feature_scaler = None
        self.food_rules = food_rules
        self.scalers_fitted = False
        self.expected_columns = []
    def fit_encoders(self, data):
        """
        Fit categorical encoders on training data.
        
        Args:
            data (DataFrame): Training data with categorical columns
        """
        self.event_time_encoder.fit(data[['Event_Time']])
        self.meal_type_encoder.fit(data[['Meal_Time']])
        self.event_type_encoder.fit(data[['Event_Type']])
        
    def collect_menu_features(self, data):
        """
        Collect menu features across all orders for proper statistical distribution.
        
        Args:
            data (DataFrame): Training data with order details
            
        Returns:
            tuple: (all_menu_features, feature_matrix)
        """
        # Efficient collection of menu features across all orders
        all_menu_features = []
        unique_orders = data['Order_Number'].unique()
        logger.info(f"Collecting menu features across {len(unique_orders)} orders")
        
        # Progress tracking for large datasets
        progress_interval = max(1, len(unique_orders) // 10)
        
        for idx, order_number in enumerate(unique_orders):
            order_data = data[data['Order_Number'] == order_number]
            
            # Skip empty orders
            if order_data.empty:
                continue
                
            # Extract order metadata efficiently
            evt_time = order_data['Event_Time'].iloc[0]
            meal_t = order_data['Meal_Time'].iloc[0]
            evt_type = order_data['Event_Type'].iloc[0]
            
            # Handle column name variations
            guests = (order_data['total_guest_count'].iloc[0] if 'total_guest_count' in order_data.columns
                    else order_data['Guest_Count'].iloc[0] if 'Guest_Count' in order_data.columns
                    else 50)  # Default fallback
            
            vegs = (order_data['Veg_Count'].iloc[0] if 'Veg_Count' in order_data.columns
                else int(0.4 * guests))  # Default fallback
                
            items = order_data['Item_name'].unique().tolist()
            
            # Build menu context - reuse existing implementation
            menu_context = self.menu_analyzer.build_menu_context(
                evt_time, meal_t, evt_type, guests, vegs, items
            )
            
            # Extract features with comprehensive context
            menu_features = self.extract_menu_features(
                data=order_data, 
                menu_context=menu_context, 
                food_rules=self.food_rules
            )
            
            all_menu_features.append(menu_features)
            
            # Log progress for large datasets
            if (idx + 1) % progress_interval == 0:
                logger.debug(f"Processed {idx + 1}/{len(unique_orders)} orders...")
        
        logger.info(f"Collected menu features from {len(all_menu_features)} orders")
        
        return all_menu_features
    
    def fit_scalers(self, data, menu_features_samples):
        """
        Fit all scalers on proper feature distributions across all training samples.
        
        Args:
            data (DataFrame): Training data with numeric columns
            menu_features_samples (list): List of menu feature dictionaries from all orders
        """
        # Fit guest count scaler - standard practice for this column
        if 'Guest_Count' in data.columns:
            guest_count = data[['Guest_Count']].values.astype(float)
            self.guest_scaler = RobustScaler()
            self.guest_scaler.fit(guest_count)
            logger.info(f"Fitted guest count scaler on {len(guest_count)} samples with shape {guest_count.shape}")
        
        # Validate inputs
        if not menu_features_samples:
            logger.warning("No menu feature samples provided - scalers not fitted")
            return
        
        # Performance optimization: estimate memory requirements
        sample_size = len(menu_features_samples)
        sample_feature_count = len(menu_features_samples[0]) if menu_features_samples else 0
        estimated_mb = (sample_size * sample_feature_count * 8) / (1024 * 1024)  # 8 bytes per float64
        
        if estimated_mb > 1000:  # 1 GB threshold
            logger.warning(f"Large feature matrix expected: ~{estimated_mb:.1f} MB. Consider batch processing.")
        
        # Step 1: Build canonical feature space across ALL samples
        logger.info(f"Building canonical feature space from {len(menu_features_samples)} samples")
        self.feature_space_manager.fit(menu_features_samples)
        feature_count = self.feature_space_manager.n_features_in_
        logger.info(f"Canonical feature space contains {feature_count} dimensions")
        
        # Step 2: Transform all samples to consistent feature matrix with shape (n_samples, n_features)
        # This is the critical architectural improvement - fitting on full distributions
        logger.info("Transforming all samples to consistent feature matrix")
        feature_matrix = self.feature_space_manager.transform(menu_features_samples)
        
        # Step 3: Fit the feature type splitter on the FULL feature matrix
        logger.info(f"Fitting specialized scalers on feature matrix with shape {feature_matrix.shape}")
        self.feature_type_splitter.fit(feature_matrix)
        
        # Log detailed statistics about fitted scalers
        type_counts = {type_name: len(indices) for type_name, indices in 
                    self.feature_type_splitter.feature_indices.items()}
        logger.info(f"Feature type distribution: {type_counts}")
        
        # Mark scalers as fitted and store feature keys for consistency
        self.scalers_fitted = True
        self.menu_feature_keys = set(self.feature_space_manager.feature_names_in_)
        logger.info("All scalers successfully fitted on proper feature distributions")
    
    
    def create_feature_pipeline(self):
        """
        Create a composite sklearn pipeline with ColumnTransformer for feature processing.
        """

        
        # Step 1: Define the transformers for different feature types
        categorical_cols = ['Event_Time', 'Meal_Time', 'Event_Type']
        
        # REUSE your pre-fitted encoders instead of creating new ones
        categorical_transformer = Pipeline([
            ('onehot_event', self.event_time_encoder),
            ('onehot_meal', self.meal_type_encoder),
            ('onehot_type', self.event_type_encoder)
        ])
                # Numerical feature processing with optional guest count
        numerical_transformer = Pipeline([
            ('scaler', RobustScaler())
        ])
        
        # Skip manual dimension calculation since ColumnTransformer will determine dimensions
        
        # Create the ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_cols),
                ('num', numerical_transformer, ['Guest_Count']),
            ],
            remainder='drop',
            verbose_feature_names_out=False
        )
        
        # Feature names mapping with simpler structure
        feature_names_mapping = {
            'categorical': {
                'start_idx': 0,
                'end_idx': None  # Will be determined during transform
            },
            'numerical': {
                'start_idx': None,  # Will be determined during transform
                'end_idx': None  
            },
            'menu_features': {
                'start_idx': None,
                'end_idx': None
            }
        }
        
        return preprocessor, feature_names_mapping
    
    def prepare_features(self, data, menu_features=None):
        """
        Prepare feature matrix with enhanced dependency features.
        
        Args:
            data (DataFrame): Input data
            menu_features (dict, optional): Menu context features
                
        Returns:
            ndarray: Feature matrix
        """
        # Ensure we have a copy to avoid SettingWithCopyWarning
        X = data.copy()
        
        # Ensure Guest_Count column exists - critical for pipeline consistency
        if 'Guest_Count' not in X.columns and 'total_guest_count' in X.columns:
            X['Guest_Count'] = X['total_guest_count']
        elif 'Guest_Count' not in X.columns:
            # Create dummy column to avoid pipeline errors
            X['Guest_Count'] = 0
        
        # 1) encode the three categorical columns (all return arrays of shape (n_rows, n_encoded))
        if not hasattr(self, '_feature_pipeline'):
            self._feature_pipeline, _ = self.create_feature_pipeline()
            self._feature_pipeline.fit(X)
        base_features = self._feature_pipeline.transform(X) 

    

        # 2) Add guest count if present
        guest_count = X[['Guest_Count']].values.astype(float)
        if self.guest_scaler is None:
            guest_scaled = guest_count
        else:
            guest_scaled = self.guest_scaler.transform(guest_count)
        features = np.hstack([base_features, guest_scaled])

        # 3) Add menu context features (broadcasted to every row)
        if menu_features is not None:
            if hasattr(self, 'menu_feature_keys') and self.menu_feature_keys:
                for key in self.menu_feature_keys:
                    if key not in menu_features:
                        menu_features[key] = 0.0
            
            # Transform menu features with consistent dimensionality
            menu_array = self.feature_space_manager.transform([menu_features])
            
            # Apply proper type-specific scaling with FeatureTypeSplitter
            menu_array_scaled = self.feature_type_splitter.transform(menu_array)
            
            # Broadcast to match number of rows in input data
            n_rows = X.shape[0]
            menu_features_repeated = np.tile(menu_array_scaled, (n_rows, 1))
            
            # Combine with base features
            features = np.hstack([base_features, menu_features_repeated])
        else:
            features = base_features
        
        return features
        
    
    
    
    
    
    

    # In feature_engineering.py, add a new method after prepare_features:

    def select_features_by_registry(self, menu_features, feature_registry):
        """
        Select features from menu_features based on feature registry.
        
        Args:
            menu_features (dict): Current menu features dictionary
            feature_registry (list): List of feature names to include
            
        Returns:
            dict: Filtered menu features with only registry features
        """
        # If no registry or empty registry, return original features
        if not feature_registry:
            return menu_features
            
        # Select only features in the registry, preserving order
        filtered_features = {}
        for feature_name in feature_registry:
            # Use 0.0 as default if feature not in current menu features
            filtered_features[feature_name] = menu_features.get(feature_name, 0.0)
        
        return filtered_features
    def adjust_feature_dimensions(self, feature_matrix, target_dims):
        """
        Adjust feature matrix to match target dimensions.
        
        Args:
            feature_matrix (ndarray): Input feature matrix
            target_dims (int): Target number of dimensions
            
        Returns:
            ndarray: Feature matrix with adjusted dimensions
        """
        current_dims = feature_matrix.shape[1]
        
        if current_dims == target_dims:
            return feature_matrix
            
        if current_dims > target_dims:
            # Truncate features to match target dimensions
            return feature_matrix[:, :target_dims]
        else:
            # Pad with zeros to match target dimensions
            import numpy as np
            padded_features = np.zeros((feature_matrix.shape[0], target_dims))
            padded_features[:, :current_dims] = feature_matrix
            return padded_features
        
    def track_feature_columns(self, X, feature_names=None):
        """
        Record the expected feature column names from training.
        
        Args:
            X (ndarray): Feature matrix
            feature_names (list, optional): Names for each feature column
        """
        if feature_names is not None and len(feature_names) == X.shape[1]:
            self.expected_columns = feature_names
        else:
            # Generate position-based names if not provided
            self.expected_columns = [f"f{i}" for i in range(X.shape[1])]
        
        logger.info(f"Tracked {len(self.expected_columns)} expected feature columns")
        return self.expected_columns

    def get_expected_feature_names(self):
        """
        Get the expected feature column names for XGBoost.
        
        Returns:
            list: Feature column names in expected order
        """
        if not self.expected_columns:
            logger.warning("No expected columns tracked! Using default empty list.")
        return self.expected_columns
        
        
        
    
    
    def track_menu_feature_keys(self, menu_features):
        """
        Track all possible menu feature keys to ensure consistency.
        
        Args:
            menu_features (dict): Current menu features
        """
        if not hasattr(self, 'menu_feature_keys'):
            self.menu_feature_keys = set()
        
        if menu_features:
            # Add all keys from this menu features dict to our master set
            self.menu_feature_keys.update(menu_features.keys())
            
            # Log the current count for debugging
            if len(self.menu_feature_keys) % 10 == 0:  # Only log occasionally to reduce spam
                logger.debug(f"Now tracking {len(self.menu_feature_keys)} unique menu feature keys")
        
        
    
    
    
    
    
        
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
        ENABLE_SECOND_ORDER_DEPS = False  # Disable second-order dependencies
        MAX_DEPENDENCY_PAIRS = 15  # Limit pairwise dependency features
        ENABLE_INTERACTION_FEATURES = False  # Disable sparse interaction features
        ENABLE_QTY_TRANSFORM = False  
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
                if cat in ['Biryani', 'Rice', 'Curries', 'Flavored_Rice', 'Dal'] and ENABLE_SECOND_ORDER_DEPS:
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
            if ENABLE_INTERACTION_FEATURES :
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
            if ENABLE_QTY_TRANSFORM:
                for cat in small_quantity_categories :
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
            pair_count = 0
            for primary, deps in food_rules.category_dependencies.items():
                for dependent in deps:
                    pair_key = f"{primary}_{dependent}"
                    if pair_key in processed_pairs:
                        continue
                        
                    processed_pairs.add(pair_key)
                    pair_count+=1
                    if pair_count > MAX_DEPENDENCY_PAIRS:
                        break
                    
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
        self.track_menu_feature_keys(features)
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