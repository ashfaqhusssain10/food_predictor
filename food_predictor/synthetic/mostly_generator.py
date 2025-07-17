# food_predictor/synthetic/mostly_generator.py
import os
import pandas as pd
import logging
import tempfile
from typing import Dict, List, Optional, Tuple
from mostlyai.sdk import MostlyAI

logger = logging.getLogger(__name__)

class MostlyAIGenerator:
    """
    Mostly AI-based synthetic data generator for DF5 catering data.
    Replaces Tonic AI implementation with superior pattern learning.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Mostly AI client"""
        self.api_key = api_key or os.getenv('MOSTLYAI_API_KEY')
        if not self.api_key:
            raise ValueError("Mostly AI API key required. Set MOSTLYAI_API_KEY environment variable.")
        
        self.client = MostlyAI(api_key=self.api_key)
        self.generator_config = None
        self.trained_model = None
        
    def prepare_df5_data(self, df5_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DF5 data for Mostly AI training with business logic preservation.
        
        Args:
            df5_data: Raw DF5 dataset
            
        Returns:
            Cleaned and prepared dataset
        """
        logger.info("Preparing DF5 data for Mostly AI training")
        
        # Make a copy to avoid modifying original
        df_clean = df5_data.copy()
        if 'Classification' in df_clean.columns:
            # If the item is 'Veg', set Veg_Count to the total guest count for that row. Otherwise, 0.
            df_clean['Veg_Count'] = df_clean.apply(
                lambda row: row['total_guest_count'] if row['Classification'] == 'Veg' else 0,
                axis=1
            )
        else:
            # As a fallback, create the column with 0 if no 'Classification' column is found.
            df_clean['Veg_Count'] = 0
        # --- END OF FIX --
        # 1. Handle missing values strategically
        df_clean['Veg_Count'] = df_clean['Veg_Count'].fillna(
            df_clean['total_guest_count'] * 0.4  # Default 40% vegetarian
        )
        
        # 2. Add derived columns for business logic
        df_clean['veg_ratio'] = df_clean['Veg_Count'] / df_clean['total_guest_count']
        df_clean['items_per_order'] = df_clean.groupby('Order_Number')['Item_name'].transform('count')
        
        # 3. Extract quantity components
        df_clean['quantity_numeric'] = df_clean['Per_person_quantity'].apply(self._extract_quantity_value)
        df_clean['quantity_unit'] = df_clean['Per_person_quantity'].apply(self._extract_unit)
        
        # 4. Validate and clean data
        valid_mask = (
            (df_clean['quantity_numeric'] > 0) &
            (df_clean['total_guest_count'] > 0) &
            (df_clean['veg_ratio'] >= 0) &
            (df_clean['veg_ratio'] <= 1) &
            (df_clean['Veg_Count'] <= df_clean['total_guest_count'])
        )
        
        df_clean = df_clean[valid_mask].reset_index(drop=True)
        
        logger.info(f"Data preparation complete: {len(df5_data)} → {len(df_clean)} valid records")
        return df_clean
    
    def configure_synthetic_generation(self, df_prepared: pd.DataFrame) -> Dict:
        """
        Configure Mostly AI for DF5-specific synthetic data generation.
        
        Args:
            df_prepared: Prepared DF5 dataset
            
        Returns:
            Configuration dictionary
        """
        logger.info("Configuring Mostly AI generation settings")
        
        # Column-specific configurations
        column_config = {
            # Categorical columns - preserve distributions
            'Order_Number': {
                'type': 'categorical',
                'strategy': 'sequence',  # Generate unique order numbers
                'privacy': 'shuffle'     # Shuffle to avoid exact matches
            },
            'Event_Type': {
                'type': 'categorical',
                'strategy': 'distribution_preserve',
                'conditional_on': ['Meal_Time', 'Event_Time']  # Context dependency
            },
            'Meal_Time': {
                'type': 'categorical', 
                'strategy': 'distribution_preserve'
            },
            'Event_Time': {
                'type': 'categorical',
                'strategy': 'distribution_preserve'
            },
            'Item_name': {
                'type': 'categorical',
                'strategy': 'distribution_preserve',
                'conditional_on': ['Category', 'Classification']  # Maintain item-category relationship
            },
            'Category': {
                'type': 'categorical',
                'strategy': 'distribution_preserve'
            },
            'Classification': {
                'type': 'categorical',
                'strategy': 'distribution_preserve'
            },
            
            # Numerical columns with constraints
            'total_guest_count': {
                'type': 'numerical',
                'strategy': 'distribution_preserve',
                'constraints': {
                    'min_value': 1,
                    'max_value': 1000,
                    'data_type': 'integer'
                }
            },
            'Veg_Count': {
                'type': 'numerical',
                'strategy': 'conditional',
                'conditional_on': ['total_guest_count'],  # Must be <= total_guest_count
                'constraints': {
                    'min_value': 0,
                    'data_type': 'integer'
                }
            },
            'quantity_numeric': {
                'type': 'numerical',
                'strategy': 'distribution_preserve',
                'conditional_on': ['Item_name', 'Category', 'total_guest_count'],
                'constraints': {
                    'min_value': 0.1,
                    'max_value': 2000.0
                }
            },
            
            # Derived columns
            'veg_ratio': {
                'type': 'derived',
                'formula': 'Veg_Count / total_guest_count'
            },
            'items_per_order': {
                'type': 'derived',
                'formula': 'count(Item_name) group by Order_Number'
            }
        }
        
        # Business logic constraints
        business_constraints = [
            {
                'name': 'veg_count_constraint',
                'rule': 'Veg_Count <= total_guest_count',
                'severity': 'error'
            },
            {
                'name': 'positive_quantities',
                'rule': 'quantity_numeric > 0',
                'severity': 'error'
            },
            {
                'name': 'valid_veg_ratio',
                'rule': '0 <= veg_ratio <= 1',
                'severity': 'error'
            },
            {
                'name': 'item_category_consistency', 
                'rule': 'Item_name matches Category mapping',
                'severity': 'warning'
            }
        ]
        
        self.generator_config = {
            'columns': column_config,
            'constraints': business_constraints,
            'generation_method': 'deep_learning',  # Use Mostly AI's advanced GAN
            'privacy_level': 'high',
            'seed': 42  # For reproducibility
        }
        
        return self.generator_config
    
    # In food_predictor/synthetic/mostly_generator.py


    # In food_predictor/synthetic/mostly_generator.py
    # In food_predictor/synthetic/mostly_generator.py

    def train_generator(self, data_path: str) -> bool: # Accept a string path
        """
        Train Mostly AI generator on prepared DF5 data using a file path.
        """
        logger.info("Training Mostly AI generator...")
        try:
            self.trained_model = self.client.train(
                data=data_path,  # Use the file path here
                #config=self.generator_config,
                #name="DF5 Catering Generator v6"
            )

            if hasattr(self.trained_model, 'wait_until_ready'):
                self.trained_model.wait_until_ready()

            logger.info("Training completed successfully!")
            return True

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            # Clean up the temporary file if training fails
            import os
            if os.path.exists(data_path):
                os.remove(data_path)
            return False

    
    def generate_synthetic_data(self, num_records: int) -> pd.DataFrame:
        """
        Generate synthetic data using trained Mostly AI model.
        
        Args:
            num_records: Number of records to generate
            
        Returns:
            Generated synthetic dataset
        """
        if not self.trained_model:
            raise ValueError("Model not trained. Call train_generator() first.")
        
        logger.info(f"Generating {num_records} synthetic records...")
        
        try:
            # Generate synthetic data
            synthetic_data = self.client.synthesize(
                generator=self.trained_model,
                size=num_records,
                seed=42  # For reproducibility
            )
            
            # Post-process to ensure business logic
            synthetic_data = self._post_process_synthetic_data(synthetic_data)
            
            logger.info(f"Successfully generated {len(synthetic_data)} records")
            return synthetic_data
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise
    
    def _post_process_synthetic_data(self, synthetic_data: pd.DataFrame) -> pd.DataFrame:
        """Apply post-processing to ensure business logic compliance."""
        df = synthetic_data.copy()
        
        # Enforce constraints that Mostly AI might not perfectly capture
        df['Veg_Count'] = df[['Veg_Count', 'total_guest_count']].min(axis=1)
        df['veg_ratio'] = df['Veg_Count'] / df['total_guest_count']
        df['veg_ratio'] = df['veg_ratio'].clip(0, 1)
        
        # Reconstruct Per_person_quantity string
        df['Per_person_quantity'] = df['quantity_numeric'].astype(str) + df['quantity_unit'].fillna('g')
        
        # Ensure positive quantities
        df['quantity_numeric'] = df['quantity_numeric'].clip(lower=0.1)
        
        return df
    
    def _extract_quantity_value(self, quantity_str: str) -> float:
        """Extract numerical value from quantity string."""
        import re
        if pd.isna(quantity_str):
            return 0.0
        match = re.search(r'(\d+\.?\d*)', str(quantity_str))
        return float(match.group(1)) if match else 0.0
    
    def _extract_unit(self, quantity_str: str) -> str:
        """Extract unit from quantity string."""
        import re
        if pd.isna(quantity_str):
            return 'g'
        match = re.search(r'[a-zA-Z]+', str(quantity_str))
        return match.group(0) if match else 'g'


class MostlyAIDataAugmentor:
    """
    High-level interface for DF5 data augmentation using Mostly AI.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.generator = MostlyAIGenerator(api_key)
        self.validation_engine = None  # Will be set up separately
    
    def augment_df5_dataset(
        self, 
        df5_data: pd.DataFrame, 
        target_multiplier: float = 3.0,
        validation_split: float = 0.2
    ) -> Dict[str, pd.DataFrame]:
        """
        Complete pipeline to augment DF5 dataset.
        
        Args:
            df5_data: Original DF5 dataset
            target_multiplier: How many times to multiply the data (e.g., 3.0 = 3x original size)
            validation_split: Fraction to hold out for validation
            
        Returns:
            Dictionary with 'synthetic', 'training', 'validation', 'combined' datasets
        """
        from sklearn.model_selection import train_test_split
        
        logger.info(f"Starting DF5 augmentation: {len(df5_data)} → {int(len(df5_data) * target_multiplier)} records")
        
        # Step 1: Prepare data
        df_prepared = self.generator.prepare_df5_data(df5_data)
        # --- ADD THIS SECTION TO FIX THE ERROR ---
        # Identify categories with more than one member
        category_counts = df_prepared['Category'].value_counts()
        valid_categories = category_counts[category_counts > 1].index.tolist()
        
        # Filter the dataframe to only include valid categories
        df_filtered = df_prepared[df_prepared['Category'].isin(valid_categories)]
        logger.info(f"Filtered out rare categories: {len(df_prepared)} → {len(df_filtered)} records")
        # --- END OF FIX ---
        # Step 2: Split for validation
        train_data, val_data = train_test_split(
            df_filtered,
            test_size=validation_split,
            stratify=df_filtered['Category'],
            random_state=42
        )
            # --- START OF NEW CODE ---
        # Save the training data to a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            train_data.to_csv(tmp.name, index=False)
            temp_data_path = tmp.name
        
        logger.info(f"Saved prepared training data to temporary file: {temp_data_path}")
        # --- END OF NEW CODE ---

        # Step 3: Configure and train
        self.generator.configure_synthetic_generation(train_data)
        success = self.generator.train_generator(temp_data_path)
        
        if not success:
            raise RuntimeError("Failed to train Mostly AI generator")
        
        # Step 4: Generate synthetic data
        target_synthetic_records = int(len(train_data) * (target_multiplier - 1))
        synthetic_data = self.generator.generate_synthetic_data(target_synthetic_records)
        
        # Step 5: Combine datasets
        combined_data = pd.concat([train_data, synthetic_data], ignore_index=True)
        
        return {
            'synthetic': synthetic_data,
            'training': train_data, 
            'validation': val_data,
            'combined': combined_data
        }