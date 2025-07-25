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
        
        self.client = MostlyAI(
            api_key=self.api_key,
            base_url='https://app.mostly.ai'
        )
        self.generator_config = None
        self.trained_generator = None
        
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
            Configuration dictionary for Mostly AI SDK 4.9.0
        """
        logger.info("Configuring Mostly AI generation settings")
        
        # Mostly AI SDK 4.9.0 configuration format
        self.generator_config = {
            'name': 'DF5 Catering Generator - Advanced GAN',
            'tables': [{
                'name': 'df5_catering',
                'data': df_prepared,
                'columns': [
                    # Categorical columns
                    {'name': 'Order_Number', 'model_encoding_type': 'TABULAR_CATEGORICAL'},
                    {'name': 'Event_Type', 'model_encoding_type': 'TABULAR_CATEGORICAL'},
                    {'name': 'Meal_Time', 'model_encoding_type': 'TABULAR_CATEGORICAL'},
                    {'name': 'Event_Time', 'model_encoding_type': 'TABULAR_CATEGORICAL'},
                    {'name': 'Item_name', 'model_encoding_type': 'TABULAR_CATEGORICAL'},
                    {'name': 'Category', 'model_encoding_type': 'TABULAR_CATEGORICAL'},
                    {'name': 'Classification', 'model_encoding_type': 'TABULAR_CATEGORICAL'},
                    {'name': 'quantity_unit', 'model_encoding_type': 'TABULAR_CATEGORICAL'},
                     
                   
                    # ✅ FIXED: Numerical columns
                    {'name': 'total_guest_count', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},
                    {'name': 'Veg_Count', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},
                    {'name': 'quantity_numeric', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},
                    {'name': 'veg_ratio', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},
                    {'name': 'items_per_order', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'}

                ],
                'tabular_model_configuration': {
                    'model': 'AUTO',  # This is Mostly AI's advanced GAN model
                    'max_training_time': None,  # No time limit for quality
                    
                }
            }]
        }
        
        return self.generator_config
    
    def train_generator(self, df_prepared: pd.DataFrame) -> bool:
        """
        Train Mostly AI generator on prepared DF5 data.
        
        Args:
            df_prepared: Prepared DF5 DataFrame
            
        Returns:
            Success status
        """
        logger.info("Training Mostly AI generator with Advanced GAN (TabularARGN)...")
        
        try:
            # Configure the generator with the prepared data
            config = self.configure_synthetic_generation(df_prepared)
            
            # Train using the Mostly AI client
            # In SDK 4.9.0, we pass the config with data included
            self.trained_generator = self.client.train(
                config=config,
                start=True,  # Start training immediately
                wait=True    # Wait for training to complete
            )
            
            logger.info("Training completed successfully!")
            logger.info(f"Generator ID: {self.trained_generator.id}")
            logger.info("Using model: TABULAR_BASIC(Basic GAN)")
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return False
    

    def wait_for_generator_ready(self, generator, timeout_minutes=10):
        """
        Wait for generator to reach DONE status before proceeding with generation.
        """
        import time
        
        logger.info("Waiting for generator to reach DONE status...")
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        while True:
            try:
                # Refresh generator status
                updated_generator = self.client.generators.get(generator.id)
                current_status = updated_generator.training_status
                
                logger.info(f"Generator status: {current_status}")
                
                # Check if generator is ready
                if current_status == "DONE":
                    logger.info("✅ Generator is ready for synthetic data generation!")
                    return True
                elif current_status in ["FAILED", "CANCELED"]:
                    logger.error(f"❌ Generator training failed with status: {current_status}")
                    return False
                elif current_status in ["IN_PROGRESS", "QUEUED", "ON_HOLD"]:
                    logger.info(f"⏳ Generator still processing (status: {current_status}), waiting...")
                
                # Check timeout
                elapsed_time = time.time() - start_time
                if elapsed_time > timeout_seconds:
                    logger.error(f"❌ Timeout waiting for generator to be ready ({timeout_minutes} minutes)")
                    return False
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error checking generator status: {str(e)}")
                time.sleep(10)  # Wait longer on error
                continue

    def generate_synthetic_data(self, num_records: int) -> pd.DataFrame:
        """
        Generate synthetic data using trained Mostly AI model.
        
        Args:
            num_records: Number of records to generate
            
        Returns:
            Generated synthetic dataset
        """
        if not self.trained_generator:
            raise ValueError("Model not trained. Call train_generator() first.")
        # 🔧 FIX: Wait for generator to be ready before generating
        if not self.wait_for_generator_ready(self.trained_generator):
            raise RuntimeError("Generator is not ready for synthetic data generation")
        logger.info(f"Generating {num_records} synthetic records...")
        
        try:
            # Generate synthetic data using the trained generator
            synthetic_dataset = self.client.generate(
                generator=self.trained_generator,
                size=num_records
            )
            
            # Get the data as DataFrame
            synthetic_data = synthetic_dataset.data()
            
            # Alternative: Use probe for quick sampling
            # synthetic_data = self.client.probe(
            #     generator=self.trained_generator,
            #     size=num_records
            # )
            
            # Convert to pandas DataFrame if needed
            if not isinstance(synthetic_data, pd.DataFrame):
                synthetic_data = pd.DataFrame(synthetic_data)
            
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
        
        # Ensure required columns exist
        if 'Veg_Count' in df.columns and 'total_guest_count' in df.columns:
            # Enforce constraints that Mostly AI might not perfectly capture
            df['Veg_Count'] = df[['Veg_Count', 'total_guest_count']].min(axis=1)
            df['veg_ratio'] = df['Veg_Count'] / df['total_guest_count']
            df['veg_ratio'] = df['veg_ratio'].clip(0, 1)
        
        # Reconstruct Per_person_quantity string if needed
        if 'quantity_numeric' in df.columns and 'quantity_unit' in df.columns:
            df['Per_person_quantity'] = df['quantity_numeric'].astype(str) + df['quantity_unit'].fillna('g')
        
        # Ensure positive quantities
        if 'quantity_numeric' in df.columns:
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
        
        # Identify categories with more than one member
        category_counts = df_prepared['Category'].value_counts()
        valid_categories = category_counts[category_counts > 1].index.tolist()
        
        # Filter the dataframe to only include valid categories
        df_filtered = df_prepared[df_prepared['Category'].isin(valid_categories)]
        logger.info(f"Filtered out rare categories: {len(df_prepared)} → {len(df_filtered)} records")
        
        # Step 2: Split for validation
        train_data, val_data = train_test_split(
            df_filtered,
            test_size=validation_split,
            stratify=df_filtered['Category'],
            random_state=42
        )
        
        # Step 3: Train generator directly with the DataFrame
        # No need to save to file - pass DataFrame directly
        success = self.generator.train_generator(train_data)
        
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