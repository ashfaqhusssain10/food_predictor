# Create: food_predictor/data/df5_analyzer.py
import pandas as pd
import numpy as np
from food_predictor.utils.quantity_utils import extract_quantity_value, extract_unit

class DF5Analyzer:
    def __init__(self, df5_data):
        self.df5_data = df5_data.copy()
        self.analysis_results = {}
    
    def analyze_data_quality(self):
        """Comprehensive data quality analysis"""
        print("=== DF5 Data Quality Analysis ===")
        print(f"Shape: {self.df5_data.shape}")
        print(f"Unique Orders: {self.df5_data['Order_Number'].nunique()}")
        print(f"Unique Items: {self.df5_data['Item_name'].nunique()}")
        print(f"Missing Values:\n{self.df5_data.isnull().sum()}")
        
        # Analyze quantities
        self.df5_data['quantity_numeric'] = self.df5_data['Per_person_quantity'].apply(extract_quantity_value)
        self.df5_data['quantity_unit'] = self.df5_data['Per_person_quantity'].apply(extract_unit)
        
        print(f"\nQuantity Statistics:")
        print(self.df5_data['quantity_numeric'].describe())
        print(f"\nUnits: {self.df5_data['quantity_unit'].value_counts()}")
        
        return self.df5_data
    
    def identify_constraints(self):
        """Identify business constraints for Tonic AI"""
        constraints = []
        
        # 1. Order-level constraints
        order_stats = self.df5_data.groupby('Order_Number').agg({
            'total_guest_count': 'first',
            'Veg_Count': 'first',
            'Item_name': 'count'
        }).rename(columns={'Item_name': 'items_per_order'})
        
        print(f"\n=== Business Constraints Identified ===")
        print(f"Items per order: {order_stats['items_per_order'].describe()}")
        print(f"Guest count range: {order_stats['total_guest_count'].min()}-{order_stats['total_guest_count'].max()}")
        
        # 2. Category constraints
        category_stats = self.df5_data.groupby('Category').agg({
            'quantity_numeric': ['min', 'max', 'mean'],
            'Item_name': 'count'
        })
        
        print(f"\nCategory constraints:")
        print(category_stats)
        
        return {
            'order_constraints': order_stats,
            'category_constraints': category_stats
        }