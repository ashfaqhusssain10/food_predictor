# Create: food_predictor/data/data_loader.py
import os
import pandas as pd

class DataLoader:
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), 'datasets')
    
    def load_df5(self):
        """Load DF5 original data"""
        df5_path = os.path.join(self.data_dir, 'DF5.xlsx')
        return pd.read_excel(df5_path)
    
    def load_db38(self):
        """Load DB38 training data"""
        db38_path = os.path.join(self.data_dir, 'DB38.xlsx')
        return pd.read_excel(db38_path)
    
    def save_synthetic_data(self, data, filename):
        """Save synthetic data"""
        synthetic_dir = os.path.join(self.data_dir, 'synthetic')
        os.makedirs(synthetic_dir, exist_ok=True)
        
        save_path = os.path.join(synthetic_dir, filename)
        data.to_excel(save_path, index=False)
        return save_path