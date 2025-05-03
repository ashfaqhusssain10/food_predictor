import pandas as pd
import numpy as np
import logging
import argparse

from food_predictor.core.model_manager import ModelManager
from food_predictor.core.feature_engineering import FeatureEngineer
from food_predictor.core.category_rules import FoodCategoryRules
from food_predictor.data.item_service import ItemService
from food_predictor.core.menu_analyzer import MenuAnalyzer
from food_predictor.pricing.price_calculator import PriceCalculator
from food_predictor.utils.quantity_utils import extract_quantity_value

def main():
    parser = argparse.ArgumentParser(description="Food Quantity Predictor & Price Calculator")
    parser.add_argument('--data', type=str, default="DB38.xlsx", help="Path to training data")
    parser.add_argument('--mode', choices=['train', 'predict', 'price'], default='train')
    parser.add_argument('--model_dir', type=str, default='./models')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    # Initialize core components
    food_rules = FoodCategoryRules()
    item_service = ItemService(food_rules)
    feature_engineer = FeatureEngineer(food_rules)
    menu_analyzer = MenuAnalyzer(food_rules, item_service)
    model_manager = ModelManager(feature_engineer, item_service, food_rules,menu_analyzer)
    price_calculator = PriceCalculator(food_rules, item_service)

    if args.mode == 'train':
        data = pd.read_excel(args.data)
        item_service.build_item_metadata(data)
        model_manager.fit(data)
        model_manager.save_models(args.model_dir)
        print("Training complete and model saved.")

    elif args.mode == 'predict':
        data = pd.read_excel(args.data)
        item_service.build_item_metadata(data)
        model_manager.load_models(args.model_dir)
        # Example: get prediction for a sample menu
        event_time = "Evening"
        meal_type = "Dinner"
        event_type = "Wedding"
        total_guest_count = 100
        veg_guest_count = 40
        selected_items = ["Paneer Butter Masala", "Biryani", "Curd", "Papad"]
        menu_analyzer.build_menu_context(event_time,meal_type,event_type,total_guest_count,veg_guest_count,selected_items)
        

        predictions = model_manager.predict(
            event_time, meal_type, event_type, total_guest_count, veg_guest_count,selected_items
        )
        if args.mode == "predict":
         for item, pred in predictions.items():
            print(f"{item}: {pred['total']} ({pred['unit']})")

    elif args.mode == 'price':
        model_manager.load_models(args.model_dir)
        # Example: get prediction and price for a sample menu
        event_time = "Evening"
        meal_type = "Dinner"
        event_type = "Wedding"
        total_guest_count = 100
        veg_guest_count = 40
        selected_items = ["Paneer Butter Masala", "Biryani", "Curd", "Papad"]

        predictions = model_manager.predict(
            event_time, meal_type, event_type, total_guest_count, veg_guest_count,selected_items
        )
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
            print(f"{item}: {pred['total']} ({unit}) | Total Price: ₹{total_price:.2f} | Per Person: ₹{per_person_price:.2f}")

if __name__ == "__main__":
    main()
