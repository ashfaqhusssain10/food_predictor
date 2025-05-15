import pandas as pd
import numpy as np
import logging
import argparse
import os
from datetime import datetime

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
    parser.add_argument('--mode', choices=['train', 'predict', 'price', 'evaluate','cv'], default='train')
    parser.add_argument('--model_dir', type=str, default='./models')
    parser.add_argument('--cv_folds', type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument('--cv_output', type=str, default='cv_results', 
                        help="Output directory for cross-validation results")
    parser.add_argument('--sample_size', type=int, default=None, 
                        help="Number of orders to randomly sample for cross-validation (default: use all data)")
    #parser.add_argument('--batch_size', type=int, default=1000,  help="Number of orders to process in each batch (memory optimization)")
    parser.add_argument('--min_item_samples', type=int, default=5, 
                        help="Minimum samples required to train an item-specific model")
    
    
    
    
    
    
    parser.add_argument('--eval_output', type=str, default='evaluation_results.xlsx', 
                        help="Path to save evaluation results (only used in 'evaluate' mode)")
    parser.add_argument('--num_orders', type=int, default=100, 
                        help="Number of orders to randomly evaluate (only used in 'evaluate' mode)")
    parser.add_argument('--eval_type', choices=['random', 'test_set'], default='random',
                        help="Evaluation type: random samples or external test set")
    parser.add_argument('--test_data', type=str, default=None,
                        help="Path to external test dataset (only used with --eval_type=test_set)")
    parser.add_argument('--batch_size', type=int, default=50,
                        help="Batch size for test set evaluation (memory optimization)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger(__name__)

    # Initialize core components
    food_rules = FoodCategoryRules()
    item_service = ItemService(food_rules)
    feature_engineer = FeatureEngineer(food_rules)
    menu_analyzer = MenuAnalyzer(food_rules, item_service)
    model_manager = ModelManager(feature_engineer, item_service, food_rules, menu_analyzer)
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
        for item, pred in predictions.items():
            print(f"{item}: {pred['total']} ({pred['unit']})")

    elif args.mode == 'price':
        data = pd.read_excel(args.data)
        item_service.build_item_metadata(data)
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
    
    elif args.mode == 'evaluate':
        # Load models for either evaluation type
        data = pd.read_excel(args.data)
        item_service.build_item_metadata(data)
        model_manager.load_models(args.model_dir)
    
        if args.eval_type == 'random':
            logger.info(f"Evaluating Model on Random {args.num_orders} Orders")
            
            # Use the random evaluation method
            evaluation_results = model_manager.evaluate_item_accuracy(
                data=data,
                food_rules=food_rules,
                feature_engineer=feature_engineer,
                item_matcher=item_service.item_matcher,
                num_orders=args.num_orders,
                random_state=42,
                output_file=args.eval_output
            )
            
            logger.info(f"Random evaluation complete! Results saved to {args.eval_output}")
            
        elif args.eval_type == 'test_set':
            if not args.test_data:
                logger.error("Error: --test_data parameter is required for test_set evaluation")
                return
                
            logger.info(f"Evaluating Model on External Test Dataset: {args.test_data}")
            
            # Use the external test set evaluation method
            evaluation_results = model_manager.evaluate_on_external_test_set(
                test_data_path=args.test_data,
                output_file=args.eval_output,
                batch_size=args.batch_size
            )
    
            logger.info(f"External test set evaluation complete! Results saved to {args.eval_output}")

    elif args.mode == 'cv':
        # Load data for cross-validation
        data = pd.read_excel(args.data)
        item_service.build_item_metadata(data)
        
        # Create output directory if it doesn't exist
        import os
        from datetime import datetime
        os.makedirs(args.cv_output, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(args.cv_output, f"cv_results_{timestamp}.xlsx")
        
        
        
        # Log parameters for streamlined cross-validation
        logger.info(f"Starting {args.cv_folds}-fold cross-validation")
        logger.info(f"Parameters: batch_size={args.batch_size}, min_item_samples={args.min_item_samples}")
        
        if args.sample_size:
            logger.info(f"Using random sample of {args.sample_size} orders from total {data['Order_Number'].nunique()} orders")
        
        # Run cross-validation with parameters for streamlined processing
        cv_results = model_manager.cross_validate(
            data=data,
            k_folds=args.cv_folds,
            random_state=42,
            batch_size=args.batch_size,
            sample_size=args.sample_size,
            min_item_samples=args.min_item_samples,
            output_file=output_file
        )
        
        # Log the overall results
        logger.info("Cross-validation completed")
        logger.info(f"Overall MAE: {cv_results['overall']['mean_mae']:.4f} ± {cv_results['overall']['std_mae']:.4f}")
        logger.info(f"Overall RMSE: {cv_results['overall']['mean_rmse']:.4f} ± {cv_results['overall']['std_rmse']:.4f}")
        logger.info(f"Overall R²: {cv_results['overall']['mean_r2']:.4f} ± {cv_results['overall']['std_r2']:.4f}")
        
        # Log top and bottom performing categories
        if 'categories' in cv_results:
            categories = []
            for category, metrics in cv_results['categories'].items():
                if 'mean_mae' in metrics:
                    categories.append((category, metrics['mean_mae'], metrics.get('count', 0)))
            
            # Sort by MAE (lower is better)
            categories.sort(key=lambda x: x[1])
            
            logger.info("Top 5 best performing categories:")
            for category, mae, count in categories[:5]:
                logger.info(f"  - {category}: MAE = {mae:.4f}, samples = {count}")
                
            logger.info("Top 5 worst performing categories:")
            for category, mae, count in categories[-5:]:
                logger.info(f"  - {category}: MAE = {mae:.4f}, samples = {count}")
        
        # Train final model on all data after cross-validation
        logger.info("Training final model on full dataset...")
        full_data = pd.read_excel(args.data)  # Reload to get full dataset
        item_service.build_item_metadata(full_data)
        model_manager.fit(full_data)
        model_path = model_manager.save_models(args.model_dir)
        logger.info(f"Final model saved to {model_path}")

if __name__ == "__main__":
    main()