import streamlit as st
import pandas as pd
import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from food_predictor.core.model_manager import ModelManager
from food_predictor.core.feature_engineering import FeatureEngineer
from food_predictor.core.category_rules import FoodCategoryRules
from food_predictor.data.item_service import ItemService
from food_predictor.core.menu_analyzer import MenuAnalyzer
from food_predictor.pricing.price_calculator import PriceCalculator
from food_predictor.utils.quantity_utils import extract_quantity_value

def load_models():
    """Load the prediction models and components"""
    food_rules = FoodCategoryRules()
    item_service = ItemService(food_rules)
    feature_engineer = FeatureEngineer(food_rules)
    menu_analyzer = MenuAnalyzer(food_rules, item_service)
    model_manager = ModelManager(feature_engineer, item_service, food_rules, menu_analyzer)
    price_calculator = PriceCalculator(food_rules, item_service)
    
    # Load training data to build item metadata
    data_path = os.path.join(os.path.dirname(__file__), "DB38.xlsx")
    data = pd.read_excel(data_path)
    item_service.build_item_metadata(data)
    
    # Load pre-trained models
    models_path = os.path.join(os.path.dirname(__file__), "models")
    model_manager.load_models(models_path)
    
    return model_manager, price_calculator, item_service

def create_app():
    """Main Streamlit application"""
    st.set_page_config(page_title="Food Quantity Predictor", layout="wide")
    
    # Application header
    st.title("Food Quantity Prediction System")
    st.markdown("Predict food quantities for events based on guest count and menu selection")
    
    # Load models (with caching for performance)
    @st.cache_resource
    def get_models():
        return load_models()
    
    try:
        model_manager, price_calculator, item_service = get_models()
        st.success("✅ Models loaded successfully")
    except Exception as e:
        st.error(f"❌ Failed to load models: {str(e)}")
        st.stop()
    
    # Create sidebar for event details
    with st.sidebar:
        st.header("Event Details")
        
        event_time = st.selectbox(
            "Event Time",
            options=["Morning", "Afternoon", "Evening", "Night"],
            index=2  # Default to Evening
        )
        
        meal_type = st.selectbox(
            "Meal Type",
            options=["Breakfast", "Lunch", "Dinner", "Snacks"],
            index=2  # Default to Dinner
        )
        
        event_type = st.selectbox(
            "Event Type",
            options=["Wedding", "Birthday", "Corporate", "Festival", "Family Gathering"],
            index=0  # Default to Wedding
        )
        
        total_guest_count = st.number_input(
            "Total Guest Count",
            min_value=1,
            max_value=1000,
            value=100
        )
        
        include_veg = st.checkbox("Include Vegetarian Guest Count", value=True)
        
        veg_guest_count = None
        if include_veg:
            veg_guest_count = st.number_input(
                "Vegetarian Guest Count",
                min_value=0,
                max_value=total_guest_count,
                value=min(40, total_guest_count)  # Default 40% or max of total
            )
    
    # Main content - Menu Selection and Results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Menu Selection")
        
        # Get unique categories from item service
        categories = sorted(set(metadata.category for metadata in item_service.item_metadata.values()))
        
        # Create multiselect for each category
        selected_items = []
        for category in categories:
            category_items = [item_name for item_name, metadata in item_service.item_metadata.items() 
                     if metadata.category == category]
            
            unique_items = []
            seen_std_names = set()
            for item in category_items:
                std_name = item_service.standardize_item_name(item)
                if std_name not in seen_std_names:
                    unique_items.append(item)
                    seen_std_names.add(std_name)
                     
            
            
            if unique_items:  # Only show if category has items
                with st.expander(f"{category} ({len(unique_items)} items)"):
                    selected_category_items = st.multiselect(
                        f"Select {category} items",
                        options=sorted(unique_items),
                        key=f"select_{category}"
                    )
                    selected_items.extend(selected_category_items)
        
        # Show selected items count
        st.info(f"Total items selected: {len(selected_items)}")
        
        # Prediction button
        predict_button = st.button("Generate Predictions", type="primary", disabled=(len(selected_items) == 0))
    
    # Results display
    with col2:
        st.header("Prediction Results")
        
        if predict_button and len(selected_items) > 0:
            with st.spinner("Generating predictions..."):
                try:
                    # Call the prediction function
                    predictions = model_manager.predict(
                        event_time, meal_type, event_type, 
                        total_guest_count, veg_guest_count, selected_items
                    )
                    
                    # Display results in a table
                    results_data = []
                    for item, pred in predictions.items():
                        qty = extract_quantity_value(pred['total'])
                        unit = pred['unit']
                        category = pred['category']
                        is_veg = pred['is_veg']
                        
                        # Calculate price if available
                        try:
                            total_price, base_price_per_unit, per_person_price = price_calculator.calculate_price(
                                converted_qty=qty,
                                category=category,
                                total_guest_count=total_guest_count,
                                item_name=item,
                                unit=unit
                            )
                        except:
                            total_price, per_person_price = None, None
                        
                        results_data.append({
                            "Item": item,
                            "Category": category,
                            "Quantity": pred['total'],
                            "Per Person": pred['per_person'],
                            "Veg/Non-veg": is_veg,
                            "Total Price": f"₹{total_price:.2f}" if total_price is not None else "N/A",
                            "Price Per Person": f"₹{per_person_price:.2f}" if per_person_price is not None else "N/A"
                        })
                    
                    # Create DataFrame for display
                    results_df = pd.DataFrame(results_data)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Calculate totals
                    if any(row["Total Price"] != "N/A" for row in results_data):
                        total_price_sum = sum(price_calculator.calculate_price(
                            extract_quantity_value(pred['total']),
                            pred['category'],
                            total_guest_count,
                            item,
                            pred['unit']
                        )[0] for item, pred in predictions.items())
                        
                        st.metric("Total Estimated Cost", f"₹{total_price_sum:.2f}")
                        st.metric("Cost Per Person", f"₹{(total_price_sum/total_guest_count):.2f}")
                    
                    # Add export button for results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="food_predictions.csv",
                        mime="text/csv"
                    )
                
                except Exception as e:
                    st.error(f"Error generating predictions: {str(e)}")
        else:
            st.info("Select items from the menu and click 'Generate Predictions' to see results")
    
    # Advanced settings in an expander (optional)
    with st.expander("Advanced Settings"):
        st.subheader("Main Item Limitation")
        max_main_items = st.slider(
            "Maximum main items per category",
            min_value=1,
            max_value=5,
            value=2,
            help="Limit the number of main items identified per category to improve prediction accuracy"
        )
        
        st.caption("Note: Reducing main items per category may improve quantity predictions")
    
    # Footer
    st.markdown("---")
    st.caption("Food Quantity Prediction System © 2025")

if __name__ == "__main__":
    create_app()