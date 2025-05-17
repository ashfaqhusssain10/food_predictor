"""
Request processor for the Food Quantity Prediction Lambda function.
Handles validation and processing of incoming requests.
"""

import json
import logging
from typing import Dict, Any, Tuple, List, Optional
from food_predictor.utils.timestamp_mapper import map_timestamp_to_model_inputs
# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class RequestProcessor:
    """
    Processes and validates incoming API requests for food prediction.
    """
    def __init__(self):
        """Initialize the request processor."""
        self.required_fields = {
            'timestamp': (str, int, float),  # Accept string, int, or float for timestamp
            'total_guest_count': int,
            'selected_items': list
        }
        self.optional_fields = {
            'veg_guest_count': int,
            'event_time': str,    # Optional if timestamp is provided
            'meal_type': str,     # Optional if timestamp is provided
            'event_type': str     # Default will be "Family Gathering" if not provided
        }
    
        
        # Valid values for categorical fields
        self.valid_event_times = ["Morning", "Afternoon", "Evening", "Night"]
        self.valid_meal_types = ["Breakfast", "Lunch", "Dinner", "Hi-tea"]
        self.valid_event_types = ["Wedding", "Birthday", "Corporate", "Festival", "Family Gathering"]### or dalna either
    
    def process_request(self, event: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, str]:
        """
        Process and validate the incoming request.
        
        Args:
            event: Lambda event object containing the request data
            
        Returns:
            Tuple of (processed_data, is_valid, error_message)
        """
        # Extract body from different event sources (API Gateway, direct invocation)
        try:
            if 'body' in event:
                if isinstance(event['body'], str):
                    body = json.loads(event['body'])
                else:
                    body = event['body']
            else:
                body = event
        except json.JSONDecodeError:
            return {}, False, "Invalid JSON in request body"
            
        # Validate required fields
        processed_data = {}
        for field, field_type in self.required_fields.items():
            if field not in body:
                return {}, False, f"Missing required field: {field}"
                
            value = body[field]
            # Allow tuple of types for flexible validation
            if isinstance(field_type, tuple):
                if not any(isinstance(value, t) for t in field_type):
                    return {}, False, f"Field {field} must be one of types: {', '.join(t.__name__ for t in field_type)}"
            elif not isinstance(value, field_type):
                return {}, False, f"Field {field} must be of type {field_type.__name__}"
                
            processed_data[field] = value
        
        # Process optional fields
        for field, field_type in self.optional_fields.items():
            if field in body:
                value = body[field]
                # Allow tuple of types for flexible validation
                if isinstance(field_type, tuple):
                    if not any(isinstance(value, t) for t in field_type):
                        return {}, False, f"Field {field} must be one of types: {', '.join(t.__name__ for t in field_type)}"
                elif not isinstance(value, field_type):
                    return {}, False, f"Field {field} must be of type {field_type.__name__}"
                processed_data[field] = value
        
        # Add default veg_guest_count if not provided (40% of total)
        if 'veg_guest_count' not in processed_data:
            processed_data['veg_guest_count'] = int(processed_data['total_guest_count'] * 0.4)
        
        # Map timestamp to event_time and meal_type if needed
        if 'timestamp' in processed_data:
            try:
                event_time, meal_type = map_timestamp_to_model_inputs(processed_data['timestamp'])
                
                # Only override if not explicitly provided
                if 'event_time' not in processed_data:
                    processed_data['event_time'] = event_time
                    
                if 'meal_type' not in processed_data:
                    processed_data['meal_type'] = meal_type
            except ValueError as e:
                return {}, False, f"Invalid timestamp format: {str(e)}"
                
        # Add default event_type if not provided
        if 'event_type' not in processed_data:
            processed_data['event_type'] = "Family Gathering"
        
        # Validate field values
        validation_errors = self._validate_field_values(processed_data)
        if validation_errors:
            return {}, False, validation_errors
            
        return processed_data, True, ""
        
    def _validate_field_values(self, data: Dict[str, Any]) -> str:
     """
     Validate specific field values.
     
     Args:
         data: Processed request data
             
     Returns:
         Error message if validation fails, empty string otherwise
     """
     # Validate event_time
     if 'event_time' not in data:
         return "Missing event_time field. Either provide timestamp or event_time directly."
             
     if data['event_time'] not in self.valid_event_times:
         return f"Invalid event_time, must be one of: {', '.join(self.valid_event_times)}"
             
     # Validate meal_type
     if 'meal_type' not in data:
         return "Missing meal_type field. Either provide timestamp or meal_type directly."
             
     if data['meal_type'] not in self.valid_meal_types:
         return f"Invalid meal_type, must be one of: {', '.join(self.valid_meal_types)}"
             
     # Validate event_type
     if 'event_type' not in data:
         return "Missing event_type field."
             
     if data['event_type'] not in self.valid_event_types:
         return f"Invalid event_type, must be one of: {', '.join(self.valid_event_types)}"
         
     # Validate numeric fields
     if data['total_guest_count'] <= 0:
         return "total_guest_count must be greater than 0"
             
     if 'veg_guest_count' in data:
         if data['veg_guest_count'] < 0 or data['veg_guest_count'] > data['total_guest_count']:
             return "veg_guest_count must be between 0 and total_guest_count"
             
     # Validate selected_items
     if 'selected_items' not in data:
         return "Missing selected_items field."
             
     if not data['selected_items']:
         return "selected_items list cannot be empty"
             
     if not all(isinstance(item, str) for item in data['selected_items']):
         return "All selected_items must be strings"
             
     return ""  # No validation errors  # No validation errors
    
    def format_response(self, prediction_results: Dict[str, Any], is_error: bool = False, error_message: str = "") -> Dict[str, Any]:
        """
        Format the API response.
        
        Args:
            prediction_results: Prediction results from the model
            is_error: Whether there was an error
            error_message: Error message if applicable
            
        Returns:
            Formatted API response
        """
        if is_error:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'success': False,
                    'error': error_message
                })
            }
        
        # Format the prediction results for API response
        formatted_results = {}
        for item, pred in prediction_results.items():
            formatted_results[item] = {
                'total': pred['total'],
                'per_person': pred['per_person'],
                'unit': pred['unit'],
                'category': pred['category'],
                'is_veg': pred['is_veg']
            }
            
            # Add price information if available
            if 'price' in pred:
                formatted_results[item]['price'] = pred['price']
                
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': True,
                'predictions': formatted_results,
                'metadata': {
                    'total_guest_count': prediction_results.get('total_guest_count'),
                    'veg_guest_count': prediction_results.get('veg_guest_count'),
                    'event_type': prediction_results.get('event_type'),
                    'meal_type': prediction_results.get('meal_type'),
                    'event_time': prediction_results.get('event_time')
                }
            })
        }