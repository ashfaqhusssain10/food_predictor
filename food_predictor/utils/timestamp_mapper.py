"""
Utility functions for mapping timestamps to event times and meal types.
"""

import datetime
from typing import Dict, Tuple, Any, Optional

def map_timestamp_to_model_inputs(timestamp: Any) -> Tuple[str, str]:
    """
    Map a timestamp to appropriate event_time and meal_type values.
    
    Args:
        timestamp: Timestamp as string (ISO format) or datetime object
        
    Returns:
        Tuple of (event_time, meal_type)
    """
    # Handle different timestamp formats
    if isinstance(timestamp, str):
        try:
            # Try to parse ISO format
            dt = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            try:
                # Try common formats
                dt = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                # Fallback
                raise ValueError(f"Unsupported timestamp format: {timestamp}")
    elif isinstance(timestamp, (int, float)):
        # Unix timestamp
        dt = datetime.datetime.fromtimestamp(timestamp)
    elif isinstance(timestamp, datetime.datetime):
        dt = timestamp
    else:
        raise ValueError(f"Unsupported timestamp type: {type(timestamp)}")
    
    # Get the hour (0-23)
    hour = dt.hour
    
    # Map hour to event_time
    if 5 <= hour < 12:
        event_time = "Morning"
    elif 12 <= hour < 17:
        event_time = "Afternoon"
    elif 17 <= hour < 21:
        event_time = "Evening"
    else:  # 21-23, 0-4
        event_time = "Night"
    
    # Map hour to meal_type
    if 5 <= hour < 11:
        meal_type = "Breakfast"
    elif 11 <= hour < 16:
        meal_type = "Lunch"
    elif 16 <= hour < 18:
        meal_type = "Hi-tea"
    elif 18 <= hour < 23:
        meal_type = "Dinner"
    else:  # Late night hours (23-4)
        meal_type = "Hi-tea"  # Default to snacks for late night
    
    return event_time, meal_type