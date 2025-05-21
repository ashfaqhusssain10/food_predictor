"""
Maps time slot strings to model inputs (event_time and meal_type).
"""

import logging

logger = logging.getLogger()

def map_timestamp_to_model_inputs(time_slot):
    """
    Maps a time slot string to event_time and meal_type.
    
    Args:
        time_slot: String representation of time slot (e.g. "7:30 AM-8:30 AM")
        
    Returns:
        Tuple of (event_time, meal_type)
    """
    # Clean up the input if needed
    if isinstance(time_slot, (int, float)):
        # If somehow a timestamp is passed, convert to string
        time_slot = str(time_slot)
    
    time_slot = time_slot.strip().upper()
    
    # Breakfast slots (Morning)
    breakfast_slots = [
        "7:30 AM-8:30 AM", 
        "8:30 AM-9:30 AM", 
        "9:30 AM-10:30 AM"
    ]
    
    # Lunch slots (Afternoon)
    lunch_slots = [
        "12:30 PM-1:30 PM", 
        "1:30 PM-2:30 PM", 
        "2:30 PM-3:30 PM"
    ]
    
    # Snacks/Hi-tea slots (Evening)
    snacks_slots = [
        "3:30 PM - 4:30 PM", 
        "4:30 PM - 5:30 PM", 
        "5:30 PM - 6:30 PM"
    ]
    
    # Dinner slots (Night)
    dinner_slots = [
        "6:30 PM - 7:30 PM", 
        "7:30 PM - 8:30 PM", 
        "8:30 PM - 9:30 PM"
    ]
    
    # Normalize input by removing spaces around hyphens
    normalized_slot = time_slot.replace(" - ", "-").replace(" -", "-").replace("- ", "-")
    
    logger.info(f"Mapping time slot: {time_slot} (normalized: {normalized_slot})")
    
    # Map to event_time and meal_type
    if any(slot.upper() in normalized_slot for slot in breakfast_slots):
        event_time = "Morning"
        meal_type = "Breakfast"
    elif any(slot.upper() in normalized_slot for slot in lunch_slots):
        event_time = "Afternoon" 
        meal_type = "Lunch"
    elif any(slot.upper() in normalized_slot for slot in snacks_slots):
        event_time = "Evening"
        meal_type = "Hi-tea"  
    elif any(slot.upper() in normalized_slot for slot in dinner_slots):
        event_time = "Night"
        meal_type = "Dinner"
    else:
        # Default case if no match is found
        logger.warning(f"Could not map time slot: {time_slot}, using defaults")
        event_time = "Evening"
        meal_type = "Dinner"
    
    logger.info(f"Mapped to event_time: {event_time}, meal_type: {meal_type}")
    return event_time, meal_type