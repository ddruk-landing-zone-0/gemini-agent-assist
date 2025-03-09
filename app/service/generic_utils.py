import os
import json
from typing import Dict, Any
from .logger import LOGGER

def load_json_data(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return None
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def save_json_data(path: str, data: Dict[str, Any]):
    # Create the directory if it doesn't exist
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)

def load_cached_results(state):
    # Load the data from cache if the cache flag is set to True
    if state['cache_flag'][state['state']]:
        cached_result = load_json_data(state['cache_location'][state['state']])
        if cached_result:
            state['results'][state['state']] = cached_result['result']
            LOGGER.info(f"State: {state['state']} | Loaded cached data and skipping the model, {len(state['results'][state['state']])} old result found")
            return state
    return None