from typing_extensions import TypedDict
from typing import List, Dict, Any

class AgentState(TypedDict):
    state: str
    model: Dict[str, Any]
    results: Dict[str, Any]
    cache_location: Dict[str, Any]
    cache_flag: Dict[str, Any]