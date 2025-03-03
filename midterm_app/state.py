from typing import TypedDict, List, Dict, Any

class FounderAnalysisState(TypedDict, total=False):
    query: str
    query_type: str
    filter_key: str
    filter_value: str
    retrieved_profiles: List[Dict[str, Any]]
    web_search_results: List[Dict[str, Any]]
    analysis_results: List[Dict[str, Any]]
    final_response: Dict[str, Any]
    error: str