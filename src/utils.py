"""
Utility functions for the LLM evaluation system.
"""

import json
import os
import math
from pathlib import Path
from typing import Dict, Optional


def load_api_keys(api_keys_file: Optional[Path] = None) -> Dict:
    """Load API keys from file or environment variables."""
    if api_keys_file is None:
        project_root = Path(__file__).parent
        api_keys_file = project_root / 'api_keys.json'
    
    if api_keys_file.exists():
        with open(api_keys_file, 'r') as f:
            return json.load(f)
    
    return {
        'openai': os.getenv('OPENAI_API_KEY'),
        'anthropic': os.getenv('ANTHROPIC_API_KEY'),
        'google': os.getenv('GOOGLE_API_KEY'),
        'ollama': {'base_url': 'http://localhost:11434'}
    }


def estimate_tokens(text: str) -> int:
    """Estimate token count (rough: 4 chars per token)."""
    return len(text) // 4


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers."""
    return numerator / denominator if denominator != 0 else default


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy and pandas types."""
    import numpy as np
    import pandas as pd
    
    def default(self, obj):
        if isinstance(obj, (self.np.integer, self.np.int64)):
            return int(obj)
        elif isinstance(obj, (self.np.floating, self.np.float64)):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, self.np.ndarray):
            return obj.tolist()
        elif isinstance(obj, self.pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, self.pd.Series):
            return obj.to_dict()
        elif isinstance(obj, self.pd.Timestamp):
            return obj.isoformat()
        return super().default(obj)

    def encode(self, obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return 'null'
        return super().encode(obj)
    
    def iterencode(self, obj, _one_shot=False):
        def replace_nan(o):
            if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
                return None
            elif isinstance(o, dict):
                return {k: replace_nan(v) for k, v in o.items()}
            elif isinstance(o, (list, tuple)):
                return [replace_nan(item) for item in o]
            return o
        return super().iterencode(replace_nan(obj), _one_shot)