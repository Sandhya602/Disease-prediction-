import os
import json
from typing import Dict, Any

def ensure_dirs(base_out: str):
    models_dir = os.path.join(base_out, "models")
    reports_dir = os.path.join(base_out, "reports")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    return models_dir, reports_dir

def save_json(d: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
  
