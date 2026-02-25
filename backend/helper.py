import json
from typing import Dict, Any
import re

def _safe_json_extract(text: str) -> Dict[str, Any]:
    # If model returns extra text, try to grab the first {...}
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {"status": "INVALID", "reply": "Invalid request."}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"status": "INVALID", "reply": "Invalid request."}