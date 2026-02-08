import json
from pathlib import Path
from typing import Dict, Optional

MAP_PATH = Path(__file__).resolve().parent / "landmarks_map.json"


def load_landmark_map() -> Dict[str, Optional[int]]:
    with MAP_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)
