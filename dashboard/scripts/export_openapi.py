"""Export the exact FastAPI schema without requiring a running server or database."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from api.main import create_app

path = Path(__file__).resolve().parents[1] / "openapi.json"
path.write_text(json.dumps(create_app().openapi(), indent=2) + "\n")
print(path)
