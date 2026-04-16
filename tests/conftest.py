"""
Pytest configuration for the project tests.

Adds the repository root and module directories to sys.path so that
all packages are importable in the new unified project structure.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Add module directories to sys.path
for subdir in ["01_data_collection", "02_data_processing", "03_rag", "04_distillation", "05_evaluation"]:
    path = str(ROOT / subdir)
    if path not in sys.path:
        sys.path.insert(0, path)

# Register text_normalization as a top-level module to satisfy the bare import
# inside preprocess_htmls.py: `from text_normalization import normalize_body`
_tn_path = ROOT / "01_data_collection" / "text_normalization.py"
if _tn_path.exists():
    _spec = importlib.util.spec_from_file_location("text_normalization", _tn_path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    sys.modules.setdefault("text_normalization", _mod)
