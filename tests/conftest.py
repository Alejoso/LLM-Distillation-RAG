"""
Pytest configuration for the Datasets-experiment tests.

Adds the repository root to sys.path so that all packages under
Scripts/ are importable as `Scripts.<subpackage>.<module>`.

Also registers `text_normalization` as a top-level module so that
preprocessHTMLs.py can resolve its bare `from text_normalization import ...`
regardless of the working directory.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Register text_normalization as a top-level module to satisfy the bare import
# inside preprocessHTMLs.py: `from text_normalization import normalize_body`
_tn_path = ROOT / "Scripts" / "ProcessHTMLs" / "text_normalization.py"
_spec = importlib.util.spec_from_file_location("text_normalization", _tn_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
sys.modules.setdefault("text_normalization", _mod)
