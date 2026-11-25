"""Make the zoidberg package importable when running pytest anywhere under
the repository (pytest by default only puts the test-file directory, not the
repo root, on ``sys.path``)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
