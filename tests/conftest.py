# pytest configuration
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "generation"))
sys.path.insert(0, str(ROOT / "src" / "data_extraction"))
sys.path.insert(0, str(ROOT / "src" / "transformation"))
sys.path.insert(0, str(ROOT / "src" / "embeddings"))
