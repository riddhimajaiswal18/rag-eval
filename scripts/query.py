#!/usr/bin/env python3
"""Query one isolated corpus using the existing retrieval implementation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.search import main

if __name__ == "__main__":
    main()
