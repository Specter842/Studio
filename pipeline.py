"""Convenience shim so the documented command works from the project root:

    python pipeline.py --clips ./inputs/local_clips --audio ./inputs/track.mp3 \
                       --out ./output/final.mp4

The real implementation lives in src/pipeline.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pipeline import main  # noqa: E402  (src/pipeline.py, not this file)

if __name__ == "__main__":
    raise SystemExit(main())
