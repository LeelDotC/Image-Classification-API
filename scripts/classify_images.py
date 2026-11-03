"""Classify local images with one reusable model instance and print actual JSON."""

import argparse
import json
from pathlib import Path
import sys

from PIL import Image

from app.model_service import ImageClassifier


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="Local image paths")
    args = parser.parse_args()

    try:
        classifier = ImageClassifier()
    except Exception as exc:
        print(f"Could not load classifier: {exc}", file=sys.stderr)
        return 1

    failed = False
    for path in args.images:
        try:
            with Image.open(path) as image:
                result = classifier.classify(image)
            print(json.dumps({"image": str(path), **result}, indent=2, allow_nan=False))
        except Exception as exc:
            print(f"Could not classify {path}: {exc}", file=sys.stderr)
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
