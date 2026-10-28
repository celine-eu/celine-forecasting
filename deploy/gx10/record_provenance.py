"""Record which serving image (and source commit) serves a Chronos-2 model dir.

Writes ``serving_provenance.json`` next to ``model_card.json`` in
``models/<version>/``, so the box can answer "which code is serving this
model" (assessment 2026-10-23, item C3). Stdlib only: it runs with the box's
``python3``, outside any venv.

The image must have been built from a git tag with the source commit baked in
as a label (see ``Dockerfile.serve``, ``ARG SOURCE_COMMIT``). The record is
refused unless that label equals the commit the tag points to, so an image
built from a dirty or untagged tree cannot be recorded as the tag.

Usage on the box, after building from tag ``$TAG`` (``$SRC`` = the checkout)::

    IMG="chronos2-serve:$TAG"
    REV='{{index .Config.Labels "org.opencontainers.image.revision"}}'
    python3 "$SRC/deploy/gx10/record_provenance.py" \\
        --model-dir "$CHRONOS_HOME/models/<version>" \\
        --image "$IMG" \\
        --image-id "$(docker image inspect -f '{{.Id}}' "$IMG")" \\
        --source-commit "$(docker image inspect -f "$REV" "$IMG")" \\
        --tag-commit "$(git -C "$SRC" rev-parse "$TAG^{commit}")" \\
        --source-tag "$TAG"

``image_id`` is the local image ID (``sha256:`` of the image config): the image
is built on the box and never pushed, so there is no registry digest.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

PROVENANCE_FILE = "serving_provenance.json"
SOURCE_REPO = "celine-eu/celine-forecasting"
SCHEMA = 1

logger = logging.getLogger("record_provenance")

_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-dir", type=Path, required=True, help="models/<version> dir")
    p.add_argument("--image", required=True, help="image reference, e.g. chronos2-serve:v0.3.0")
    p.add_argument("--image-id", required=True, help="docker image inspect -f '{{.Id}}'")
    p.add_argument("--source-commit", required=True, help="image label image.revision")
    p.add_argument("--tag-commit", required=True, help="git rev-parse <tag>^{commit}")
    p.add_argument("--source-tag", required=True, help="git tag the image was built from")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Validate the inputs and write ``serving_provenance.json``.

    Args:
        argv: Command-line arguments (default: ``sys.argv[1:]``).

    Returns:
        0 on success.

    Raises:
        SystemExit: On a missing model card, a malformed value, or an image
            whose source commit is not the commit of the tag.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = _parse(argv)
    card_path = a.model_dir / "model_card.json"
    if not card_path.is_file():
        raise SystemExit(f"no model_card.json in {a.model_dir}")
    if not _IMAGE_ID.fullmatch(a.image_id):
        raise SystemExit(f"--image-id must be sha256:<64 hex>, got {a.image_id!r}")
    for flag, value in (("--source-commit", a.source_commit), ("--tag-commit", a.tag_commit)):
        if not _COMMIT.fullmatch(value):
            raise SystemExit(f"{flag} must be a full 40-hex commit, got {value!r}")
    if not a.source_tag.strip():
        raise SystemExit("--source-tag must not be empty")
    if a.source_commit != a.tag_commit:
        raise SystemExit(
            f"image {a.image} was not built from tag {a.source_tag}: "
            f"label {a.source_commit} != tag commit {a.tag_commit}"
        )

    card = json.loads(card_path.read_text())
    record = {
        "schema": SCHEMA,
        "model_version": card.get("model_version"),
        "image": a.image,
        "image_id": a.image_id,
        "source_repo": SOURCE_REPO,
        "source_tag": a.source_tag,
        "source_commit": a.source_commit,
        "recorded_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out = a.model_dir / PROVENANCE_FILE
    out.write_text(json.dumps(record, indent=2) + "\n")
    out.chmod(0o644)
    logger.info("provenance written: %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
