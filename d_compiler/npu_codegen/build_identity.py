"""Source identity shared by the native builder and Python loader."""
import hashlib
from pathlib import Path


def source_hash(directory=None):
    directory = Path(directory or __file__).resolve().parent if directory is None else Path(directory)
    digest = hashlib.sha256()
    paths = sorted([*directory.glob("*.cc"), *directory.glob("*.h")], key=lambda p: p.name)
    for path in paths:
        if path.name == "build_identity.h":
            continue
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    (here / "build_identity.h").write_text(
        '#pragma once\n#define NPU_BUILD_SOURCE_HASH "' + source_hash(here) + '"\n')
