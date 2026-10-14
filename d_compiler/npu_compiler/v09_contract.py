"""Shared storage, operation and native ABI contract (generated C++ peer)."""
import hashlib
import json
from pathlib import Path

_BYTES = Path(__file__).with_suffix(".json").read_bytes()
CONTRACT = json.loads(_BYTES)
CONTRACT_HASH = hashlib.sha256(_BYTES).hexdigest()
MAX_VLEN = CONTRACT["max_vlen"]
NUMERIC_POLICY = CONTRACT["numeric_policy"]
