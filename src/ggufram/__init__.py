"""ggufram — what will this GGUF actually cost in unified memory?

The pure-stdlib core extracted from Hearthia's RAM budget gate: a GGUF
header reader and the KV-cache / resident-RAM arithmetic that answers the
question file sizes cannot.
"""

from ggufram.gguf import RamProfile, model_ram_profile, read_metadata
from ggufram.ram import (
    KV_BYTES_PER_ELEMENT,
    Estimate,
    estimate_from_profile,
    estimate_resident_ram,
    kv_cache_bytes,
    set_fits,
)
from ggufram.version import __version__

__all__ = [
    "Estimate",
    "KV_BYTES_PER_ELEMENT",
    "RamProfile",
    "__version__",
    "estimate_from_profile",
    "estimate_resident_ram",
    "kv_cache_bytes",
    "model_ram_profile",
    "read_metadata",
    "set_fits",
]
