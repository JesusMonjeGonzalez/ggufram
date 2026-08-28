"""Unified-memory arithmetic for GGUF models.

KV cache scales with layers, KV heads, head dimensions and context — never
with file size. Two models of similar size can differ tenfold in real cost:
gemma-4-12B costs 408 MB per 1K context tokens, Qwen3.6-35B-A3B costs
42.5 MB. File size is a lie; the header is the truth.

Everything here is pure arithmetic — no OS calls, no I/O, no dependencies.
"""

from dataclasses import dataclass

from ggufram.gguf import RamProfile

# Bytes per cached element, including the block scales the quantised
# formats carry (q8_0 stores 32 values plus one f16 scale, and so on).
KV_BYTES_PER_ELEMENT: dict[str, float] = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 8.5 / 8,
    "q5_1": 6.0 / 8,
    "q5_0": 5.5 / 8,
    "q4_1": 5.0 / 8,
    "q4_0": 4.5 / 8,
}

# Compute-buffer floor: at least this much beyond weights + KV.
_MIN_OVERHEAD = 256 * 1024**2


def kv_cache_bytes(
    n_layer: int,
    n_kv_heads: int,
    k_len: int,
    v_len: int,
    ctx: int,
    cache_type: str = "q8_0",
) -> int:
    """Exact KV cache size for a model at a given context length.

    All parameters come from the GGUF header (``block_count``,
    ``attention.head_count_kv``, ``attention.key_length``,
    ``attention.value_length``) — ``RamProfile`` carries them.
    """
    try:
        bytes_per_element = KV_BYTES_PER_ELEMENT[cache_type]
    except KeyError:
        raise ValueError(
            f"unknown cache type {cache_type!r}; "
            f"expected one of {', '.join(sorted(KV_BYTES_PER_ELEMENT))}"
        ) from None
    per_token = n_layer * (k_len + v_len) * n_kv_heads * bytes_per_element
    return int(per_token * ctx)


def estimate_resident_ram(file_size: int, kv_bytes: int, overhead_ratio: float = 0.05) -> int:
    """RAM a loaded model actually holds: weights + KV cache + compute buffers.

    Weights are memory-mapped but become resident once the GPU wires them, so
    the file size is the right figure on Apple Silicon.
    """
    overhead = max(int(file_size * overhead_ratio), _MIN_OVERHEAD)
    return file_size + kv_bytes + overhead


@dataclass(frozen=True)
class Estimate:
    """Resident-RAM estimate for one model."""

    resident_bytes: int
    known: bool  # True when derived from the GGUF header, False when a guess
    detail: str


def estimate_from_profile(
    profile: RamProfile | None,
    ctx: int | None = None,
    cache_type: str = "q8_0",
    file_size: int | None = None,
    fallback_ratio: float = 1.3,
) -> Estimate:
    """Resident-RAM estimate from a ``RamProfile`` (or a file-size guess).

    With a profile the result is header-derived: weights + KV cache at the
    requested context (defaults to the model's training context) + compute
    buffers. Without one, the estimate is file size × ``fallback_ratio`` —
    a guess that is wrong in both directions, flagged with ``known=False``.
    """
    if profile is None:
        size = file_size or 0
        guess = max(int(size * fallback_ratio), size + _MIN_OVERHEAD)
        return Estimate(guess, False, "file-size guess — GGUF header unreadable")

    size = file_size if file_size is not None else profile.file_size
    context = ctx or profile.context_length
    kv = kv_cache_bytes(
        profile.n_layer,
        profile.n_kv_heads,
        profile.k_len,
        profile.v_len,
        context,
        cache_type=cache_type,
    )
    est = estimate_resident_ram(size, kv)
    detail = (
        f"weights {size / 2**30:.1f} + KV {kv / 2**30:.1f} GiB @ {context:,} tok ctx ({cache_type})"
    )
    return Estimate(est, True, detail)


def set_fits(estimates: list[int], available_ram: int, wired_limit: int) -> bool:
    """Whether a set of co-resident models fits.

    Checking models one at a time is what lets a machine freeze: a 35B, an
    embeddings model and an autocomplete model can each fit alone, yet
    together exceed the wired ceiling. Wired memory cannot be paged out, so
    the OS strangles everything else instead of failing. Always check sets.
    """
    total = sum(estimates)
    return total < wired_limit and total < available_ram
