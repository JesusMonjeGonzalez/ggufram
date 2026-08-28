import pytest

from ggufram import KV_BYTES_PER_ELEMENT, Estimate, estimate_from_profile, kv_cache_bytes, set_fits
from ggufram.gguf import RamProfile

GIB = 2**30

# gemma-class geometry: the KV-heavy end of the spectrum
PROFILE = RamProfile(
    n_layer=48,
    n_kv_heads=8,
    k_len=256,
    v_len=256,
    context_length=131072,
    file_size=8 * GIB,
)


def test_kv_cache_bytes_matches_closed_form():
    # 48 layers * (256+256) * 8 heads * 1.0625 B/elem (q8_0) * 32768 ctx
    assert kv_cache_bytes(48, 8, 256, 256, 32768, cache_type="q8_0") == int(
        48 * 512 * 8 * 1.0625 * 32768
    )


def test_kv_cache_scales_with_context_not_file_size():
    kv_32k = kv_cache_bytes(48, 8, 256, 256, 32768)
    kv_64k = kv_cache_bytes(48, 8, 256, 256, 65536)
    assert kv_64k == 2 * kv_32k


def test_kv_cache_quantisation_ladder_orders():
    ladder = ["f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"]
    sizes = [kv_cache_bytes(48, 8, 256, 256, 32768, cache_type=ct) for ct in ladder]
    assert sizes == sorted(sizes, reverse=True)


def test_kv_cache_unknown_type_raises():
    with pytest.raises(ValueError, match="unknown cache type"):
        kv_cache_bytes(48, 8, 256, 256, 32768, cache_type="q9_9")


def test_estimate_from_profile_is_known():
    est = estimate_from_profile(PROFILE, ctx=32768)
    kv = kv_cache_bytes(48, 8, 256, 256, 32768)
    assert est.known is True
    assert est.resident_bytes == 8 * GIB + kv + max(int(8 * GIB * 0.05), 256 * 1024**2)
    assert "32,768" in est.detail


def test_estimate_defaults_to_training_context():
    est = estimate_from_profile(PROFILE)
    assert est.known is True
    assert "131,072" in est.detail


def test_estimate_without_profile_is_a_flagged_guess():
    est = estimate_from_profile(None, file_size=10 * GIB)
    assert est.known is False
    assert est.resident_bytes == int(10 * GIB * 1.3)
    assert "guess" in est.detail


def test_estimate_guess_floors_at_min_overhead():
    est = estimate_from_profile(None, file_size=0)
    assert est.resident_bytes == 256 * 1024**2


def test_estimate_file_size_override():
    est = estimate_from_profile(PROFILE, ctx=4096, file_size=4 * GIB)
    assert est.resident_bytes == 4 * GIB + kv_cache_bytes(48, 8, 256, 256, 4096) + 256 * 1024**2


def test_set_fits_checks_the_set_not_the_members():
    wired = 24 * GIB
    available = 24 * GIB
    a, b, c = 10 * GIB, 10 * GIB, 5 * GIB
    assert set_fits([a], available, wired)
    assert set_fits([b], available, wired)
    assert set_fits([c], available, wired)
    assert not set_fits([a, b, c], available, wired)  # 25 GiB together
    assert set_fits([a, c], available, wired)


def test_estimate_dataclass_shape():
    e = Estimate(123, True, "detail")
    assert e.resident_bytes == 123 and e.known is True and e.detail == "detail"


def test_kv_table_covers_documented_types():
    for ct in ("f32", "f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"):
        assert ct in KV_BYTES_PER_ELEMENT
