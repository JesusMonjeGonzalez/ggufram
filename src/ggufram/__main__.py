"""ggufram — one-line cost report for a GGUF file."""

import argparse
import sys
from pathlib import Path

from ggufram.gguf import model_ram_profile
from ggufram.ram import estimate_from_profile, kv_cache_bytes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ggufram",
        description=(
            "What will this GGUF cost in unified memory? Header-derived "
            "KV-cache and resident-RAM arithmetic — pure stdlib, no model "
            "data is read beyond the header."
        ),
    )
    parser.add_argument("gguf", type=Path, help="path to a .gguf file")
    parser.add_argument("--ctx", type=int, default=0, help="context length override")
    parser.add_argument(
        "--cache", default="q8_0", help="KV cache type (default q8_0; f16, q8_0, q5_1, q4_0…)"
    )
    args = parser.parse_args(argv)

    profile = model_ram_profile(args.gguf)
    if profile is None:
        print(f"{args.gguf.name}: header unreadable — cannot estimate", file=sys.stderr)
        return 1

    est = estimate_from_profile(profile, ctx=args.ctx or None, cache_type=args.cache)
    per_1k = kv_cache_bytes(
        profile.n_layer,
        profile.n_kv_heads,
        profile.k_len,
        profile.v_len,
        1024,
        cache_type=args.cache,
    )
    print(f"{args.gguf.name}")
    geo = (
        f"{profile.n_layer} layers · {profile.n_kv_heads} KV heads · "
        f"{profile.k_len}+{profile.v_len} head dims"
    )
    print(f"  architecture geometry : {geo}")
    print(f"  {est.detail}")
    print(f"  KV cost per 1K tokens : {per_1k / 2**20:.1f} MiB")
    print(f"  resident estimate     : {est.resident_bytes / 2**30:.1f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
