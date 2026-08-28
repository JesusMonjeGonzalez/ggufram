# ggufram

**What will this GGUF actually cost in unified memory?**

File size is a lie. The KV cache scales with layers, KV heads, head
dimensions and context — never with the file — so two models of similar
size can differ **tenfold** in real resident cost:

| model | KV cost per 1K context tokens |
|---|---|
| gemma-4-12B | ~408 MB |
| Qwen3.6-35B-A3B | ~42.5 MB |

`ggufram` (GGUF RAM) reads only a GGUF header — a few kilobytes, arrays are
skipped with seeks — and computes the real numbers: KV cache at a given
context, resident footprint (weights + KV + compute buffers), and whether a
*set* of co-resident models fits under a wired-memory ceiling.

Pure Python standard library. Zero dependencies. The core that powers the
[RAM budget gate](https://github.com/JesusMonjeGonzalez/hearthia) in
Hearthia.

## Install

```bash
uv tool install git+https://github.com/JesusMonjeGonzalez/ggufram.git
# or as a library
uv add ggufram
```

## CLI

```text
$ ggufram gemma-notes-12b.gguf --ctx 32768
gemma-notes-12b.gguf
  architecture geometry : 48 layers · 8 KV heads · 256+256 head dims
  weights 8.1 + KV 6.0 GiB @ 32,768 tok ctx (q8_0)
  KV cost per 1K tokens : 184.3 MiB
  resident estimate     : 15.2 GiB
```

## Library

```python
from ggufram import estimate_from_profile, model_ram_profile, kv_cache_bytes, set_fits

profile = model_ram_profile(Path("gemma-notes-12b.gguf"))
est = estimate_from_profile(profile)  # header-derived, or a flagged guess
# est.resident_bytes, est.known, est.detail

kv = kv_cache_bytes(48, 8, 256, 256, ctx=32768)  # raw arithmetic if you prefer
set_fits([est_a.resident_bytes, est_b.resident_bytes], available_ram, wired_limit)
```

Why `set_fits` exists: a 35B, an embeddings model and an autocomplete model
can each fit alone — and together freeze the Mac. Wired memory cannot be
paged out; the OS strangles everything else instead of failing. **Always
check sets.**

## Design notes

- **Header only.** The reader skips arrays with seeks, so a 40 GB file costs
  the same as a 4 KB one. No tensor data is ever touched.
- **No guesses presented as facts.** When a header is unreadable the
  estimate falls back to file size × 1.3 and is flagged `known=False`.
- **No OS calls.** This package is pure arithmetic; platform concerns (wired
  limits, sysctl) belong to the caller. See Hearthia for a full control
  plane built on top.

## Used by

- [Hearthia](https://github.com/JesusMonjeGonzalez/hearthia) — the
  self-tending fire for local models: warm-on-demand, budget-enforced
  lifecycle for llama.cpp on Apple Silicon.

Using it? Open a PR to add your project.

## License

MIT
