# Stage 3 (Nemotron Terminal) — pinned source, re-chunk, and cache provenance

Everything needed to prove which data produced the Stage 3 cache, and to rebuild it.
These files previously existed only in `$SCRATCH/tmp` on Vista, which is a single
uncopied location; the cache is correct but was unverifiable without them.

Stage 3 substitutes **Nemotron Terminal** for OpenCode, because the exact fixed-EOT
OpenCode pre-tokenized cache could not be obtained. It follows the
[marin#8172](https://github.com/marin-community/marin/pull/8172) pinned selection.

## provenance/

| File | What it is |
|---|---|
| `upstream_pinned_files.json` | The 29 relative paths, read from marin#8172 at frozen head `9383df73c`. The independent source of truth — never derived from a directory walk. |
| `stage3_source_hashes.json` | sha256 of each file **as pinned**, before re-chunking. Read-only; the gate no longer rewrites it. |
| `stage3_ondisk_hashes.json` | sha256 of each file as it exists after re-chunking. |
| `RECHUNK_PROVENANCE.json` | The transformation record for the 3 rewritten files: original and re-chunked sha256, row counts, row groups before/after, and `content_identical`. |
| `nemotron_stage3_paths.txt` | The explicit 29 absolute paths handed to `prepare-data`. No glob. |
| `cache_v4_snowball_provenance.json` | The sidecar written next to the built cache: 29 shard hashes, dataset revision, tokenizer sha, format identity. |

Dataset: `nvidia/Nemotron-Terminal-Corpus` at revision `a1667c4ffdadea02a89bffe4f1bb7ca2ff19f8d9`.

## The chain

`upstream_pinned_files.json` → `stage3_source_hashes.json` → `RECHUNK_PROVENANCE.json`
→ `stage3_ondisk_hashes.json` → `cache_v4_snowball_provenance.json`

It verifies offline from this directory alone:

- 29 = 29 = 29 = 29 across the pinned selection, original hashes, on-disk hashes, and path list
- 26/26 untouched files byte-identical to their pinned originals
- 3 re-chunked files match `rechunked_sha256`, and their `original_sha256` matches the pinned original, with `content_identical: true`
- the cache sidecar's 29 shard keys equal the pinned set

## Why the re-chunk was needed

`code.parquet`, `math.parquet` and `swe.parquet` each held a single row group above
2 GB. `pyarrow.read_row_group()` raises `ArrowNotImplementedError` on row groups that
large with nested columns, so tokenization could not read them. They were rewritten at
`rows_per_group=2000`, streaming both files in lockstep and hashing the `conversations`
column row by row to prove content equality. Row order, row count, and schema are
unchanged; only row-group boundaries differ. The other 26 files were not touched.

## scripts/

| File | Role |
|---|---|
| `rechunk_nemotron.py` | Rewrites the 3 oversized files and emits `RECHUNK_PROVENANCE.json`. |
| `build_nemotron_cache_v4.sbatch` | Builds the cache from the explicit path list with `--expect-files 29`. |
| `stage3_gate.py` | The 8-check parity gate. Check 1 reads the pinned selection from the frozen upstream file, not the local module, so a local edit cannot move both sides of the comparison. |
| `finalize_stage3_cache.sh` | Post-build: proves completion, writes the provenance sidecar, validates it, and reports the cache constants. |

## Result

Cache `nemotron-terminal-cache-v4`: **366,154 examples, 6,075,088,769 tokens, 19 shards**.
The example count equals the pinned corpus row count exactly, so every source row is
represented 1:1. A full epoch is 2,896.8 steps, so the mandated 1,888-step budget is
0.65 epochs and fits without repetition. Stage 3 gate: 9 pass, 0 fail, 0 skip.

## Two bugs this exists to prevent recurring

**Non-recursive glob.** `prepare-data` expanded `$SNAP/**/*.parquet` with `glob.glob()`
lacking `recursive=True`, so `**` collapsed to `*` and matched only the 3 shards at
depth 1. The 26 nested `synthetic_tasks/` shards — 139,841 of 366,154 rows — were
silently dropped, while a `find`-based gate in the same script counted 29 and passed.
A cache built that way ran for 1h34m before it was caught.

**Basename-keyed provenance.** The sidecar keyed shards by `Path(p).name`. Twenty-six
of these files are all named `data_filtered.parquet`, so 29 shards collapsed to 4 keys
and 25 hashes were discarded after being computed. Now keyed by path relative to the
shards' common root.

Both are fixed in `vista_snowball_chat.py` on the Vista `codex/snowball-chat-vista`
branch, with a per-stage `source_files` count enforced during provenance validation so
a truncated source set fails the gate instead of validating clean.
