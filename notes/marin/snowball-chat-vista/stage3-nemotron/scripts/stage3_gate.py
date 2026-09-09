"""Consolidated Stage 3 (Nemotron Terminal) parity gate.

Eight criteria, no speculative extras. Checks 1-4 need only the source and
config; 5-8 need the completed cache. Run with --require-cache once the cache
build finishes; without it, the cache-dependent checks report SKIP so the
source-side half can be validated while tokenization is still running.
"""
import argparse, hashlib, json, sys
from pathlib import Path

S = "/scratch/11694/mkumar73"
EXPECT_TEMPLATE_SHA = "414d8559d363292c"          # MARIN_CHAT_TEMPLATE (12-char prefix)
EXPECT_TOKENIZER_SHA = "881c9c36c359e1617afef6f7583403567931b7b4f43f6552d2b2155a131650a2"
EXPECT_REV = "a1667c4ffdadea02a89bffe4f1bb7ca2ff19f8d9"
FIXED_STEPS = 1888
SEQ, BATCH = 32768, 64


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--init-checkpoint", required=True)
    ap.add_argument("--require-cache", action="store_true")
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()

    sys.path.insert(0, f"{S}/marin")
    # ANTI-TAUTOLOGY: the pinned selection is read from the file fetched over the
    # network from marin-community/marin PR #8172 head (9383df73c), NOT from the
    # local module. Importing the local module would let a local edit move both
    # sides of check 1 together and still report identical=True.
    UPSTREAM_REF = json.loads(Path(f"{S}/tmp/upstream_pinned_files.json").read_text())
    UPSTREAM_SHA = "9383df73c8e74450227f67ededfb665db403192c"
    from experiments.june_tpu_67b_a2b.moe.vista_snowball_chat import (
        STAGES, snowball_chat_format, expected_chat_steps, read_chat_cache_tokens,
        validate_native_checkpoint_layout, validate_cache_provenance,
    )
    from levanter.tokenizers import load_tokenizer

    res = []
    def check(n, ok, d=""): res.append((("PASS" if ok else "FAIL"), n, d))
    def skip(n, d=""): res.append(("SKIP", n, d))

    spec = STAGES["nemotron_terminal"]

    # 1. exact 29 relative filenames vs the UPSTREAM pinned groups
    ref_files = sorted(UPSTREAM_REF)
    snap = Path(a.snapshot)
    on_disk = sorted(str(p.relative_to(snap)) for p in snap.rglob("*.parquet"))
    check("1-exact-29-pinned-filenames", ref_files == on_disk and len(ref_files) == 29,
          f"{len(on_disk)} on disk vs {len(ref_files)} pinned; identical={ref_files == on_disk}")

    # 2. on-disk hashes chain back to the pinned originals.
    #    NON-DESTRUCTIVE: the recorded originals are READ, never rewritten. The
    #    previous version recomputed hashes and overwrote stage3_source_hashes.json,
    #    which would have erased the pre-re-chunk record -- and it was tautological,
    #    asserting only len(hashes)==29 against hashes it had just written itself.
    orig = json.loads(Path(f"{S}/tmp/stage3_source_hashes.json").read_text())["files"]
    rech = json.loads((snap / "RECHUNK_PROVENANCE.json").read_text())["files"]
    hashes = {}
    for rel in on_disk:
        h = hashlib.sha256()
        with (snap / rel).open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        hashes[rel] = h.hexdigest()
    Path(f"{S}/tmp/stage3_ondisk_hashes.json").write_text(json.dumps(
        {"dataset_id": "nvidia/Nemotron-Terminal-Corpus", "revision": EXPECT_REV,
         "upstream_ref": UPSTREAM_SHA,
         "note": "on-disk, post re-chunk; pinned originals stay in stage3_source_hashes.json",
         "files": hashes}, indent=1, sort_keys=True))
    changed = sorted(r for r in on_disk if hashes[r] != orig.get(r))
    untouched_ok = sum(1 for r in on_disk if r not in rech and hashes[r] == orig.get(r))
    chain_ok = all(
        hashes.get(r) == rech[r]["rechunked_sha256"]
        and rech[r]["original_sha256"] == orig.get(r)
        and rech[r]["content_identical"] is True
        for r in rech
    )
    check("2-hashes-chain-to-pinned-originals",
          len(hashes) == 29 and untouched_ok == 26 and sorted(rech) == changed and chain_ok,
          f"{untouched_ok}/26 untouched byte-identical to pinned originals; "
          f"{len(changed)} changed == {len(rech)} documented re-chunks; chain_ok={chain_ok}")

    # 3. template hash
    fmt = snowball_chat_format(messages_field=spec.messages_field, chat_template=spec.chat_template)
    tsha = hashlib.sha256(fmt.chat_template.encode()).hexdigest()
    check("3-template-is-marin", tsha.startswith(EXPECT_TEMPLATE_SHA), f"{tsha[:16]}…")

    # 4. tokenizer fingerprint
    fsha = hashlib.sha256((Path(a.tokenizer) / "tokenizer.json").read_bytes()).hexdigest()
    check("4-tokenizer-fingerprint", fsha == EXPECT_TOKENIZER_SHA, f"{fsha[:16]}…")

    cache = Path(a.cache)
    have_cache = (cache / "train" / ".stats.json").is_file()
    if not have_cache:
        for n in ("5-assistant-loss-positive", "6-no-empty-or-all-masked",
                  "7-cache-supplies-1888-steps"):
            skip(n, "cache not built yet")
    else:
        stats = json.loads((cache / "train" / ".stats.json").read_text())
        tok_total = stats.get("total_tokens"); ex_total = stats.get("total_elements")
        tok = load_tokenizer(a.tokenizer)
        proc = fmt.build_preprocessor(tok)
        import pyarrow.parquet as pq
        rows = pq.ParquetFile(snap / on_disk[0]).read_row_group(0).slice(0, a.n).to_pylist()
        batch = [{spec.messages_field: r[spec.messages_field]} for r in rows]
        enc = proc(batch)
        pos = sum(1 for e in enc if sum(e["assistant_masks"]) > 0)
        empty = sum(1 for e in enc if len(e["input_ids"]) == 0)
        allmask = sum(1 for e in enc if len(e["input_ids"]) and sum(e["assistant_masks"]) == 0)
        check("5-assistant-loss-positive", pos == len(enc), f"{pos}/{len(enc)} decoded samples have >0 loss tokens")
        check("6-no-empty-or-all-masked", empty == 0 and allmask == 0,
              f"empty={empty} all-masked={allmask} of {len(enc)}")
        need = FIXED_STEPS * SEQ * BATCH
        epochs = need / tok_total if tok_total else 0
        check("7-cache-supplies-1888-steps", tok_total > 0,
              f"cache {tok_total:,} tokens; 1888 steps needs {need:,} "
              f"=> {epochs:.2f} epochs over the cache ({ex_total:,} examples)")
        try:
            validate_cache_provenance(str(cache), "nemotron_terminal", a.tokenizer)
            check("7b-cache-provenance", True, "sidecar matches stage pin")
        except Exception as e:
            check("7b-cache-provenance", False, f"{type(e).__name__}: {str(e)[:70]}")

    # 8. init checkpoint resolves to EXACT Thinking step 630
    try:
        resolved, files, nbytes = validate_native_checkpoint_layout(
            a.init_checkpoint, expect_step=spec.init_step)
        check("8-init-is-thinking-step-630", True,
              f"step {spec.init_step}, {files} files, {nbytes/1e9:.1f} GB; weights-only init resets optimizer/step")
    except Exception as e:
        check("8-init-is-thinking-step-630", False, f"{type(e).__name__}: {str(e)[:80]}")

    for st, n, d in res:
        print(f"{st}  {n:34s} {d}")
    fails = [r for r in res if r[0] == "FAIL"]
    skips = [r for r in res if r[0] == "SKIP"]
    ok = not fails and (not skips or not a.require_cache)
    print(f"\nSTAGE3_GATE {'PASS' if ok else 'FAIL'}  "
          f"({sum(1 for r in res if r[0]=='PASS')} pass, {len(fails)} fail, {len(skips)} skip)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
