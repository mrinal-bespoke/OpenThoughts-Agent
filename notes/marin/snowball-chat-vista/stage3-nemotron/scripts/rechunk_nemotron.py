"""Re-chunk the three oversized Nemotron parquet files.

Only the PHYSICAL row-group layout changes. The originals stay untouched and
remain the hashed source of truth; these copies exist solely because
pq.read_row_group() cannot materialize a >2GB row group holding a nested column
(ArrowNotImplementedError / ArrowMemoryError), which is how zephyr's reader loads
parquet.

Equality is proven, not asserted: schema, row count, row order and the full
`conversations` content are streamed and compared after writing.
"""
import hashlib, json, sys
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

TARGETS = ["dataset_adapters/code.parquet",
           "dataset_adapters/math.parquet",
           "dataset_adapters/swe.parquet"]
ROWS_PER_GROUP = 2000     # keeps each row group far below the 2GB nested limit
BATCH = 250


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    snap, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    record = {"transformation": "row-group re-chunk only; content unchanged",
              "rows_per_group": ROWS_PER_GROUP, "files": {}}
    for rel in TARGETS:
        src, dst = snap / rel, out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        pf = pq.ParquetFile(src)
        schema = pf.schema_arrow
        print(f"[{rel}] {pf.metadata.num_rows:,} rows, "
              f"{pf.metadata.num_row_groups} rg -> rechunking", flush=True)

        written = 0
        with pq.ParquetWriter(dst, schema, compression="zstd") as w:
            buf = []
            for b in pf.iter_batches(batch_size=BATCH):
                buf.append(b)
                if sum(x.num_rows for x in buf) >= ROWS_PER_GROUP:
                    t = pa.Table.from_batches(buf, schema=schema)
                    w.write_table(t, row_group_size=ROWS_PER_GROUP)
                    written += t.num_rows; buf = []
            if buf:
                t = pa.Table.from_batches(buf, schema=schema)
                w.write_table(t, row_group_size=ROWS_PER_GROUP)
                written += t.num_rows

        # ---- verification: schema, count, ORDER and content, streamed in lockstep
        npf = pq.ParquetFile(dst)
        assert npf.schema_arrow.equals(schema), f"{rel}: schema differs"
        assert npf.metadata.num_rows == pf.metadata.num_rows, (
            f"{rel}: {npf.metadata.num_rows} != {pf.metadata.num_rows}")
        a = pq.ParquetFile(src).iter_batches(batch_size=BATCH, columns=["conversations"])
        b_ = pq.ParquetFile(dst).iter_batches(batch_size=BATCH, columns=["conversations"])
        ha, hb, na = hashlib.sha256(), hashlib.sha256(), 0
        for xa in a:
            for row in xa.column(0).to_pylist():
                ha.update(repr(row).encode()); na += 1
        nb = 0
        for xb in b_:
            for row in xb.column(0).to_pylist():
                hb.update(repr(row).encode()); nb += 1
        assert na == nb == pf.metadata.num_rows, f"{rel}: streamed {na} vs {nb}"
        assert ha.hexdigest() == hb.hexdigest(), f"{rel}: conversations content differs"
        # the re-chunked file must actually be readable the way zephyr reads it
        npf.read_row_group(0, columns=["conversations"])

        record["files"][rel] = {
            "original_sha256": sha_file(src),
            "rechunked_sha256": sha_file(dst),
            "rows": pf.metadata.num_rows,
            "row_groups_before": pf.metadata.num_row_groups,
            "row_groups_after": npf.metadata.num_row_groups,
            "conversations_content_sha256": ha.hexdigest(),
            "content_identical": True,
        }
        print(f"  OK rows={written:,} rg {pf.metadata.num_row_groups}->{npf.metadata.num_row_groups} "
              f"content_sha={ha.hexdigest()[:16]} read_row_group=OK", flush=True)

    (out_dir / "RECHUNK_PROVENANCE.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    print("RECHUNK_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
