# IVF_RQ — RaBitQ-Quantized Vector Index

`IvfRq` combines **IVF partitioning** with **RaBitQ quantization**. Unlike
`IvfPq` (which quantizes groups of dimensions via a learned codebook) and
`IvfSq` (which quantizes each dimension with a uniform min/max), RaBitQ
quantizes **each dimension independently into a small number of bits**
(typically 1). The query vector stays in full precision at search time —
distance is computed asymmetrically against the bit-packed stored vectors.

This index was added as a middle ground between PQ and SQ: similar
compression ratios to PQ, no subvector alignment requirement, and no
codebook training step.

```
  ┌─────────────────────────────────────────────────────────┐
  │                      IVF_RQ Index                       │
  │                                                         │
  │  ┌───────────────────────────────────────────────┐      │
  │  │  IVF — K-means partitioning (same as IvfPq)   │      │
  │  └──────┬────────────┬────────────┬──────────────┘      │
  │         │            │            │                     │
  │   ┌─────▼──┐   ┌─────▼──┐   ┌─────▼──┐                  │
  │   │ Part 0 │   │ Part 1 │   │  ...   │                  │
  │   │        │   │        │   │        │                  │
  │   │ RaBitQ │   │ RaBitQ │   │ RaBitQ │   Bit-packed     │
  │   │ codes  │   │ codes  │   │ codes  │   per-dim codes  │
  │   └────────┘   └────────┘   └────────┘                  │
  └─────────────────────────────────────────────────────────┘
```

---

## When to Use IVF_RQ

Pick `IvfRq` when:

- **Extreme compression is needed** — `num_bits = 1` packs one float32
  dimension into a single bit (32× footprint reduction).
- **Vector dimension is not divisible by 8 or 16** — PQ needs
  `dim % num_sub_vectors == 0` for good SIMD; RQ has no such constraint.
- **Index build time matters** — RQ skips per-subvector codebook
  K-means that PQ runs for each of its `num_sub_vectors` segments.
- **Dataset will use a `refine_factor`** — RQ's compressed scores are
  coarser than PQ's, so raw re-ranking is especially valuable.

Avoid when:

- The dataset already fits comfortably in RAM as float32 (`IvfFlat` gives
  exact distances for free).
- You need **Hamming** distance — `IvfRq` supports L2, Cosine, and Dot only.
- You cannot tolerate a **slow, memory-intensive training phase** — the
  builder explicitly warns about this
  (`rust/lancedb/src/index/vector.rs:336–337`).

---

## How RaBitQ Works

RaBitQ (published 2024) is a binary quantization scheme. For LanceDB
purposes, the important behaviour is:

1. **Preprocessing:** vectors are optionally rotated so their components are
   more isotropically distributed.
2. **Per-dimension encoding:** each dimension is quantized to `num_bits`
   bits. With `num_bits = 1` (default), this reduces to a sign bit per
   dimension plus a small per-vector norm scalar.
3. **Asymmetric distance:** the query remains float32; stored vectors are
   bit-packed. Distance is accumulated one dimension at a time using fast
   bitwise operations, no decompression needed.

Contrast with siblings:

```
  PQ:  [v0..v15] [v16..v31] ...      → 8 codebook IDs × 8 bits = 8 bytes
       Each group quantized to a 2^num_bits codebook entry
       Distance: precomputed query-to-codebook table, then sum

  SQ:  [v0] [v1] [v2] ... [v127]     → 128 uint8 = 128 bytes
       Each dim quantized to uint8 via a fixed per-dim min/max
       Distance: dequantize + compute

  RQ:  [v0] [v1] [v2] ... [v127]     → 128 × num_bits bits
       Each dim independently quantized to num_bits bits
       Distance: bitwise accumulate against float32 query
```

### Compression at a Glance

For a 128-dim float32 vector (512 bytes raw):

| Encoding                 | Stored size | Ratio |
|--------------------------|-------------|-------|
| Raw float32 (IvfFlat)    | 512 B       | 1×    |
| IvfSq (uint8)            | 128 B       | 4×    |
| IvfPq (16 subvec, 8 bit) | 16 B        | 32×   |
| **IvfRq (num_bits=1)**   | **16 B**    | **32×** |
| IvfRq (num_bits=4)       | 64 B        | 8×    |

RQ at 1 bit/dim matches PQ at 16 subvectors × 8 bits/code in footprint —
without any codebook training.

---

## Parameters

Source: `rust/lancedb/src/index/vector.rs:326–374` for the struct and
`rust/lancedb/src/table.rs:1971–1986` for parameter conversion.

| Parameter               | Default        | Description                                       |
|-------------------------|----------------|---------------------------------------------------|
| `distance_type`         | `L2`           | `L2`, `Cosine`, or `Dot`. Must match at search.   |
| `num_partitions`        | `√num_rows`    | Number of IVF K-means clusters.                   |
| `num_bits`              | **`1`**        | Bits per dimension for RaBitQ encoding.           |
| `sample_rate`           | `256`          | Training sample = `sample_rate × num_partitions`. |
| `max_iterations`        | `50`           | K-means convergence limit.                        |
| `target_partition_size` | `~8192`        | Alternative to `num_partitions` (auto-derives).   |

The `num_bits` default of 1 is the most aggressive point of the RQ
spectrum — users who need higher accuracy should bump it to 2–8.

Defaults cross-checked against:
- Rust: `IvfRqIndexBuilder::default` —
  `rust/lancedb/src/index/vector.rs:353–364`.
- `RQBuildParams::new(index.num_bits.unwrap_or(1) as u8)` —
  `rust/lancedb/src/table.rs:1979`.
- Python: `num_bits: int = 1` —
  `python/python/lancedb/index.py:690`.
- TypeScript: "the default is 1 bit per dimension" —
  `nodejs/lancedb/indices.ts:130–136`.

---

## Creating an IVF_RQ Index

### Rust

```rust
use lancedb::index::{Index, vector::IvfRqIndexBuilder};
use lancedb::DistanceType;

# async fn create(table: &lancedb::Table) -> lancedb::Result<()> {
table
    .create_index(
        &["vector"],
        Index::IvfRq(
            IvfRqIndexBuilder::default()
                .distance_type(DistanceType::Cosine)
                .num_partitions(256)
                .num_bits(2),
        ),
    )
    .execute()
    .await?;
# Ok(())
# }
```

### Python

```python
import lancedb
from lancedb.index import IvfRq

table.create_index(
    "vector",
    config=IvfRq(
        distance_type="cosine",
        num_partitions=256,
        num_bits=2,
    ),
)
```

### TypeScript

```typescript
import { Index } from "@lancedb/lancedb";

await table.createIndex("vector", {
  config: Index.ivfRq({
    distanceType: "cosine",
    numPartitions: 256,
    numBits: 2,
  }),
});
```

---

## Searching an IVF_RQ Index

Search uses the same parameters as any IVF-based index (`nprobes`,
`refine_factor`, `lower_bound`, `upper_bound`), but RQ-specific guidance:

- **`refine_factor` is strongly recommended.** RQ's single-bit codes are
  noisier than PQ's 8-bit subvector codes; fetch `N × R` candidates and
  re-rank on raw vectors for meaningfully better recall.
- **`nprobes` behaves identically to IvfPq** — each partition is scanned
  with compressed distances, then merged.
- **`distance_type` at search must match training.** Mismatched metrics
  silently produce invalid results.

```rust
# async fn search(table: &lancedb::Table, q: Vec<f32>) -> lancedb::Result<()> {
use lancedb::query::{QueryBase, ExecutableQuery};
use lancedb::DistanceType;

let stream = table
    .vector_search(q)?
    .distance_type(DistanceType::Cosine)
    .limit(10)
    .minimum_nprobes(40)?
    .refine_factor(5)
    .execute()
    .await?;
# Ok(())
# }
```

See [`search_path.md`](./search_path.md) for the exact sequence of steps
that executes when this query runs.

---

## IVF_RQ vs IVF_PQ vs IVF_SQ

| Aspect                   | IvfPq                       | IvfSq                   | IvfRq                         |
|--------------------------|-----------------------------|-------------------------|-------------------------------|
| Granularity              | Subvectors of `dim/16` dims | Per-dimension           | Per-dimension                 |
| Codebook training        | Yes (K-means per subvec)    | No (min/max only)       | No (rotation optional)        |
| Typical compression      | 16–64×                      | 4×                      | 4×–32× (via `num_bits`)       |
| Dim divisibility need    | Yes (`dim % subvecs == 0`)  | No                      | No                            |
| Distance compute         | Codebook LUT + sum (ADC)    | Dequantize + compute    | Bitwise accumulate            |
| Supports Hamming         | No                          | No                      | No                            |
| Refine benefit           | Moderate                    | Low                     | High                          |
| Build time               | Slow (codebook K-means)     | Fast                    | Medium (rotation + IVF only)  |
| Training memory          | High                        | Low                     | High (warned in builder doc)  |

Rule of thumb:

- **Tight memory budget, Gigabyte-scale dataset** → `IvfPq` (well-trodden).
- **Accuracy matters, smaller dataset** → `IvfSq` (simple, forgiving).
- **Odd dimension, want fast builds or aggressive compression without
  codebooks** → `IvfRq`.

---

## Mapping to `lance-index`

`IvfRqIndexBuilder` → `RQBuildParams::new(num_bits)` + `IvfBuildParams` →
`VectorIndexParams::with_ivf_rq_params(distance_type, ivf, rq)`.
All conversion happens in `make_index_params` at
`rust/lancedb/src/table.rs:1971–1986`. The underlying RaBitQ algorithm
lives in the `lance-index` crate under `lance_index::vector::bq`.

---

## Cross-Language Bindings

| Language    | Type                   | Location                                      |
|-------------|------------------------|-----------------------------------------------|
| Rust        | `IvfRqIndexBuilder`    | `rust/lancedb/src/index/vector.rs:326–374`    |
| Python      | `IvfRq` dataclass      | `python/python/lancedb/index.py:643–693`      |
| TypeScript  | `IvfRqOptions`         | `nodejs/lancedb/indices.ts:115–184`           |

All three expose the same parameter set. The Python and TypeScript
bindings forward to the Rust builder via PyO3 and napi-rs respectively.
