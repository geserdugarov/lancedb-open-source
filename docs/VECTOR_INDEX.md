# Vector Index Architecture in LanceDB

This document provides a thorough description of LanceDB's vector indexing
system: how vectors are indexed, searched, and the trade-offs between index types.

---

## Overview

LanceDB delegates vector index construction to the `lance-index` crate and
wraps it with a high-level builder API. All vector indices use a two-level
architecture:

1. **Partitioning** (IVF) — divides vectors into clusters for coarse search
2. **Encoding** — compresses vectors within each partition for fast fine search

Optionally, a **graph layer** (HNSW) replaces brute-force scanning within
partitions.

```
  ┌─────────────────────────────────────────────────────────┐
  │                   Vector Index                          │
  │                                                         │
  │  ┌─────────────────────────────────────────────────┐    │
  │  │            IVF (Partitioning Layer)              │    │
  │  │                                                  │    │
  │  │  Cluster vectors into partitions via K-means     │    │
  │  │  Each partition has a centroid                    │    │
  │  └──────────┬───────────────────────────────────────┘    │
  │             │                                            │
  │       ┌─────┴─────┬──────────┬──────────┐                │
  │       ▼           ▼          ▼          ▼                │
  │  ┌─────────┐ ┌─────────┐ ┌───────┐ ┌───────┐            │
  │  │Partition│ │Partition│ │  ...  │ │Part. N│            │
  │  │    0    │ │    1    │ │       │ │       │            │
  │  └────┬────┘ └────┬────┘ └───┬───┘ └───┬───┘            │
  │       │           │          │         │                 │
  │  ┌────▼───────────▼──────────▼─────────▼──────────┐     │
  │  │         Encoding / Search Layer                 │     │
  │  │                                                 │     │
  │  │  Option A: Flat (raw vectors, exact distances)  │     │
  │  │  Option B: PQ  (product quantization)           │     │
  │  │  Option C: SQ  (scalar quantization)            │     │
  │  │  Option D: RQ  (RabitQ quantization)            │     │
  │  │  Option E: HNSW+PQ (graph + product quant.)     │     │
  │  │  Option F: HNSW+SQ (graph + scalar quant.)      │     │
  │  └─────────────────────────────────────────────────┘     │
  └─────────────────────────────────────────────────────────┘
```

---

## Supported Index Types

| Index Type   | Partitioning | Within-Partition Search | Compression |
|-------------|-------------|------------------------|-------------|
| `IvfFlat`    | IVF          | Brute force             | None        |
| `IvfPq`      | IVF          | Brute force             | Product Q.  |
| `IvfSq`      | IVF          | Brute force             | Scalar Q.   |
| `IvfRq`      | IVF          | Brute force             | RabitQ      |
| `IvfHnswPq`  | IVF          | HNSW graph              | Product Q.  |
| `IvfHnswSq`  | IVF          | HNSW graph              | Scalar Q.   |

**Default:** When `Index::Auto` is specified, LanceDB uses **IVF_PQ** with L2
distance, `num_sub_vectors = dim/16`, and `num_bits = 8`.

---

## IVF — Inverted File Partitioning

IVF partitions the vector space using **K-means clustering**. At query time,
only the closest partitions are searched, dramatically reducing the search space.

### How It Works

```
  Training Phase:
  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │  1. Sample vectors from dataset                  │
  │     (sample_rate × num_partitions vectors)       │
  │                                                  │
  │  2. Run K-means clustering                       │
  │     ┌───────────────────────────────────────┐    │
  │     │  Iterate up to max_iterations times:  │    │
  │     │    - Assign vectors to nearest centroid│    │
  │     │    - Recompute centroids              │    │
  │     │    - Check for convergence            │    │
  │     └───────────────────────────────────────┘    │
  │                                                  │
  │  3. Store centroids + partition assignments      │
  └──────────────────────────────────────────────────┘

  Search Phase:
  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │  1. Compute distance from query to ALL centroids │
  │                                                  │
  │  2. Select top nprobes closest partitions        │
  │                                                  │
  │  3. Search within selected partitions only       │
  │     (brute force or HNSW, depending on type)     │
  │                                                  │
  │  4. Merge results across partitions              │
  └──────────────────────────────────────────────────┘
```

### Parameters

| Parameter              | Default         | Description                                  |
|-----------------------|-----------------|----------------------------------------------|
| `num_partitions`       | `√num_rows`     | Number of K-means clusters                   |
| `target_partition_size`| —               | Alternative: auto-calculate num_partitions   |
| `sample_rate`          | 256             | Training sample = sample_rate × partitions   |
| `max_iterations`       | 50              | K-means convergence limit                    |

### Trade-offs

- **More partitions** → faster search (smaller partitions) but slower training
  and risk of empty/tiny partitions
- **Fewer partitions** → slower search but more robust clustering
- **Rule of thumb:** `num_partitions ≈ √N` where N = number of rows

### Search Parameter: nprobes

At query time, `nprobes` controls how many partitions are searched:

```
  nprobes = 1     Search only the closest partition      (fastest, lowest recall)
  nprobes = 20    Search 20 closest partitions           (default)
  nprobes = all   Search every partition                 (slowest, 100% recall)
```

LanceDB supports `minimum_nprobes` and `maximum_nprobes` to allow dynamic
expansion when the initial search is insufficient.

---

## PQ — Product Quantization

PQ compresses vectors by dividing them into subvectors and independently
quantizing each subvector into a small codebook.

### How It Works

```
  Original vector (128 dimensions, float32):
  ┌────────────────────────────────────────────────────────────────┐
  │ v0  v1  v2 ... v15 │ v16 v17 ... v31 │ ... │ v112 ... v127  │
  └────────┬────────────┴────────┬─────────┴─────┴───────┬────────┘
           │                     │                       │
     subvector 0            subvector 1          ... subvector 7
     (16 dims)              (16 dims)                (16 dims)

  For each subvector position, train a codebook of 2^num_bits entries:
  ┌────────────────┐
  │ Codebook 0     │     256 centroids for subvector position 0
  │  code 0: [...]│
  │  code 1: [...]│
  │  ...          │
  │  code 255:[..]│
  └────────────────┘

  Compressed vector (8 bytes instead of 512 bytes):
  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
  │ c0   │ c1   │ c2   │ c3   │ c4   │ c5   │ c6   │ c7   │
  │(8bit)│(8bit)│(8bit)│(8bit)│(8bit)│(8bit)│(8bit)│(8bit)│
  └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
                    64× compression!
```

### Parameters

| Parameter          | Default           | Description                              |
|-------------------|-------------------|------------------------------------------|
| `num_sub_vectors`  | `dim / 16`        | Number of subvector segments             |
| `num_bits`         | 8                 | Bits per code (256 codebook entries)     |

### Auto-Selection of num_sub_vectors

The dimension must be evenly divisible by `num_sub_vectors`. LanceDB uses this
heuristic (from `rust/lancedb/src/index/vector.rs`):

```
  if dim % 16 == 0  →  num_sub_vectors = dim / 16     (preferred for SIMD)
  if dim % 8  == 0  →  num_sub_vectors = dim / 8
  else              →  num_sub_vectors = 1             (poor performance)
```

### Compression Ratio

For a 128-dimensional float32 vector:
- Original: 128 × 4 bytes = **512 bytes**
- PQ (8 sub-vectors, 8-bit codes): 8 × 1 byte = **8 bytes** → **64× compression**

### Distance Computation

PQ enables fast **asymmetric distance computation (ADC)**:
1. Precompute distance from query subvectors to all codebook entries
2. For each compressed vector, look up precomputed distances and sum
3. No decompression needed — pure table lookups

### Refinement

Since PQ distances are approximate, a **refine step** can improve accuracy:

```
  refine_factor = 5, limit = 10:

  1. Retrieve top 50 candidates using PQ distances  (limit × refine_factor)
  2. Fetch original uncompressed vectors for these 50
  3. Compute exact distances
  4. Return top 10 by exact distance
```

---

## SQ — Scalar Quantization

SQ is a simpler quantization scheme that maps each float32 component to an
8-bit integer.

### How It Works

```
  For each dimension, compute min and max across training data:

  float32 value:    min=─1.5  ──────────────────── max=2.3
                      │                              │
  uint8 value:        0 ──────────────────────────── 255

  Quantized value = round(255 × (value - min) / (max - min))
```

### Compression Ratio

- float32 → uint8: **4× compression**
- Less compression than PQ, but better accuracy per bit
- No codebook training needed — faster index construction

### When to Use SQ vs PQ

| Aspect      | PQ                          | SQ                      |
|------------|-----------------------------|-------------------------|
| Compression | High (16–64×)               | Moderate (4×)           |
| Accuracy    | Lower                       | Higher                  |
| Training    | Slower (codebook learning)  | Faster (min/max only)   |
| Best for    | Very large datasets         | Medium datasets         |

---

## RQ — RabitQ Quantization

RabitQ is an alternative quantization method available as `IvfRq`. It provides
a different accuracy/compression trade-off compared to PQ and SQ.

| Parameter  | Default | Description                        |
|-----------|---------|-------------------------------------|
| `num_bits` | —       | Bits for RabitQ quantization        |

---

## HNSW — Hierarchical Navigable Small World

HNSW builds a navigable graph within each IVF partition for fast approximate
nearest neighbor search, replacing brute-force scanning.

### How It Works

```
  Layer 2 (sparse):     A ─────────── D
                        │             │
                        │             │
  Layer 1 (medium):     A ── B ────── D ── E
                        │    │        │    │
                        │    │        │    │
  Layer 0 (dense):      A ── B ── C ── D ── E ── F ── G ── H
                        │    │    │    │    │    │    │    │
                        └────┴────┴────┴────┴────┴────┴────┘
                              All vectors present here

  Search: Enter at top layer, greedily descend to lower layers,
  widen search at bottom layer.
```

**Construction:**
1. Insert vectors one at a time
2. Each vector randomly assigned a maximum layer
3. At each layer, connect to `m` nearest neighbors
4. Use `ef_construction` candidates to find best neighbors

**Search:**
1. Start at top layer entry point
2. Greedily traverse to nearest neighbor at each layer
3. Descend to next layer
4. At bottom layer, expand search with `ef` candidates
5. Return top-k results

### Parameters

| Parameter          | Default | Description                                |
|-------------------|---------|---------------------------------------------|
| `m` (num_edges)    | 20      | Max neighbors per node per layer            |
| `ef_construction`  | 300     | Candidates during graph construction        |
| `ef` (search time) | 1.5×k   | Candidates during search (k = result limit) |

### Trade-offs

```
  m (edges per node):
    Low (5-10)    → Faster build, less memory, lower recall
    Default (20)  → Good balance
    High (40-100) → Slower build, more memory, higher recall

  ef_construction:
    Low (100)     → Faster build, lower index quality
    Default (300) → Good balance
    High (500+)   → Slower build, better index quality

  ef (search time):
    Low (< k)     → Very fast, poor recall
    Default (1.5k)→ Good balance
    High (100+)   → Slower search, near-perfect recall
```

### HNSW + IVF

In LanceDB, HNSW is always combined with IVF (`IvfHnswPq` / `IvfHnswSq`):

```
  ┌──────────────────────────────────────────────┐
  │ IVF: Select top nprobes partitions           │
  └──────────┬────────────┬──────────────────────┘
             │            │
      ┌──────▼──────┐ ┌───▼────────┐
      │ Partition 0 │ │ Partition 1│   ...
      │             │ │            │
      │ ┌─────────┐ │ │ ┌────────┐ │
      │ │  HNSW   │ │ │ │  HNSW  │ │   Each partition has
      │ │  Graph  │ │ │ │  Graph │ │   its own HNSW graph
      │ └────┬────┘ │ │ └───┬────┘ │
      │      │      │ │     │      │
      │ ┌────▼────┐ │ │ ┌───▼────┐ │
      │ │ PQ / SQ │ │ │ │ PQ / SQ│ │   Vectors stored
      │ │ encoded │ │ │ │ encoded│ │   in compressed form
      │ └─────────┘ │ │ └────────┘ │
      └─────────────┘ └────────────┘
```

This gives two levels of approximation:
1. **IVF** narrows the search to a few partitions
2. **HNSW** efficiently finds nearest neighbors within each partition

---

## Distance Metrics

All vector indices support these distance metrics:

```
  ┌────────────────────────────────────────────────────────────┐
  │                                                            │
  │  L2 (Euclidean):  d(a,b) = Σ(aᵢ - bᵢ)²                   │
  │    Range: [0, ∞)                                           │
  │    Use: General purpose, when magnitude matters            │
  │                                                            │
  │  Cosine:  d(a,b) = 1 - (a·b)/(‖a‖·‖b‖)                   │
  │    Range: [0, 2]                                           │
  │    Use: Text embeddings, when direction matters            │
  │                                                            │
  │  Dot:  d(a,b) = -a·b                                      │
  │    Range: (-∞, ∞)                                          │
  │    Use: When vectors are pre-normalized                    │
  │                                                            │
  │  Hamming:  d(a,b) = number of differing bits               │
  │    Use: Binary vectors                                     │
  │                                                            │
  └────────────────────────────────────────────────────────────┘
```

**Critical:** The distance metric used during index training **must match** the
metric used during search. Mismatched metrics produce invalid results.

---

## End-to-End Vector Search Flow

```
  User: table.vector_search([0.1, 0.2, ...]).limit(10).nprobes(20).execute()
    │
    ▼
  ┌───────────────────────────────────────────────────────┐
  │ 1. Build VectorQueryRequest                          │
  │    query_vector = [0.1, 0.2, ...]                    │
  │    column = auto-detect vector column                 │
  │    top_k = 10                                         │
  │    nprobes = 20                                       │
  └──────────────────────┬────────────────────────────────┘
                         │
                         ▼
  ┌───────────────────────────────────────────────────────┐
  │ 2. Configure Lance Scanner                           │
  │    scanner.nearest(column, vector, 10)               │
  │    scanner.nprobes(20)                                │
  │    scanner.use_index(true)                            │
  │    scanner.prefilter(filter) if any                   │
  └──────────────────────┬────────────────────────────────┘
                         │
                         ▼
  ┌───────────────────────────────────────────────────────┐
  │ 3. Lance Scanner executes with index                 │
  │                                                      │
  │    a. IVF: Compute query-to-centroid distances       │
  │       Select top 20 partitions                       │
  │                                                      │
  │    b. Within each partition:                         │
  │       - IvfFlat: Brute-force exact distance          │
  │       - IvfPq/Sq: Compressed distance (ADC)          │
  │       - IvfHnsw*: Graph traversal + compressed dist  │
  │                                                      │
  │    c. Merge results from all partitions              │
  │    d. Apply refine_factor if set                     │
  │    e. Return top 10 with distances                   │
  └──────────────────────┬────────────────────────────────┘
                         │
                         ▼
  ┌───────────────────────────────────────────────────────┐
  │ 4. DataFusion execution plan                         │
  │    Apply post-filters, projections, limit            │
  │    Stream RecordBatch results to caller              │
  └──────────────────────────────────────────────────────┘
```

---

## Search Parameters at Query Time

| Parameter        | Default   | Description                                       |
|-----------------|-----------|---------------------------------------------------|
| `nprobes`        | 20        | Number of IVF partitions to search                |
| `ef`             | 1.5 × k   | HNSW search candidates (only for HNSW indices)    |
| `refine_factor`  | None      | Multiplier for refinement with exact distances    |
| `distance_type`  | L2        | Must match training metric                        |
| `use_index`      | true      | Set false for brute-force (no index)              |
| `lower_bound`    | None      | Minimum distance threshold                        |
| `upper_bound`    | None      | Maximum distance threshold                        |

---

## Choosing an Index Type

```
  Dataset size?
  │
  ├── < 10K vectors
  │     └── No index needed (brute force is fast enough)
  │
  ├── 10K – 1M vectors
  │     │
  │     ├── Memory constrained?
  │     │     ├── Yes → IvfPq (high compression)
  │     │     └── No  → IvfSq (better accuracy) or IvfFlat (exact)
  │     │
  │     └── Need highest recall?
  │           └── IvfHnswSq (graph + moderate compression)
  │
  └── > 1M vectors
        │
        ├── Prioritize speed → IvfPq (smallest footprint)
        ├── Prioritize recall → IvfHnswPq or IvfHnswSq
        └── Balanced → IvfPq with refine_factor
```

---

## Index Creation Examples

### Auto Index (IVF_PQ defaults)

```rust
table.create_index(&["vector"], Index::Auto)
    .execute()
    .await?;
```

### Custom IVF_PQ

```rust
table.create_index(
    &["vector"],
    Index::IvfPq(
        IvfPqIndexBuilder::default()
            .distance_type(DistanceType::Cosine)
            .num_partitions(256)
            .num_sub_vectors(16)
            .num_bits(8)
    ),
)
.execute()
.await?;
```

### IVF_HNSW_SQ (High Recall)

```rust
table.create_index(
    &["vector"],
    Index::IvfHnswSq(
        IvfHnswSqIndexBuilder::default()
            .distance_type(DistanceType::Cosine)
            .num_partitions(100)
            .num_edges(30)
            .ef_construction(400)
    ),
)
.execute()
.await?;
```

### Vector Search with Tuning

```rust
let results = table.vector_search(&query_vector)?
    .distance_type(DistanceType::Cosine)
    .limit(20)
    .nprobes(40)
    .refine_factor(5)
    .execute()
    .await?;
```

---

## Integration with lance-index Crate

LanceDB's index builders translate to `lance-index` parameter structs:

```
  LanceDB                          lance-index / lance
  ──────────────────────────────    ─────────────────────────────
  IvfPqIndexBuilder            →   IvfBuildParams + PQBuildParams
  IvfSqIndexBuilder            →   IvfBuildParams + SQBuildParams
  IvfRqIndexBuilder            →   IvfBuildParams + RQBuildParams
  IvfHnswPqIndexBuilder        →   IvfBuildParams + HnswBuildParams + PQBuildParams
  IvfHnswSqIndexBuilder        →   IvfBuildParams + HnswBuildParams + SQBuildParams
                                      │
                                      ▼
                               VectorIndexParams
                                      │
                                      ▼
                               dataset.create_index()
```

The conversion happens in `make_index_params()` in `rust/lancedb/src/table.rs`,
which maps the high-level builder options to the low-level lance parameter
structs before delegating to the Lance dataset's index creation machinery.
