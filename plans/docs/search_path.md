# Vector Search Path — How LanceDB Finds the Closest N Vectors

This document traces what actually happens — in memory, on disk, and across
crate boundaries — when a user runs:

```rust
table
    .vector_search(q)?
    .limit(N)
    .minimum_nprobes(20)?
    .refine_factor(5)
    .execute()
    .await?;
```

Scope: the local (`NativeTable`) path is described in detail; the remote
(`RemoteTable`) path is contrasted briefly at the end. Everything below
refers to code at commit `2a98474b` on branch `main-described`.

---

## The Short Version

```
                  ┌───────────────────────────────────────────┐
                  │  lancedb crate                            │
                  │                                           │
  vector_search() ─▶ Query builder ─▶ VectorQueryRequest ─┐   │
                                                          │   │
                  ┌───────────────────────────────────────┼───┘
                  │                                       │
                  │  lance crate (Scanner + DataFusion)   │
                  │                                       │
                  │   Scanner.nearest / nprobes / refine ─┘
                  │     │
                  │     ▼
                  │   ExecutionPlan ───▶ execute_plan ──▶ RecordBatch stream
                  │     │
                  │  ┌──┴──────────────────────────────────────┐
                  │  │  lance-index crate                      │
                  │  │                                         │
                  │  │  IVF centroids ─▶ pick top nprobes      │
                  │  │        │                                │
                  │  │  Partition scan (Flat/PQ/SQ/RQ/HNSW)    │
                  │  │        │                                │
                  │  │  Cross-partition merge (top-k heap)     │
                  │  │        │                                │
                  │  │  Refine (raw-vector re-rank)            │
                  │  └─────────────────────────────────────────┘
                  └───────────────────────────────────────────┘
```

Five stages follow: builder → plan → index read → refine → stream.

---

## Stage 1 — Builder Construction

Entry point: `Table::vector_search` simply delegates to
`Query::nearest_to`:

```rust
// rust/lancedb/src/table.rs:972–974
pub fn vector_search(&self, query: impl IntoQueryVector) -> Result<VectorQuery> {
    self.query().nearest_to(query)
}
```

`Query::nearest_to` (at `rust/lancedb/src/query.rs:840–850`) promotes a
plain `Query` into a `VectorQuery`, stores the query vector, and sets a
default limit of `DEFAULT_TOP_K = 10` (`query.rs:34`) if the caller did
not.

The state object is `VectorQueryRequest` at `query.rs:896–922`:

| Field                    | Purpose                                             |
|--------------------------|-----------------------------------------------------|
| `base: QueryRequest`     | filter, projection, `limit`, `offset`, prefilter    |
| `column: Option<String>` | vector column; auto-detected if `None`              |
| `query_vector: Vec<Arc<dyn Array>>` | one or more query vectors (multivector OK) |
| `minimum_nprobes: usize` | starts at 20 (`query.rs:930`)                       |
| `maximum_nprobes: Option<usize>`    | dynamic expansion ceiling                |
| `refine_factor: Option<u32>`        | raw re-rank multiplier                   |
| `ef: Option<usize>`                 | HNSW search breadth                      |
| `distance_type: Option<DistanceType>` | overrides the index default            |
| `lower_bound`, `upper_bound`        | distance range gating                    |
| `use_index: bool`                   | set false to force brute force           |

Builder methods like `.limit()`, `.minimum_nprobes()`, `.refine_factor()`
only mutate this struct. **No I/O, no compute, no index read** happens
in Stage 1.

---

## Stage 2 — Plan Creation

`VectorQuery::execute_with_options` (at
`rust/lancedb/src/query.rs:1298–1316`) picks a branch:

- If the request has a `full_text_search` set → `execute_hybrid`.
- Otherwise → `inner_execute_with_options`, which calls
  `self.parent.create_plan(&AnyQuery::VectorQuery(self.request), ...)`.

For a local table, `NativeTable::create_plan` dispatches to
`rust/lancedb/src/table/query.rs::create_plan` (`table/query.rs:81–246`).
This function translates `VectorQueryRequest` into a Lance `Scanner`:

```rust
// rust/lancedb/src/table/query.rs:145–245 (elided)
let mut scanner: Scanner = ds_ref.scan();

let top_k = query.base.limit.unwrap_or(DEFAULT_TOP_K)
          + query.base.offset.unwrap_or(0);
scanner.nearest(&column, query_vector.as_ref(), top_k)?;

scanner.minimum_nprobes(query.minimum_nprobes);
if let Some(maximum_nprobes) = query.maximum_nprobes {
    scanner.maximum_nprobes(maximum_nprobes);
}
if let Some(ef) = query.ef { scanner.ef(ef); }

scanner.distance_range(query.lower_bound, query.upper_bound);
scanner.use_index(query.use_index);
scanner.prefilter(query.base.prefilter);

// projection, with_row_id, batch_size, fast_search ...

if let Some(filter) = &query.base.filter { /* sql / substrait / df expr */ }
if let Some(fts) = &query.base.full_text_search { scanner.full_text_search(fts.clone())?; }
if let Some(refine_factor) = query.refine_factor { scanner.refine(refine_factor); }
if let Some(dt) = query.distance_type { scanner.distance_metric(dt.into()); }

Ok(scanner.create_plan().await?)
```

`scanner.create_plan()` is where control leaves the `lancedb` crate and
enters `lance`. The returned `Arc<dyn ExecutionPlan>` is a DataFusion
plan — it has not executed yet; only the schema, partitions, and child
plans are resolved.

Multi-vector and multivector requests take a slightly different path
(`table/query.rs:96–143`) that creates one plan per query vector and
unions them via `create_multi_vector_plan`.

---

## Stage 3 — How the Index Is Read

This is the core of the walkthrough. The machinery lives in the
`lance` and `lance-index` crates. The description below is assembled from
the LanceDB-side Scanner configuration, the index types' builder
parameters, and what each encoding algorithm must do to answer a top-N
query.

### 3a. Index selection

When `scanner.create_plan()` runs, Lance looks up the dataset manifest
and finds any vector indices on the chosen column. It picks one and
inspects its type:

- No index present, or `use_index = false` → **fallback to exact KNN**:
  a sequential scan of the vector column, computing the distance to every
  row and keeping a top-N heap. This is acceptable up to ~10K rows but
  degrades linearly past that.
- Index present and usable → the Stage 3b–3f pipeline runs.

### 3b. Prefilter

Before any partition reads, Lance resolves the filter expression:

- **`prefilter = true` (default)**: the filter is evaluated first, producing
  a row-id bitmap. The index scan consults this bitmap to skip ineligible
  rows within partitions. Safer for selective filters — the top-N heap
  never gets padded with rows that will be thrown away.
- **`prefilter = false`**: the index returns top-N (approximately) first,
  then the filter is applied to the result batches. Faster on large,
  non-selective filters but risks undershooting `N` after filtering.

### 3c. IVF partition selection

All `IvfFlat`, `IvfPq`, `IvfSq`, `IvfRq`, `IvfHnswPq`, and `IvfHnswSq`
share this step.

1. **Load centroids.** The index's centroid table is small
   (`num_partitions × dim × 4 B` for float32, so typically <10 MB). Lance
   loads it from the index file and caches it across queries.
2. **Compute query-to-centroid distances.** Same `distance_type` used at
   training time — L2, Cosine, Dot, or (for Flat+binary) Hamming.
3. **Top-nprobes heap.** A fixed-size heap of `minimum_nprobes` entries
   keeps the closest partitions. `maximum_nprobes`, if set, allows the
   downstream stages to pull more partitions if the initial top-k heap is
   not yet full.

For a 1B-row dataset with 30K partitions, this is one broadcast distance
scan over ~30K vectors — fast, linear in `num_partitions`, independent
of `N`.

### 3d. Within-partition scan

Each of the selected partitions has its own encoded vector storage. What
happens there depends on the encoding:

**IvfFlat** — raw vectors stored in the partition.

1. Load the partition's vector column page.
2. Compute exact `distance(query, v)` for every row (SIMD when possible).
3. Push into a partition-local top-N heap.

No approximation within the partition; the only approximation is which
partitions got selected.

**IvfPq** — asymmetric distance computation (ADC):

1. Precompute a lookup table once per partition:
   `dist_table[s][c] = distance(query_subvec_s, codebook[s][c])` for
   `s ∈ [0, num_sub_vectors)` and `c ∈ [0, 2^num_bits)`.
2. For each encoded vector (`num_sub_vectors` bytes at 8 bits), sum
   `dist_table[s][code_s]` across s. Pure table lookups, no
   decompression, no float math per candidate.
3. Push into partition-local top-N heap.

**IvfSq** — scalar dequantization:

1. For each row, convert uint8 → f32 using per-dim min/max.
2. Compute distance, push into heap.
3. Cheaper than Flat (one cache line per 4 dimensions) but slightly more
   CPU-heavy than PQ.

**IvfRq** — bit-unpacked asymmetric distance:

1. Read the partition's bit-packed codes (`num_bits` per dim).
2. For each vector, accumulate per-dim distance contributions from the
   float32 query against the bit-quantized code. At `num_bits = 1`, this
   reduces to XOR-popcount-style operations.
3. Push into partition-local top-N heap.

The query stays float32 throughout — only the stored vectors are
compressed. See [`ivf_rq_index.md`](./ivf_rq_index.md) for details on the
encoding itself.

**IvfHnswPq / IvfHnswSq** — graph search within the partition:

1. Enter the partition's HNSW graph at the pre-selected entry node.
2. Greedily descend through the graph layers until reaching layer 0.
3. At layer 0, expand a candidate set of size `ef` (defaults to
   `1.5 × top_k`), scoring each candidate with the compressed encoding
   (PQ or SQ ADC).
4. Return the top-N from the graph's priority queue.

The graph avoids scanning every vector in the partition; for large
partitions this is much faster than brute force, at the cost of more
memory for the graph edges.

### 3e. Cross-partition merge

Each selected partition produced a top-N heap (or top-`ef` list for
HNSW). These are merged via a single global heap of size `N` (or
`N × refine_factor` if refinement is on) keyed by compressed distance.

If a `maximum_nprobes` was provided and the global heap is still short
of `N` eligible rows after initial probes (e.g., because of prefilter
exclusions), Lance pulls more partitions from the ranked list until
either the heap fills or the ceiling is hit.

### 3f. Refinement (optional)

Triggered by `refine_factor = R > 1`:

1. The preceding stages return top `N × R` candidates, sorted by
   **compressed** distance.
2. For each candidate row-id, Lance reads the **raw** float32 vector
   from the vector column (random reads — this is the expensive part of
   refinement).
3. Recompute exact distance against the full-precision query.
4. Re-sort and take the top `N`.

Refinement is most valuable with aggressive compression (`IvfRq` with
`num_bits = 1`, or `IvfPq` with small `num_sub_vectors`). For `IvfFlat`
it is a no-op — the compressed distance was already exact.

The Scanner call site is `scanner.refine(refine_factor)` at
`rust/lancedb/src/table/query.rs:233–235`.

### 3g. Postfilter

If `prefilter = false`, the filter is applied here. Rows that fail the
predicate are dropped from the result batches. The final batch count may
be less than `N`; the caller is responsible for handling this (or for
setting `prefilter = true` upstream).

### 3h. Projection and streaming

The Scanner plan attaches the requested columns via
`scanner.project` / `project_with_transform` (called in Stage 2). Two
synthetic columns may also be appended: `_distance` (always, when a
vector search ran) and `_rowid` (if `with_row_id` was set).

At execution time, `execute_generic_query` (`table/query.rs:65–79`)
wraps the plan:

```rust
let plan = create_plan(table, query, options.clone()).await?;
let inner = execute_plan(plan, Default::default())?;
let inner = MaxBatchLengthStream::new_boxed(inner, options.max_batch_length as usize);
let inner = if let Some(timeout) = options.timeout {
    TimeoutStream::new_boxed(inner, timeout)
} else {
    inner
};
Ok(DatasetRecordBatchStream::new(inner))
```

`execute_plan` is from `lance_datafusion::exec`. It returns a
`SendableRecordBatchStream`; LanceDB wraps it in:

- `MaxBatchLengthStream` — splits batches that exceed the requested
  batch size.
- `TimeoutStream` — cancels the stream after the configured timeout.
- `DatasetRecordBatchStream` — the public streaming handle returned to
  the caller.

The caller can iterate batches asynchronously or collect them into a
single `RecordBatch`.

---

## Refinement in One Diagram

```
  nprobes partitions
         │
         ▼
  ┌──────────────┐
  │ Compressed   │   PQ / SQ / RQ codes → approximate distances
  │ top N·R      │
  └──────┬───────┘
         │  (random raw-vector reads only for these N·R ids)
         ▼
  ┌──────────────┐
  │ Raw float32  │   Exact distances
  │ re-rank      │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ Final top N  │   Returned to user
  └──────────────┘
```

Without refinement (`refine_factor = None`), the compressed distances
from the first box are the ones that rank the final results.

---

## Stage 4 — Remote Path (Contrast)

`RemoteTable::query` (`rust/lancedb/src/remote/table.rs:697–744`) posts a
JSON body to `/v1/table/{id}/query/`. The body
(`apply_vector_query_params`, `remote/table.rs:524–617`) mirrors
`VectorQueryRequest` one-to-one:

```json
{
  "version": 42,
  "vector": [0.1, 0.2, ...],
  "vector_column": "embedding",
  "k": 10,
  "offset": 0,
  "minimum_nprobes": 20,
  "maximum_nprobes": null,
  "refine_factor": 5,
  "ef": null,
  "lower_bound": null,
  "upper_bound": null,
  "columns": ["id", "text"],
  "filter": "status = 'active'",
  "prefilter": true,
  "distance_type": "L2",
  "bypass_vector_index": false,
  "with_row_id": false,
  "full_text_query": null
}
```

The server-side performs Stages 3a–3f (same code, same algorithms) and
streams back the result as Arrow IPC. The client side decodes it via
`read_arrow_stream`. From the caller's perspective, the only visible
differences from local are:

- Higher per-query latency (network round trip + server scheduling).
- `use_index` is controlled via `bypass_vector_index` in the request.
- Retries and timeouts are managed by `RestfulLanceDbClient`
  (exponential backoff with jitter per `remote/retry.rs`).

---

## Search-Time Tuning Cheat Sheet

| Knob               | Affects which stage   | Trade-off                                        |
|--------------------|-----------------------|--------------------------------------------------|
| `minimum_nprobes`  | 3c                    | Higher → more partitions scanned, better recall. |
| `maximum_nprobes`  | 3c, 3e                | Caps dynamic expansion when heap is short.       |
| `ef`               | 3d (HNSW only)        | Higher → wider graph search, better recall.      |
| `refine_factor`    | 3f                    | Higher → extra raw-vector reads, better recall.  |
| `prefilter`        | 3b vs 3g              | `true` is recall-safe; `false` can be faster.    |
| `use_index`        | skips 3a–3f           | Forces brute-force exact scan.                   |
| `distance_type`    | 3c, 3d                | Must match training, else silent corruption.     |
| `lower/upper_bound`| 3d, 3e                | Range gating; excludes rows outside.             |

Tuning rule of thumb:

1. Start with defaults (`nprobes=20`, no refine).
2. If recall is too low and the index is compressed (PQ/RQ/SQ): add
   `refine_factor = 5`, not more `nprobes`.
3. If recall is still low: increase `nprobes` next.
4. For `IvfHnswPq/Sq`: tune `ef` before `nprobes` — the graph already
   narrows the search.

---

## File Map

| File                                                           | Role                        |
|----------------------------------------------------------------|-----------------------------|
| `rust/lancedb/src/table.rs` (≈972–974)                         | `vector_search` entry point |
| `rust/lancedb/src/query.rs` (≈34, 840–850, 896–930, 1298–1316) | `VectorQuery*` state + execute |
| `rust/lancedb/src/table/query.rs` (≈60–246)                    | Scanner config + stream wrap |
| `rust/lancedb/src/remote/table.rs` (≈524–744)                  | Remote query body + HTTP path |
| `lance` and `lance-index` crates                               | Stage 3 internals (external) |
