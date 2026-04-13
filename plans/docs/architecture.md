# LanceDB Architecture Overview

LanceDB is a database for retrieval — vector, full-text, and hybrid search — built
on the [Lance](https://github.com/lancedb/lance) columnar data format. It supports
two backends: **local** (in-process, like SQLite) and **remote** (LanceDB Cloud).

The core is written in **Rust**, with bindings for **Python** (PyO3),
**TypeScript** (napi-rs), and **Java**.

---

## Project Layout

```
lancedb/
├── rust/lancedb/          Rust core library
│   └── src/
│       ├── lib.rs             Root module, DistanceType enum
│       ├── connection.rs      ConnectBuilder, Connection
│       ├── table.rs           Table, BaseTable trait, NativeTable
│       ├── query.rs           Query, VectorQuery, ExecutableQuery
│       ├── index.rs           Index types, IndexBuilder
│       ├── database.rs        Database trait
│       ├── error.rs           Error types
│       ├── embeddings.rs      EmbeddingFunction, EmbeddingRegistry
│       ├── remote/            LanceDB Cloud (HTTP client)
│       │   ├── client.rs          RestfulLanceDbClient
│       │   ├── db.rs              RemoteDatabase
│       │   ├── table.rs           RemoteTable
│       │   └── retry.rs          Retry logic
│       ├── table/             Table operations
│       │   ├── add_data.rs        AddDataBuilder
│       │   ├── query.rs           Query execution
│       │   ├── update.rs          UpdateBuilder
│       │   ├── merge.rs           MergeInsertBuilder
│       │   ├── dataset.rs         DatasetConsistencyWrapper
│       │   └── optimize.rs        Compaction, cleanup
│       ├── index/             Index internals
│       │   ├── vector.rs          Vector index builders
│       │   ├── scalar.rs          Scalar index builders
│       │   └── waiter.rs          Async index polling
│       └── data/              Data ingestion
│           ├── scannable.rs       Scannable trait
│           └── sanitize.rs        Data validation
├── python/                Python bindings (PyO3)
│   ├── src/                   Rust → Python bridge
│   └── python/lancedb/        Python API layer
├── nodejs/                TypeScript bindings (napi-rs)
│   └── src/                   Rust → TS bridge + TS classes
└── java/                  Java bindings
```

---

## High-Level Architecture

```
                     ┌──────────────────────────────────────────────┐
                     │              User Application                │
                     └──────┬──────────┬──────────────┬─────────────┘
                            │          │              │
                     ┌──────▼──┐  ┌────▼─────┐  ┌────▼────┐
                     │ Python  │  │ Node.js  │  │  Java   │
                     │  (PyO3) │  │ (napi-rs)│  │         │
                     └──────┬──┘  └────┬─────┘  └────┬────┘
                            │          │              │
                     ┌──────▼──────────▼──────────────▼─────────────┐
                     │              Rust Core (lancedb crate)       │
                     │                                              │
                     │  ┌────────────┐   ┌─────────┐   ┌─────────┐  │
                     │  │ Connection │──▶│  Table  │──▶│  Query  │  │
                     │  └────────────┘   └─────────┘   └─────────┘  │
                     │                                              │
                     │  ┌────────────┐   ┌─────────┐   ┌─────────┐  │
                     │  │ Embeddings │   │  Index  │   │  Data   │  │
                     │  └────────────┘   └─────────┘   └─────────┘  │
                     └───────────┬─────────────────────┬────────────┘
                                 │                     │
                 ┌───────────────▼──┐           ┌──────▼──────────┐
                 │   NativeTable    │           │   RemoteTable   │
                 │  (Local Backend) │           │ (Cloud Backend) │
                 └────────┬─────────┘           └──────┬──────────┘
                          │                            │
                 ┌────────▼─────────┐           ┌──────▼──────────┐
                 │  Lance Dataset   │           │  HTTP Client    │
                 │  (lance crate)   │           │  (reqwest)      │
                 └────────┬─────────┘           └──────┬──────────┘
                          │                            │
                 ┌────────▼─────────┐           ┌──────▼──────────┐
                 │   Object Store   │           │  LanceDB Cloud  │
                 │  (local/S3/GCS/  │           │    REST API     │
                 │   Azure/OSS)     │           └─────────────────┘
                 └──────────────────┘
```

---

## Core Abstractions

### Connection & Database

The entry point is `connect(uri)`, which routes to the appropriate backend:

```
connect(uri) ─────────────────────────────────────┐
  │                                               │
  ├── uri = "db://..."  ──▶  RemoteDatabase       │
  │                          (LanceDB Cloud)      │
  │                                               │
  └── uri = path/s3/gs  ──▶  ListingDatabase      │
                             (local / object      │
                              store)              │
                                                  │
  Both implement the Database trait:              │
  ┌───────────────────────────────────────────┐   │
  │ Database (async trait)                    │   │
  │                                           │   │
  │  Table operations:                        │   │
  │   list_tables(request) -> ListTablesResp  │   │
  │   create_table(request) -> Table          │   │
  │   open_table(request) -> Table            │   │
  │   drop_table(name, namespace_path)        │   │
  │   drop_all_tables(namespace_path)         │   │
  │   clone_table(request)                    │   │
  │   rename_table(old, new, ns_path, ns_path)│   │
  │                                           │   │
  │  Namespace operations:                    │   │
  │   list_namespaces(request)                │   │
  │   create_namespace(request)               │   │
  │   drop_namespace(request)                 │   │
  │   describe_namespace(request)             │   │
  │   namespace_client()                      │   │
  │   namespace_client_config()               │   │
  └───────────────────────────────────────────┘   │
```

### Table

`Table` is the central abstraction. It wraps either a `NativeTable` (local Lance
dataset) or a `RemoteTable` (HTTP client) behind the `BaseTable` trait:

```
  ┌──────────────────────────────────────────────────────┐
  │ Table                                                │
  │   inner: Arc<dyn BaseTable>                          │
  │   database: Option<Arc<dyn Database>>                │
  │   embedding_registry: Arc<dyn EmbeddingRegistry>     │
  └───────────────────┬──────────────────────────────────┘
                      │
           ┌──────────▼──────────┐
           │ BaseTable (trait)   │
           │                     │
           │  schema()           │
           │  count_rows()       │
           │  add(data)          │
           │  update()           │
           │  delete()           │
           │  merge_insert()     │
           │  query()            │
           │  create_index()     │
           │  list_indices()     │
           │  optimize()         │
           │  version()          │
           │  checkout()         │
           │  list_versions()    │
           └──────┬───────┬──────┘
                  │       │
     ┌────────────▼┐  ┌───▼───────────┐
     │ NativeTable │  │  RemoteTable  │
     │             │  │               │
     │ dataset:    │  │ client: HTTP  │
     │   Lance     │  │ schema_cache  │
     │   Dataset   │  │ server_ver    │
     └─────────────┘  └───────────────┘
```

### Query System

Queries use a **builder pattern** with two primary types:

```
table.query()                    table.vector_search(vec)
    │                                │
    ▼                                ▼
┌─────────┐                   ┌──────────────┐
│  Query  │                   │ VectorQuery  │
│         │                   │              │
│ limit() │                   │ limit()      │
│ offset()│                   │ nprobes()    │
│ filter()│                   │ ef()         │
│ select()│                   │ refine()     │
│ fts()   │                   │ distance()   │
│ rerank()│                   │ filter()     │
└────┬────┘                   └──────┬───────┘
     │                               │
     │    Both implement             │
     │    ExecutableQuery             │
     ▼                               ▼
  execute() ──▶ SendableRecordBatchStream
```

**Query execution path (local):**

```
execute()
  │
  ├─▶ create_plan()
  │     │
  │     ├── Build Lance Scanner
  │     ├── Apply nearest(column, vector, k) for vector search
  │     ├── Apply prefilter / postfilter
  │     ├── Apply full-text search
  │     ├── Apply projection (select columns)
  │     └── Apply limit / offset
  │
  └─▶ DataFusion execute_plan()
        │
        └── Stream RecordBatch results
```

**Query execution path (remote):**

```
execute()
  │
  └─▶ HTTP POST /v1/table/{id}/query/
        │
        └── Deserialize Arrow IPC response
```

---

## Data Ingestion

```
table.add(data)
  │
  ▼
┌──────────────────┐
│  AddDataBuilder  │
│                  │
│  mode: Append    │    ┌─────────────────────────────┐
│    or Overwrite  │    │   Scannable (trait)         │
│                  │◄───│                             │
│  data: Scannable │    │   RecordBatch               │
│                  │    │   Vec<RecordBatch>          │
│  embeddings?     │    │   RecordBatchReader         │
└────────┬─────────┘    │   SendableRecordBatchStream │
         │              └─────────────────────────────┘
         ▼
  ┌──────────────┐      ┌───────────────────┐
  │ Apply        │─────▶│ Validate schema   │
  │ Embeddings   │      │ + cast columns    │
  └──────────────┘      └─────────┬─────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │  NaN vector check          │
                    │  (Error or Keep)           │
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │  Write to Lance Dataset    │
                    │  (local) or HTTP POST      │
                    │  (remote)                  │
                    └────────────────────────────┘
```

Other write operations:

| Operation        | Builder              | Description                              |
|------------------|----------------------|------------------------------------------|
| `add()`          | `AddDataBuilder`     | Append or overwrite data                 |
| `update()`       | `UpdateBuilder`      | SET columns WHERE filter                 |
| `delete()`       | direct call          | Delete rows matching predicate           |
| `merge_insert()` | `MergeInsertBuilder` | Upsert with matched/not-matched clauses  |

---

## Index System

LanceDB supports vector, scalar, and full-text indices:

```
┌────────────────────────────────────────────────────────────────┐
│                      Index Types                               │
│                                                                │
│  Vector Indices              Scalar Indices     Other          │
│  ┌───────────────┐           ┌──────────┐       ┌─────────┐    │
│  │ IvfFlat       │           │ BTree    │       │ FTS     │    │
│  │ IvfPq         │           │ Bitmap   │       │ (BM25)  │    │
│  │ IvfSq         │           │ LabelList│       └─────────┘    │
│  │ IvfRq         │           └──────────┘                      │
│  │ IvfHnswPq     │                                             │
│  │ IvfHnswSq     │           Auto: picks type                  │
│  └───────────────┘           based on column dtype             │
└────────────────────────────────────────────────────────────────┘
```

**Index creation flow:**

```
table.create_index(&["vector"], Index::IvfPq(...))
  │
  ▼
┌──────────────┐
│ IndexBuilder │──▶ make_index_params()
└──────┬───────┘          │
       │           ┌──────▼────────────────────────┐
       │           │ Convert to lance_index types: │
       │           │  IvfBuildParams               │
       │           │  PQBuildParams                │
       │           │  SQBuildParams                │
       │           │  HnswBuildParams              │
       │           │  RQBuildParams                │
       │           └──────┬────────────────────────┘
       │                  │
       ▼                  ▼
  dataset.create_index(params)
       │
       └──▶ Lance core builds index on disk
```

---

## Embedding System

```
┌─────────────────────────────────────────────────────┐
│ EmbeddingRegistry                                   │
│   register(name, Arc<dyn EmbeddingFunction>)        │
│   get(name) -> Option<Arc<dyn EmbeddingFunction>>   │
│                                                     │
│   In-memory impl: MemoryRegistry (HashMap + RwLock) │
└──────────┬──────────────────────────────────────────┘
           │
           │  Registered providers:
           ▼
  ┌────────────────┐  ┌──────────────────────┐  ┌─────────┐
  │    OpenAI      │  │ Sentence Transformers│  │ Bedrock │
  │ (API-based)    │  │ (local Candle)       │  │ (AWS)   │
  │                │  │                      │  │         │
  │ ada-002: 1536D │  │ HuggingFace models   │  │ Titan   │
  │ 3-small: 1536D │  │ CPU/GPU inference    │  │ Cohere  │
  │ 3-large: 3072D │  │                      │  │         │
  └────────────────┘  └──────────────────────┘  └─────────┘
```

Embeddings are applied during data ingestion: `AddDataBuilder` wraps the data
source in `WithEmbeddingsScannable`, which computes embedding columns on the fly.

Schema stores `ColumnDefinition` metadata to distinguish physical columns from
embedding-generated columns.

---

## Remote (Cloud) Backend

```
  ┌──────────────────────────────────────────────┐
  │  RemoteDatabase                              │
  │    client: RestfulLanceDbClient              │
  │    table_cache: Cache<String, RemoteTable>   │
  └──────────┬───────────────────────────────────┘
             │
             ▼
  ┌──────────────────────────────────────────────┐
  │  ClientConfig                                │
  │    timeout / retry config                    │
  │    TLS (mTLS support)                        │
  │    custom header providers                   │
  │    user_id: Option<String>                   │
  │      (sent as x-lancedb-user-id header)      │
  └──────────┬───────────────────────────────────┘
             │
             ▼
  ┌──────────────────────────────────────────────┐
  │  RestfulLanceDbClient                        │
  │    host, retry_config, sender                │
  │    id_delimiter, header_provider             │
  └──────────┬───────────────────────────────────┘
             │
             │  REST API calls
             ▼
  ┌──────────────────────────────────────────────┐
  │  LanceDB Cloud                               │
  │                                              │
  │  /v1/table/{id}/query/      Vector search    │
  │  /v1/table/{id}/insert/     Data insert      │
  │  /v1/table/{id}/update/     Row updates      │
  │  /v1/table/{id}/delete/     Row deletes      │
  │  /v1/table/{id}/create_index/  Index build   │
  │  /v1/table/{id}/merge/      Merge insert     │
  └──────────────────────────────────────────────┘
```

**Retry strategy:** Exponential backoff with jitter, separate counters for
request/connect/read failures, status-code-based retry decisions.

**Server versioning:** `ServerVersion` gates features by semantic version
(e.g., multivector >= 0.2.0, multipart write >= 0.4.0).

---

## Storage Layer

LanceDB stores data in the **Lance columnar format**, which supports:

- **Versioning:** Every write creates a new version; time-travel queries supported
- **Zero-copy reads:** Memory-mapped access to columnar data
- **Object store backends:** Local filesystem, S3, GCS, Azure, Alibaba OSS

```
  ┌────────────────────────────────────────────────┐
  │  DatasetConsistencyWrapper                     │
  │    Wraps Lance Dataset with:                   │
  │    - Thread-safe access (Arc + RwLock)         │
  │    - Lazy initialization                       │
  │    - Read consistency interval                 │
  │    - Managed versioning (namespace commits)    │
  └───────────────────┬────────────────────────────┘
                      │
                      ▼
  ┌────────────────────────────────────────────────┐
  │  Lance Dataset                                 │
  │    Columnar data in Arrow format               │
  │    Manifest-based versioning                   │
  │    Fragment-based storage                      │
  └───────────────────┬────────────────────────────┘
                      │
           ┌──────────┴──────────┐
           ▼                     ▼
    ┌────────────┐       ┌──────────────┐
    │  Local FS  │       │ Object Store │
    │            │       │ (S3/GCS/     │
    │            │       │  Azure/OSS)  │
    └────────────┘       └──────────────┘
```

---

## Distance Metrics

```
  DistanceType enum:
  ┌───────────┬────────────────────────────────────┐
  │ L2        │ Euclidean distance     [0, ∞)      │  ← default
  │ Cosine    │ Cosine similarity      [0, 2]      │
  │ Dot       │ Dot product            (-∞, ∞)     │
  │ Hamming   │ Hamming distance       (binary)    │
  └───────────┴────────────────────────────────────┘

  IMPORTANT: The distance type used at index TRAINING time
  must match the distance type used at SEARCH time.
```

---

## Key Dependencies

| Crate           | Purpose                                      |
|-----------------|----------------------------------------------|
| `lance`         | Core columnar dataset (read, write, scan)    |
| `lance-index`   | Vector and scalar index building             |
| `arrow`         | Apache Arrow data format                     |
| `datafusion`    | SQL query planning and execution             |
| `tokio`         | Async runtime                                |
| `reqwest`       | HTTP client (remote feature)                 |
| `async-trait`   | Async trait support                          |

---

## Error Handling

All operations return `lancedb::Result<T>`, wrapping the `Error` enum:

| Variant                    | Description                          |
|----------------------------|--------------------------------------|
| `InvalidTableName`         | Bad table name                       |
| `InvalidInput`             | Bad parameters                       |
| `TableNotFound`            | Table does not exist                 |
| `TableAlreadyExists`       | Table creation conflict              |
| `IndexNotFound`            | Index does not exist                 |
| `EmbeddingFunctionNotFound`| Missing embedding provider           |
| `Schema`                   | Schema mismatch                      |
| `Lance` / `Arrow`          | Wrapped errors from lance/arrow      |
| `Http` / `Retry`           | Network errors (remote backend)      |
| `NotSupported`             | Feature not available                |

Nested errors from Lance, Arrow, and DataFusion are automatically unwrapped
and converted to the appropriate LanceDB error variant.
