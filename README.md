<div align="center">

# 🎓 Algorithmic Learning Path Generator

**An intelligent system for automated learning path construction**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Documentation](#-documentation)

</div>

---

## 📖 Overview

The **Algorithmic Learning Path Generator** is a zero-label inference system that automatically constructs prerequisite learning paths from educational content. It processes web articles and YouTube videos to extract learning concepts, canonicalizes them through semantic clustering, and builds a directed acyclic graph (DAG) representing concept dependencies.

### 🎯 Use Cases

- **Curriculum Planning**: Automatically generate learning paths for educational content
- **Knowledge Mapping**: Visualize prerequisite relationships between concepts
- **Content Organization**: Structure educational resources by difficulty and dependencies
- **Self-Learning**: Discover optimal learning sequences for complex topics

### ✨ Key Features

- **🔄 Multi-Phase Pipeline**: Automated ingestion → extraction → graph construction
- **🌐 Multi-Source Support**: Process web articles and YouTube transcripts
- **🧠 NLP-Powered**: Concept extraction using NLTK and sentence transformers
- **🎯 Semantic Clustering**: Canonical concept identification with 85%+ similarity
- **📊 Difficulty Scoring**: Automatic beginner/intermediate/advanced classification
- **🔗 DAG Construction**: FAISS-accelerated prerequisite graph with cycle detection
- **🎚️ Calibration Tools**: Sweep and optimize graph parameters automatically
- **✅ Comprehensive Testing**: 98+ tests covering all pipeline stages

---

## 🚀 Quick Start

```bash
# 1. Clone and setup environment
git clone <repository-url>
cd SAH
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
python -m nltk.downloader punkt_tab averaged_perceptron_tagger_eng

# 3. Run the complete pipeline
python -m src.ingest --input data/urls.json --db ./data/resources.db
python -m src.extract_concepts --db ./data/resources.db
python -m src.graph_builder --db ./data/resources.db

# 4. Verify installation
pytest tests/ -v -k "not live"
```

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 2GB+ RAM for embedding models
- Optional: CUDA-capable GPU for faster processing

### Step-by-Step Setup

```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate environment
# Linux/Mac:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# 3. Install Python packages
pip install -r requirements.txt

# 4. Download NLTK resources
python -m nltk.downloader punkt_tab averaged_perceptron_tagger_eng

# 5. Verify installation
python -c "import faiss, nltk, sentence_transformers; print('✓ All dependencies installed')"
```

---

## 📚 Usage

### Complete Pipeline Example

```bash
# Step 1: Ingest resources (Phase 1)
python -m src.ingest \
    --input data/urls.json \
    --db ./data/resources.db \
    --batch-size 5

# Step 2: Extract and canonicalize concepts (Phase 2)
python -m src.extract_concepts \
    --db ./data/resources.db \
    --batch-size 64 \
    --cluster-threshold 0.85

# Step 3: Build concept graph (Phase 3)
python -m src.graph_builder \
    --db ./data/resources.db \
    --top-k 15 \
    --max-out 8 \
    --sim-min 0.50 \
    --conf-threshold 0.55

# Optional: Calibrate parameters
python -m src.graph_builder --db ./data/resources.db --calibrate-only

# Apply calibrated configuration
python -m src.graph_builder --db ./data/resources.db \
    --apply-config ./data/phase3_calibration/best_config.json
```

### Phase 1: Resource Ingestion

Fetches and processes learning content from URLs.

```bash
python -m src.ingest \
    --input data/urls.json \
    --db ./data/resources.db \
    --batch-size 5
```

**Input Format** (`data/urls.json`):
```json
{
  "urls": [
    "https://example.com/machine-learning-basics",
    "https://youtube.com/watch?v=..."
  ]
}
```

**Supported Content Types:**
- 📄 Web articles (via Trafilatura)
- 🎥 YouTube videos (automatic transcript extraction)

**Output**: SQLite database with extracted content and metadata

### Phase 2: Concept Extraction

Extracts and canonicalizes learning concepts using NLP.

```bash
python -m src.extract_concepts \
    --db ./data/resources.db \
    --batch-size 64 \
    --cluster-threshold 0.85
```

**Process:**
1. Extracts noun phrases using NLTK POS tagging
2. Generates embeddings with `all-MiniLM-L6-v2`
3. Clusters similar concepts (agglomerative clustering)
4. Scores difficulty (beginner/intermediate/advanced)

**Output**: `phase2_summary.json`
```json
{
  "total_resources": 250,
  "raw_concepts": 45678,
  "canonical_concepts": 13071,
  "avg_concepts_per_resource": 182.7
}
```

### Phase 3: Graph Construction

Builds a prerequisite DAG using FAISS nearest-neighbor search.

```bash
python -m src.graph_builder \
    --db ./data/resources.db \
    --top-k 15 \
    --max-out 8 \
    --sim-min 0.50 \
    --conf-threshold 0.55
```

**Algorithm:**
1. Build FAISS index on concept embeddings
2. Find k-nearest neighbors for each concept
3. Score edges: `0.6×similarity + 0.3×difficulty_gap + 0.1×co-occurrence`
4. Direction: easier → harder concepts
5. Prune to top N edges per source
6. Break cycles greedily
7. Validate DAG properties

**Output**: `phase3_summary.json`
```json
{
  "total_concepts": 13071,
  "total_edges": 39510,
  "avg_out_degree": 3.27,
  "max_depth": 32,
  "isolated_nodes_count": 999,
  "is_dag": true
}
```

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────┐
│   URLs Input    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      Phase 1: Ingestion
│  Web Fetchers   │───►  • Trafilatura (articles)
│  YouTube API    │      • YouTube transcripts
└────────┬────────┘      • Content validation
         │
         ▼
┌─────────────────┐
│  RawResources   │
│    (SQLite)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      Phase 2: Extraction
│  NLP Pipeline   │───►  • NLTK POS tagging
│  Clustering     │      • Semantic embeddings
│  Difficulty     │      • Concept canonicalization
└────────┬────────┘      • Difficulty scoring
         │
         ▼
┌─────────────────┐
│ Canonical       │
│ Concepts (DB)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      Phase 3: Graph Building
│  FAISS Index    │───►  • k-NN search
│  Edge Scoring   │      • Prerequisite detection
│  DAG Validator  │      • Cycle breaking
└────────┬────────┘      • Topological validation
         │
         ▼
┌─────────────────┐
│  Learning Path  │
│   Graph (DAG)   │
└─────────────────┘
```

### Project Structure

```
SAH/
├── 📄 README.md
├── 📄 requirements.txt
│
├── 📁 src/                           # Core application code
│   ├── ingest.py                     # Phase 1: Resource ingestion CLI
│   ├── extract_concepts.py           # Phase 2: Concept extraction CLI
│   ├── graph_builder.py              # Phase 3: Graph construction CLI
│   ├── phase3_calibrate.py           # Parameter calibration engine
│   ├── phase3_recalibrate_utils.py   # Confidence scoring utilities
│   │
│   ├── db.py                         # Phase 1+2 database layer
│   ├── db_graph.py                   # Phase 3 graph database
│   ├── models.py                     # Pydantic data models
│   ├── faiss_index.py                # FAISS indexing & search
│   ├── dag_validator.py              # Cycle detection & metrics
│   │
│   ├── utils.py                      # Logging, retry, serialization
│   ├── utils_graph.py                # Graph-specific utilities
│   │
│   ├── 📁 fetchers/                  # Content fetching modules
│   │   ├── trafilatura_fetcher.py    # Web article extraction
│   │   └── youtube_fetcher.py        # YouTube transcript API
│   │
│   ├── 📁 extractors/                # Concept extraction
│   │   └── spacy_extractor.py        # NLTK-based NP extraction
│   │
│   ├── 📁 embeddings/                # Embedding generation
│   │   └── embedder.py               # Sentence transformer wrapper
│   │
│   └── 📁 canonicalization/          # Concept clustering
│       └── clusterer.py              # Agglomerative clustering
│
├── 📁 tests/                         # Comprehensive test suite
│   ├── test_ingest.py                # 35 Phase 1 tests
│   ├── test_extract_concepts.py      # 24 Phase 2 tests
│   ├── test_graph_builder.py         # 20 Phase 3 tests
│   ├── test_phase3_calibration.py    # 19 calibration tests
│   └── sample_*.json                 # Test fixtures
│
├── 📁 data/                          # Generated data & outputs
│   ├── resources.db                  # SQLite database
│   ├── *_summary.json                # Pipeline summaries
│   └── phase3_calibration/           # Calibration results
│
└── 📁 scripts/                       # Utility scripts
    ├── check_youtube.py
    └── verify_calibration.py
```

---

## 🔧 Phase 3 CLI Reference

### Command-Line Arguments

```bash
python -m src.graph_builder \
    --db ./data/resources.db \
    --top-k 15 \
    --max-out 8 \
    --sim-min 0.50 \
    --conf-threshold 0.55
```

| Parameter          | Default               | Description                                        |
|--------------------|-----------------------|----------------------------------------------------|
| `--db`             | `./data/resources.db` | Path to SQLite database                            |
| `--top-k`          | `15`                  | Number of FAISS neighbors per concept              |
| `--max-out`        | `8`                   | Maximum outgoing edges per source node             |
| `--sim-min`        | `0.50`                | Minimum cosine similarity threshold (0.0-1.0)      |
| `--conf-threshold` | `0.55`                | Minimum confidence score for edge inclusion        |
| `--use-gpu-faiss`  | `off`                 | Enable GPU-accelerated FAISS (requires faiss-gpu)  |
| `--dry-run`        | `off`                 | Compute but don't persist edges (testing mode)     |

### Phase 3 Pipeline Steps

The graph construction follows this algorithmic sequence:

1. **Load** — Canonical concepts + mean-pooled embeddings from `ExtractedConcepts`
2. **FAISS** — Build `IndexFlatIP` (cosine via inner-product on normalized vectors)
3. **Candidates** — Find top-k neighbors per concept, filter by similarity minimum
4. **Score** — Calculate confidence: `0.60×sim + 0.30×diff_gap + 0.10×co-occurrence`
5. **Direction** — Create edges only from easier → harder (difficulty_score ordering)
6. **Prune** — Keep top `max_out` edges per source node
7. **Acyclicity** — Greedy cycle-breaking (remove minimum-confidence edge per cycle)
8. **Persist** — Write to `ConceptEdges` table (idempotent clear + insert)
9. **Validate** — Topological sort check + graph metrics

### Calibration Workflow

Optimize graph parameters automatically:

```bash
# 1. Run calibration sweep (generates reports, no DB writes)
python -m src.graph_builder --db ./data/resources.db --calibrate-only

# 2. Review results in data/phase3_calibration/
#    - best_config.json                    (recommended settings)
#    - phase3_recalibration_report.json    (detailed analysis)
#    - baseline_result.json / balanced_result.json / high_recall_result.json

# 3. Apply best configuration
python -m src.graph_builder --db ./data/resources.db \
    --apply-config ./data/phase3_calibration/best_config.json
```

**Calibration Profiles:**
- **Baseline**: Conservative settings, high precision
- **Balanced**: Optimized precision/recall trade-off
- **High Recall**: More edges, better concept coverage

### Expected Output

**`phase3_summary.json`:**
```json
{
  "total_concepts": 13071,
  "total_edges": 39510,
  "avg_out_degree": 3.27,
  "max_depth": 32,
  "isolated_nodes_count": 999,
  "is_dag": true,
  "removed_edges": 0
}
```

---

## 🗃️ Database Schema

The system uses SQLite with three main phases of data storage:

### Phase 1: `RawResources`

Stores raw ingested content from web sources.

| Column         | Type      | Constraints                                     |
|----------------|-----------|-------------------------------------------------|
| `id`           | INTEGER   | PRIMARY KEY AUTOINCREMENT                       |
| `url`          | TEXT      | UNIQUE NOT NULL                                 |
| `content_type` | TEXT      | CHECK('article', 'youtube', 'unknown')          |
| `title`        | TEXT      | Extracted title                                 |
| `raw_text`     | TEXT      | Full text content                               |
| `status`       | TEXT      | CHECK(ok/no_content/failed/skipped/...)         |
| `extracted_at` | TIMESTAMP | Ingestion timestamp                             |
| `notes`        | TEXT      | Error messages or metadata                      |

### Phase 2: `ExtractedConcepts`

Individual concept extractions with embeddings.

| Column         | Type      | Constraints                                     |
|----------------|-----------|-------------------------------------------------|
| `id`           | INTEGER   | PRIMARY KEY AUTOINCREMENT                       |
| `resource_id`  | INTEGER   | FK → RawResources(id)                           |
| `concept`      | TEXT      | NOT NULL                                        |
| `canonical_id` | INTEGER   | FK → CanonicalConcepts(id)                      |
| `sentence`     | TEXT      | Context sentence                                |
| `embedding`    | BLOB      | float32 × 384 (sentence-transformers)           |
| `created_at`   | TIMESTAMP | Extraction timestamp                            |

**Index:** `idx_resource` on `resource_id`, `idx_canonical` on `canonical_id`

### Phase 2: `CanonicalConcepts`

Clustered canonical representations of concepts.

| Column              | Type      | Constraints                                |
|---------------------|-----------|--------------------------------------------|
| `id`                | INTEGER   | PRIMARY KEY AUTOINCREMENT                  |
| `canonical_concept` | TEXT      | UNIQUE NOT NULL                            |
| `difficulty_score`  | REAL      | Range [0.0, 1.0]                          |
| `difficulty_bucket` | TEXT      | CHECK('beginner','intermediate','advanced')|
| `example_sentence`  | TEXT      | Representative context                     |
| `resource_count`    | INTEGER   | Number of resources mentioning concept     |
| `created_at`        | TIMESTAMP | Canonicalization timestamp                 |

### Phase 3: `ConceptEdges`

Prerequisite relationships forming the DAG.

| Column              | Type      | Constraints                                |
|---------------------|-----------|--------------------------------------------|
| `id`                | INTEGER   | PRIMARY KEY AUTOINCREMENT                  |
| `source_concept_id` | INTEGER   | FK → CanonicalConcepts(id), indexed        |
| `target_concept_id` | INTEGER   | FK → CanonicalConcepts(id), indexed        |
| `similarity`        | REAL      | NOT NULL, cosine similarity [0, 1]         |
| `confidence`        | REAL      | NOT NULL, edge score [0, 1]                |
| `created_at`        | TIMESTAMP | Edge creation timestamp                    |

**Constraints:** 
- UNIQUE(`source_concept_id`, `target_concept_id`)
- **Indexes:** `idx_source` on `source_concept_id`, `idx_target` on `target_concept_id`

---

## 🧪 Testing

The project includes 98+ comprehensive tests covering all pipeline stages.

### Running Tests

```bash
# Run all tests (no network required)
pytest tests/ -v -k "not live"

# Run tests by phase
pytest tests/test_ingest.py -v                # Phase 1 (35 tests)
pytest tests/test_extract_concepts.py -v      # Phase 2 (24 tests)
pytest tests/test_graph_builder.py -v         # Phase 3 (20 tests)
pytest tests/test_phase3_calibration.py -v    # Calibration (19 tests)

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_graph_builder.py::test_dag_validation -v
```

### Test Features

- ✅ **No External Dependencies**: All tests use deterministic fixtures
- ✅ **Fast Execution**: Complete test suite runs in ~30 seconds
- ✅ **Isolated**: Each test uses temporary databases
- ✅ **Comprehensive**: Covers success/failure paths and edge cases

---

## 🛠️ Tech Stack

| Layer                  | Technology                                    |
|------------------------|-----------------------------------------------|
| **NLP**                | NLTK 3.x (POS tagging, NP chunking)          |
| **Embeddings**         | sentence-transformers (`all-MiniLM-L6-v2`)   |
| **Clustering**         | scikit-learn (AgglomerativeClustering)       |
| **Vector Search**      | FAISS (CPU/GPU) `IndexFlatIP`                |
| **Graph Processing**   | NetworkX (cycle detection, topological sort) |
| **Database**           | SQLite3 (WAL mode, foreign keys)             |
| **Validation**         | Pydantic 2.x                                 |
| **Testing**            | pytest                                       |
| **Content Extraction** | Trafilatura, youtube-transcript-api          |

---

## 🐛 Troubleshooting

### Common Issues

| Problem                           | Solution                                                                 |
|-----------------------------------|--------------------------------------------------------------------------|
| `No CanonicalConcepts found`      | Ensure Phase 2 completed successfully. Check `phase2_summary.json`.     |
| `ModuleNotFoundError: faiss`      | Install FAISS: `pip install faiss-cpu`                                  |
| FAISS GPU not detected            | Install `faiss-gpu` and verify CUDA: `python -c "import faiss; print(faiss.get_num_gpus())"` |
| Too many isolated nodes           | Reduce thresholds: `--sim-min 0.40 --conf-threshold 0.45`              |
| Graph construction is slow        | Reduce search space: `--top-k 10 --max-out 5`                          |
| High memory usage                 | Process in smaller batches or use GPU FAISS                              |
| `N > 25k concepts` warning        | Consider `IndexIVF` or `IndexHNSW` for large-scale datasets             |
| Cycles detected in graph          | Pipeline auto-removes cycles via greedy algorithm                        |
| YouTube transcript unavailable    | Video may have transcripts disabled. Check with `scripts/check_youtube.py` |

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
# Set logging level
export LOG_LEVEL=DEBUG

# Run with verbose output
python -m src.graph_builder --db ./data/resources.db --dry-run
```

### Performance Tips

1. **GPU Acceleration**: Use `--use-gpu-faiss` for 10-50× speedup on large datasets
2. **Batch Processing**: Increase `--batch-size` if you have sufficient RAM
3. **Parallel Processing**: FAISS and clustering operations use all available CPU cores
4. **Database Optimization**: SQLite WAL mode is enabled by default for concurrency

---

## 📊 Performance Metrics

Typical performance on standard hardware (Intel i7, 16GB RAM):

| Phase | Operation            | Time (250 resources) | Memory |
|-------|----------------------|----------------------|--------|
| 1     | Ingestion            | ~5 minutes           | <500MB |
| 2     | Concept Extraction   | ~10 minutes          | ~2GB   |
| 2     | Clustering           | ~3 minutes           | ~1GB   |
| 3     | FAISS Index Build    | ~20 seconds          | ~1GB   |
| 3     | Edge Generation      | ~1 minute            | ~1.5GB |
| 3     | Cycle Detection      | <5 seconds           | <500MB |

**Total Pipeline**: ~20 minutes for 250 resources → 13,000 concepts → 40,000 edges

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd SAH

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy

# Run tests before committing
pytest tests/ -v
black src/ tests/
flake8 src/ tests/
```

### Contribution Guidelines

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Write** tests for new functionality
4. **Ensure** all tests pass: `pytest tests/ -v`
5. **Format** code: `black src/ tests/`
6. **Commit** changes: `git commit -m 'Add amazing feature'`
7. **Push** to branch: `git push origin feature/amazing-feature`
8. **Open** a Pull Request

### Areas for Contribution

- 🌐 Additional content fetchers (PDF, DOI, arXiv)
- 📊 Graph visualization tools
- 🔍 Alternative concept extraction methods
- 🎯 Improved difficulty scoring algorithms
- 📈 Performance optimizations
- 📚 Documentation improvements
- 🧪 Additional test cases

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **sentence-transformers** for efficient semantic embeddings
- **FAISS** (Facebook AI Similarity Search) for vector search
- **Trafilatura** for robust web content extraction
- **NetworkX** for graph algorithms
- **NLTK** for natural language processing

---

## 📞 Support

- 📖 **Documentation**: See inline docstrings and this README
- 🐛 **Issues**: Report bugs via GitHub Issues
- 💬 **Discussions**: Use GitHub Discussions for questions
- 📧 **Contact**: [Your contact information]

---

<div align="center">

**Built with ❤️ for automated learning path generation**

[⬆ Back to Top](#-algorithmic-learning-path-generator)

</div>
