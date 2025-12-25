DATASET_ID ?= irlspbru/RusLawOD
SPLIT ?= train
N ?= 1000
SEED ?= 42
OUTPUT_DIR ?= data/raw
STRATEGY ?= stream
DATA_FILES ?=
STATUS_FILTER ?=
DOC_TYPE_FILTER ?=
CHUNK_SIZE_CHARS ?= 1024
OVERLAP_CHARS ?= 64

.PHONY: help venv sync lint format test test-network serve ui ingest build-index eval hf-slice clean

help:
	@echo "Targets:"
	@echo "  make venv                - create virtual environment (uv)"
	@echo "  make sync                - install/sync dependencies (uv sync)"
	@echo "  make lint                - ruff lint"
	@echo "  make lint-fix            - ruff lint and fix"
	@echo "  make format              - ruff format"
	@echo "  make fmt                 - lint and format"
	@echo "  make check               - lint and test"
	@echo "  make test                - pytest (no network/slow)"
	@echo "  make test-network        - pytest including network tests"
	@echo "  make serve               - run FastAPI (uvicorn)"
	@echo "  make ui                  - run Streamlit UI"
	@echo "  make ingest              - run ingestion script"
	@echo "  make build-index         - build indexes (bm25/faiss)"
	@echo "  make eval                - run RAGAS evaluation"
	@echo "  make hf-slice            - slice HF dataset"
	@echo "  make retrieve            - offline CLI retrieval"
	@echo "  make docs-catalog        - build docs catalog CSV"
	@echo "  make evalset-draft       - generate draft evalset (auto questions)"
	@echo "  make evalset-autolink    - autolink draft evalset to chunks"
	@echo "  make evalset-llm         - LLM-assisted evalset generation (network)"
	@echo "  make evalset-validate    - validate evalset.jsonl"
	@echo "  make eval-ragas          - RAGAS retrieval evaluation"
	@echo "  make clean               - remove caches"

venv:
	uv venv

sync:
	uv sync

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check --fix .

format:
	uv run ruff format .

fmt: format lint-fix

check: lint test

test:
	uv run pytest

test-network:
	uv run pytest -m "network"

serve:
	uv run python scripts/serve.py

ui:
	uv run streamlit run ui/streamlit_app.py

ingest:
	uv run python scripts/ingest.py

build-index:
	uv run python scripts/build_index.py

eval:
	uv run python scripts/evaluate.py

hf-slice:
	DATASET_ID=$(DATASET_ID) \
	SPLIT=$(SPLIT) \
	N=$(N) \
	SEED=$(SEED) \
	OUTPUT_DIR=$(OUTPUT_DIR) \
	STRATEGY=$(STRATEGY) \
	DATA_FILES="$(DATA_FILES)" \
	STATUS_FILTER="$(STATUS_FILTER)" \
	DOC_TYPE_FILTER="$(DOC_TYPE_FILTER)" \
	uv run python scripts/hf_slice.py

retrieve:
	uv run python scripts/retrieve_cli.py --query "$(QUERY)" --k $(K) --retriever $(RETRIEVER) --chunk-size-chars $(CHUNK_SIZE_CHARS) --overlap-chars $(OVERLAP_CHARS)

docs-catalog:
	uv run python scripts/docs_catalog.py

evalset-draft:
	uv run python scripts/generate_evalset_draft.py

evalset-autolink:
	uv run python scripts/autolink_evalset_chunks.py

evalset-llm:
	uv run python scripts/generate_evalset_llm.py

evalset-validate:
	uv run python scripts/validate_evalset.py

eval-ragas:
	uv run python scripts/evaluate_ragas.py --retrievers $(RETRIEVERS) --k $(K)

chunk-stats:
	uv run python scripts/chunk_stats.py --chunk-size-chars $(CHUNK_SIZE_CHARS) --overlap-chars $(OVERLAP_CHARS)

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__
	find . -type d -name "__pycache__" -prune -exec rm -rf {} \;
