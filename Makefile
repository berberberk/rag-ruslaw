.PHONY: help venv sync lint format test test-network serve ui ingest build-index eval clean

help:
	@echo "Targets:"
	@echo "  make venv          - create virtual environment (uv)"
	@echo "  make sync          - install/sync dependencies (uv sync)"
	@echo "  make lint          - ruff lint"
	@echo "  make lint-fix      - ruff lint and fix"
	@echo "  make format        - ruff format"
	@echo "  make fmt           - lint and format"
	@echo "  make check         - lint and test"
	@echo "  make test          - pytest (no network/slow)"
	@echo "  make test-network  - pytest including network tests"
	@echo "  make serve         - run FastAPI (uvicorn)"
	@echo "  make ui            - run Streamlit UI"
	@echo "  make ingest        - run ingestion script"
	@echo "  make build-index   - build indexes (bm25/faiss)"
	@echo "  make eval          - run RAGAS evaluation"
	@echo "  make clean         - remove caches"

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

fmt:
	format lint-fix

check:
	lint test

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

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__
	find . -type d -name "__pycache__" -prune -exec rm -rf {} \;
