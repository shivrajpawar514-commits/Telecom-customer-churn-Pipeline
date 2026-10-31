.PHONY: help setup clean train test drift serve reproduce all

PYTHON ?= $(shell which python3.11 || which python3)
VENV ?= .venv
BIN = $(VENV)/bin

help:
	@echo "Available commands:"
	@echo "  make setup      - Create virtual environment and install dependencies"
	@echo "  make train      - Execute the end-to-end training pipeline"
	@echo "  make test       - Run full automated test suite with pytest"
	@echo "  make drift      - Run Evidently drift detection and generate alert manifest"
	@echo "  make serve      - Launch FastAPI serving application with uvicorn"
	@echo "  make reproduce  - One-command full reproduction: train, test, drift, and verify"
	@echo "  make clean      - Clean temporary caches and compilation files"

setup:
	@echo "Setting up clean virtual environment in $(VENV)..."
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	@echo "Setup complete. Activate with: source $(VENV)/bin/activate"

clean:
	@echo "Cleaning caches..."
	rm -rf .pytest_cache __pycache__ src/__pycache__ src/*/__pycache__ tests/__pycache__ monitoring/__pycache__

train:
	@echo "Executing complete training pipeline..."
	@if [ -d "$(VENV)" ]; then \
		$(BIN)/python -m src.pipeline.training_pipeline; \
	else \
		$(PYTHON) -m src.pipeline.training_pipeline; \
	fi

test:
	@echo "Running test suite..."
	@if [ -d "$(VENV)" ]; then \
		$(BIN)/pytest -v tests/; \
	else \
		pytest -v tests/; \
	fi

drift:
	@echo "Running drift detection audit..."
	@if [ -d "$(VENV)" ]; then \
		$(BIN)/python monitoring/monitor_drift.py; \
	else \
		$(PYTHON) -m monitoring.monitor_drift; \
	fi

serve:
	@echo "Starting FastAPI service on http://127.0.0.1:8000..."
	@if [ -d "$(VENV)" ]; then \
		$(BIN)/uvicorn app:app --reload --port 8000; \
	else \
		uvicorn app:app --reload --port 8000; \
	fi

reproduce:
	@echo "============================================================"
	@echo " ONE-COMMAND REPRODUCIBILITY EXECUTION"
	@echo "============================================================"
	@chmod +x ./reproduce.sh
	@./reproduce.sh

all: reproduce
