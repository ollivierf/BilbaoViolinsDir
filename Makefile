.PHONY: install run test sync-results sync-data clean help

PYTHON := python3
VENV := venv

# Server configuration - override with environment variables
SERVER_USER ?= ollivief
SERVER_HOST ?= mesu
SERVER_PROJECT_PATH ?= ~/Projets/BilbaoViolinsDir

help:
	@echo "Local commands:"
	@echo "  make install        Create venv and install dependencies"
	@echo "  make run            Run main script"
	@echo "  make test           Run tests"
	@echo "  make clean          Remove generated files"
	@echo ""
	@echo "Sync commands (set SERVER_USER, SERVER_HOST first):"
	@echo "  make sync-results   Download results from server"
	@echo "  make sync-data      Upload data to server"

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

run:
	$(VENV)/bin/python scripts/run_computation.py

test:
	$(VENV)/bin/python -m pytest tests/

sync-results:
	rsync -avz --progress $(SERVER_USER)@$(SERVER_HOST):$(SERVER_PROJECT_PATH)/results/ ./results/

sync-data:
	rsync -avz --progress ./data/ $(SERVER_USER)@$(SERVER_HOST):$(SERVER_PROJECT_PATH)/data/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
