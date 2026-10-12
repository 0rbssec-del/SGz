PYTHON := python
PIP := pip
PYTEST := pytest
PROJECT := critical-asymmetry-llm

.DEFAULT_GOAL := help

help:
	@echo "install, test, lint, format, clean, run-bench, build, zip, docker-build"

install:
	$(PIP) install -r requirements.txt && $(PIP) install -e .

test:
	$(PYTEST) tests/ -v --cov=src

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/

run-bench:
	$(PYTHON) experiments/run_benchmark.py

zip:
	$(PYTHON) build_zip.py

docker-build:
	docker build -t $(PROJECT):latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down
