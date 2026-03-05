.DEFAULT_GOAL := help
PYTHON        := uv run
SRC           := src/vas
TESTS         := tests

.PHONY: help
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
.PHONY: install
install: ## Install all dependencies
	uv sync --all-groups

.PHONY: hooks
hooks: ## Install pre-commit hooks
	$(PYTHON) pre-commit install --install-hooks

.PHONY: setup
setup: install hooks ## Full dev environment setup

# ---------------------------------------------------------------------------
# Formatting and Linting
# ---------------------------------------------------------------------------
.PHONY: format
format: ## Auto-format code with ruff
	$(PYTHON) ruff format $(SRC) $(TESTS)
	$(PYTHON) ruff check --fix $(SRC) $(TESTS)

.PHONY: lint
lint: ## Lint code with ruff
	$(PYTHON) ruff format --check $(SRC) $(TESTS)
	$(PYTHON) ruff check $(SRC) $(TESTS)

# ---------------------------------------------------------------------------
# Type Checking
# ---------------------------------------------------------------------------
.PHONY: typecheck
typecheck: ## Run mypy strict type checking
	$(PYTHON) mypy $(SRC)

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
.PHONY: test
test: ## Run tests with pytest + coverage
	$(PYTHON) pytest

.PHONY: test-fast
test-fast: ## Run tests without coverage
	$(PYTHON) pytest --no-cov -x

# ---------------------------------------------------------------------------
# CI
# ---------------------------------------------------------------------------
.PHONY: ci
ci: lint typecheck test ## Run full CI pipeline locally

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
.PHONY: clean
clean: ## Remove build/cache artifacts
	rm -rf .mypy_cache .pytest_cache .ruff_cache .coverage coverage.xml htmlcov dist
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
