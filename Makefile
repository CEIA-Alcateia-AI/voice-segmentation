.DEFAULT_GOAL := help

# Diretório do pacote fonte
SRC := src/voice_segmentation
TESTS := tests

# Detecta o python do ambiente ativo
PYTHON := python

.PHONY: help install install-dev lint format typecheck test pre-commit clean

help: ## Exibe esta mensagem de ajuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

bootstrap: ## Inicializa pip no venv caso não esteja disponível (rode uma vez após criar o venv)
	$(PYTHON) -m ensurepip --upgrade
	$(PYTHON) -m pip install --upgrade pip

install: ## Instala as dependências de produção
	$(PYTHON) -m pip install -e .

install-dev: ## Instala todas as dependências de desenvolvimento
	$(PYTHON) -m pip install -e ".[dev,test]"
	$(PYTHON) -m pre_commit install

lint: ## Executa o linter (ruff)
	$(PYTHON) -m ruff check $(SRC)

lint-fix: ## Executa o linter e aplica correções automáticas
	$(PYTHON) -m ruff check --fix $(SRC)

format: ## Formata o código (ruff format)
	$(PYTHON) -m ruff format $(SRC) $(TESTS)

format-check: ## Verifica formatação sem alterar arquivos
	$(PYTHON) -m ruff format --check $(SRC) $(TESTS)

typecheck: ## Verifica tipos estáticos com mypy
	$(PYTHON) -m mypy $(SRC)

test: ## Executa a suíte de testes
	$(PYTHON) -m pytest $(TESTS) -v

test-cov: ## Executa testes com relatório de cobertura
	$(PYTHON) -m pytest $(TESTS) -v --cov=$(SRC) --cov-report=term-missing

pre-commit: ## Executa todos os hooks do pre-commit sobre os arquivos rastreados
	$(PYTHON) -m pre_commit run --all-files

clean: ## Remove artefatos de build e cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name ".coverage" -delete
