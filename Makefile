PYTHON_VERSION = 3.9.15
PYTHON_PREFIX := $(shell pyenv prefix $(PYTHON_VERSION))
PYTHON_BIN := $(PYTHON_PREFIX)/bin/python

.PHONY: install
install: pyenv_setup poetry_setup
	@VENV_PATH=$$(poetry env info --path); \
	SITE_PACKAGES_DIR="$${VENV_PATH}/lib/python3.9/site-packages"; \
	PROJECT_ROOT=$$(echo $${VENV_PATH} | rev | cut -d'/' -f2- | rev); \
	VENV_NAME=$$(basename `dirname $${VENV_PATH}`); \
	echo $${PROJECT_ROOT} > $${SITE_PACKAGES_DIR}/$${VENV_NAME}.pth

.PHONY: check_pyenv
check_pyenv:
	@if ! command -v pyenv >/dev/null 2>&1; then \
		echo "Error: pyenv is not installed. please visit https://github.com/pyenv/pyenv#installation in details."; \
		exit 1; \
	fi

.PHONY: check_poetry
check_poetry:
	@if ! command -v poetry >/dev/null 2>&1; then \
		echo "Error: poetry is not installed. please visit https://python-poetry.org/docs/#installation in details."; \
		exit 1; \
	fi

.PHONY: pyenv_setup
pyenv_setup:
	pyenv install -s $(PYTHON_VERSION)
	pyenv global $(PYTHON_VERSION)

.PHONY: poetry_setup
poetry_setup:
	@test -n "$(PYTHON_PREFIX)" || (echo "[ERROR] Failed to obtain the prefix for $(PYTHON_VERSION) from pyenv."; exit 1)
	@test -x "$(PYTHON_BIN)"   || (echo "[ERROR] $(PYTHON_BIN) not found (verify that 'pyenv install $(PYTHON_VERSION)' has been run)."; exit 1)
	poetry env use "$(PYTHON_BIN)"
	rm -f poetry.lock
	poetry lock --no-cache
	poetry install

.PHONY: run
run:
	bash bin/demo.sh

.PHONY: run_nohup
run_nohup:
	bash bin/demo.sh -n

.PHONY: cuda_check
cuda_check:
	poetry run python tests/test_cuda.py

.PHONY: freeze
freeze:
	poetry run pip freeze > requirements.txt

.PHONY: latex
latex:
	latexmk -cd docs/reports/memo.tex