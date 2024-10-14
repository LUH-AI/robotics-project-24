.PHONY: clean clean-build clean-pyc clean-test coverage dist docs help install check format
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test clean-docs ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

clean-docs: ## remove docs artifacts
	rm -fr docs/build
	rm -fr docs/api
	rm -fr docs/examples

ruff: ## run ruff as a formatter
	uvx ruff format deployment_code training_code
	uvx ruff check --silent --exit-zero --no-cache --fix deployment_code training_code
	uvx ruff check --exit-zero deployment_code training_code
isort:
	uvx isort deployment_code training_code

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/heinrich_template.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ training_code
	$(MAKE) -C docs clean
	$(MAKE) -C docs html

install-isaacsim:
	pip install --upgrade pip
	pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
	pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit-sdk isaacsim-extscache-kit isaacsim-app --extra-index-url https://pypi.nvidia.com

install-isaaclab:
	git submodule update --init --recursive
	cd training_code/IsaacLab && ./isaaclab.sh --install skrl

install-train: install-isaacsim install-isaaclab 

install-deploy:
	git submodule update --init --recursive
	pip install pygame
	pip install spatialmath-python
	pip install onnxruntime
	cd deployment_code/unitree_sdk2/ && mkdir build && cd build && cmake .. && make install

install: clean install-train install-deploy

check:
	pre-commit run --all-files

format:
	make ruff
	make isort

ping:
	ping 192.168.123.18

ai-ping:
	ping heinrich

ssh:
	ssh -X unitree@192.168.123.18

ai-ssh:
	ssh -X unitree@heinrich