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
	uvx ruff format heinrich_template
	uvx ruff check --silent --exit-zero --no-cache --fix heinrich_template
	uvx ruff check --exit-zero heinrich_template
isort:
	uvx isort heinrich_template tests

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/heinrich_template.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ training_code
	$(MAKE) -C docs clean
	$(MAKE) -C docs html

install: clean ## install the package to the active Python's site-packages
	uv pip install -e ".[dev]"

check:
	pre-commit run --all-files

format:
	make ruff
	make isort

build-lcm:
	rm -rf lcm/build
	mkdir lcm/build
	cd lcm/build && cmake .. && make && sudo make install

update-sdk:
	rm -rf walk-these-ways-go2/go2_gym_deploy/unitree_sdk2_bin/library/unitree_sdk2/build
	cd walk-these-ways-go2/go2_gym_deploy/unitree_sdk2_bin/library/unitree_sdk2 && sudo ./install.sh
	mkdir walk-these-ways-go2/go2_gym_deploy/unitree_sdk2_bin/library/unitree_sdk2/build
	cd walk-these-ways-go2/go2_gym_deploy/unitree_sdk2_bin/library/unitree_sdk2/build && cmake .. && make

build-go2-lcm:
	rm -rf walk-these-ways-go2/go2_gym_deploy/build
	mkdir walk-these-ways-go2/go2_gym_deploy/build
	cd walk-these-ways-go2/go2_gym_deploy/build && cmake .. && make -j

build-backend: build-lcm update-sdk build-go2-lcm

ping:
	ping 192.168.123.18

ai-ping:
	ping heinrich

test-lcm:
	echo "Testing lcm reception. Make sure to shut this down correctly else control will not work."
	cd walk-these-ways-go2/go2_gym_deploy/build && sudo ./lcm_receive

start-lcm:
	echo "Starting lcm on eth0. If this is not the correct interface address for Heinrich, manually excecute:"
	echo "cd walk-these-ways-go2/go2_gym_deploy/build && sudo ./lcm_position_go2 <interface>"
	echo ""
	echo "You'll see the correct interface address when running 'make test-lcm'"
	cd walk-these-ways-go2/go2_gym_deploy/build && sudo ./lcm_position_go2 eth0

deploy-example:
	cd walk-these-ways-go2/go2_gym_deploy/scripts && python deploy_policy.py

ssh:
	ssh -X unitree@192.168.123.18

ai-ssh:
	ssh -X unitree@heinrich