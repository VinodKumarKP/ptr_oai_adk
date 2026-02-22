SHELL := /bin/bash

.PHONY: install install-all test clean

install:
	pip install uv
	uv venv
	. .venv/bin/activate && uv pip install -e "packages/agent-core[all]"
	. .venv/bin/activate && uv pip install pytest pytest-cov pytest-asyncio

install-all:
	pip install uv
	uv venv
	. .venv/bin/activate && uv pip install -e "packages/agent-core[all]"
	. .venv/bin/activate && uv pip install pytest pytest-cov pytest-asyncio litellm

test:
	cd packages/agent-core && pytest
	cd packages/agent-server && pytest
	cd packages/agent-evaluator && pytest
	cd packages/openai-core && pytest
	cd packages/aws-strands-core && pytest
	cd packages/langgraph-core && pytest
	cd packages/crewai-core && pytest

clean:
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
