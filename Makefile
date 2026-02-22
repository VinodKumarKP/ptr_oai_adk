SHELL := /bin/bash

PACKAGES := agent-core crewai-core openai-core agent-server langgraph-core agent-evaluator aws-strands-core

.PHONY: all install test clean $(PACKAGES)

all: install test

install: $(addprefix install-,$(PACKAGES))

test: $(addprefix test-,$(PACKAGES))

# Generic install rule for each package
install-%:
	@echo "Installing dependencies for $*..."
	cd packages/$* && \
	uv venv && \
	. .venv/bin/activate && \
	if [ "$*" == "agent-core" ]; then \
		uv pip install -e ".[all]"; \
	else \
		uv pip install -e .; \
	fi && \
	uv pip install pytest pytest-cov pytest-asyncio

# Generic test rule for each package
test-%:
	@echo "Running tests for $*..."
	cd packages/$* && \
	if [ -d ".venv" ]; then \
		. .venv/bin/activate && pytest; \
	else \
		echo "No virtualenv found for $*. Run 'make install-$*' first."; \
		exit 1; \
	fi

clean:
	@echo "Cleaning up..."
	find . -type d -name ".venv" -exec rm -rf {} +
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf .coverage htmlcov
