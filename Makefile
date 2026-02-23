SHELL := /bin/bash

# List of all packages
PACKAGES := agent-core openai-core agent-server langgraph-core agent-evaluator aws-strands-core mcp-core crewai-core template-generator

.PHONY: all install test clean help $(PACKAGES)

# Default target
help:
	@echo "Available commands:"
	@echo "  make install          - Install/Update all packages in isolated venvs"
	@echo "  make test             - Run tests for all packages"
	@echo "  make <package>        - Install and test a specific package (e.g., make agent-core)"
	@echo "  make install-<pkg>    - Install/Update a specific package"
	@echo "  make test-<pkg>       - Test a specific package"
	@echo "  make clean            - Remove all .venv and cache files"
	@echo ""
	@echo "Packages: $(PACKAGES)"

all: install test

# Install/Update all packages
install: $(addprefix install-,$(PACKAGES))

# Test all packages
test: $(addprefix test-,$(PACKAGES))

# Rule to install and test a specific package (e.g., make agent-core)
$(PACKAGES): %: install-% test-%

# Generic install rule for each package
# This is efficient: it only creates the venv if missing,
# and uv pip install will only install new/changed dependencies.
install-%:
	@echo "----------------------------------------------------------------"
	@echo "Installing/Updating dependencies for $*..."
	@echo "----------------------------------------------------------------"
	cd packages/$* && \
	if [ ! -d ".venv" ]; then uv venv; fi && \
	. .venv/bin/activate && \
	if [ "$*" == "agent-core" ]; then \
		uv pip install -e ".[all]"; \
	else \
		uv pip install -e .; \
	fi && \
	uv pip install pytest pytest-cov pytest-asyncio litellm pytest-mock

# Generic test rule for each package
test-%:
	@echo "----------------------------------------------------------------"
	@echo "Running tests for $*..."
	@echo "----------------------------------------------------------------"
	cd packages/$* && \
	if [ -d ".venv" ]; then \
		. .venv/bin/activate && pytest; \
	else \
		echo "Error: No virtualenv found for $*. Run 'make install-$*' first."; \
		exit 1; \
	fi

clean:
	@echo "Cleaning up all virtual environments and caches..."
	find . -type d -name ".venv" -exec rm -rf {} +
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf .coverage htmlcov
