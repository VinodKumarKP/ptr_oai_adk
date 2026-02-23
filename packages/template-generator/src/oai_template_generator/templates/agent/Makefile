CONFIG_DIR := agentic_registry_agents/agents_config
MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
SERVICES := $(shell ls agentic_registry_agents/agents_config/*.yaml | xargs -n1 basename | sed 's/\.yaml//')

define KILL_MCP_SERVER
 @if docker ps --format '{{.Names}}' | grep -w $(1); then \
  echo "Stopping and removing container $(1)"; \
  docker stop $(1) && docker rm $(1); \
 elif docker ps -a --format '{{.Names}}' | grep -w $(1); then \
  echo "Removing stopped container $(1)"; \
  docker rm $(1); \
 else \
  echo "Container $(1) does not exist."; \
 fi
endef

define RUN_MCP_SERVER
 @if docker ps --format '{{.Names}}' | grep -w $(1); then \
  echo "Container $(1) is already running"; \
 else \
  if docker ps -a --format '{{.Names}}' | grep -w $(1); then \
   echo "Removing existing stopped container $(1)"; \
   docker rm $(1); \
  fi; \
  $(MAKE) docker-build; \
  docker run --rm -d -p $(2):$(2) --name $(1) \
  --env MCP_SERVER_NAME=$(1) mcp_server:latest --transport sse; \
 fi
endef

DOCKER_COMPOSE := $(shell if command -v docker-compose >/dev/null 2>&1; then echo "docker-compose"; else echo "docker compose"; fi)

define RUN_DOCKER_COMPOSE
 $(MAKE) generate-compose
 echo "Executing compose up";
 @if [ -z "$(1)" ]; then \
  if [ "$(2)" = "no-cache" ]; then \
   echo "Building all services with no-cache"; \
   $(DOCKER_COMPOSE) build --no-cache; \
  else \
   $(DOCKER_COMPOSE) build; \
  fi; \
  $(DOCKER_COMPOSE) up -d --build --force-recreate; \
 else \
  $(DOCKER_COMPOSE) build $(1) --no-cache; \
  $(DOCKER_COMPOSE) up -d $(1) --build; \
 fi; \
 docker system prune --force;
endef

define DOWN_DOCKER_COMPOSE
 echo "Executing compose down";
 @if [ -z "$(1)" ]; then \
  $(DOCKER_COMPOSE) down; \
 else \
  $(DOCKER_COMPOSE) stop $(1) && $(DOCKER_COMPOSE) rm -f $(1); \
 fi;
endef

define ENSURE_CLONE
	echo "Ensuring configs are present"
endef

ensure-clone:
	$(call ENSURE_CLONE)

install:
	pip install .

uninstall:
	pip uninstall .

docker-build: ensure-clone
	docker build -f Dockerfile . -t agent_server --build-arg GITHUB_TOKEN=$$GITHUB_TOKEN --no-cache

generate-compose: ensure-clone
	uv run docker_compose_generator.py

start-all: ensure-clone
	$(call RUN_DOCKER_COMPOSE)

start-all-no-cache: ensure-clone
	$(call RUN_DOCKER_COMPOSE,,no-cache)

stop-all: ensure-clone
	$(call DOWN_DOCKER_COMPOSE)

# List all discovered services
list-services: ensure-clone
	@echo "Available services:"
	@if [ -d "$(MAKEFILE_DIR)/$(CONFIG_DIR)" ]; then \
		for yaml_file in $(MAKEFILE_DIR)/$(CONFIG_DIR)/*.yaml; do \
			if [ -f "$$yaml_file" ]; then \
				service=$$(basename "$$yaml_file" .yaml); \
				echo "  - $$service"; \
			fi; \
		done; \
	else \
		echo "  No services found. Run 'make ensure-clone' first."; \
	fi


# Generic service handler that works for any service
start-%: ensure-clone
	@test -f "$(MAKEFILE_DIR)/$(CONFIG_DIR)/*.yaml" || (echo "Service '$*' not found. Run 'make list-services' to see available services." && exit 1)
	@echo "Starting service: $*"
	$(call RUN_DOCKER_COMPOSE,$*)

start_no_cache-%: ensure-clone
	@test -f "$(MAKEFILE_DIR)/$(CONFIG_DIR)/*.yaml" || (echo "Service '$*' not found. Run 'make list-services' to see available services." && exit 1)
	@echo "Starting service: $*"
	$(call RUN_DOCKER_COMPOSE,$*,no-cache)

stop-%: ensure-clone
	@test -f "$(MAKEFILE_DIR)/$(CONFIG_DIR)/*.yaml" || (echo "Service '$*' not found. Run 'make list-services' to see available services." && exit 1)
	@echo "Stopping service: $*"
	$(call DOWN_DOCKER_COMPOSE,$*)

restart-%: ensure-clone
	@$(MAKE) stop-$*
	@$(MAKE) start-$*
	@echo "Restarted service: $*"


help:
	@echo "Available commands:"
	@echo "  install                Install the Python package"
	@echo "  uninstall              Uninstall the Python package"
	@echo "  docker-build           Build the MCP server Docker image"
	@echo "  start-all              Start all services using docker compose"
	@echo "  start-all-no-cache     Start all services with no-cache build"
	@echo "  stop-all               Stop all services using docker compose"
	@echo "  list-services          List all discovered services"
	@echo "  start-<service>        Start a specific service (replace <service> with name)"
	@echo "  stop-<service>         Stop a specific service (replace <service> with name)"
	@echo "  restart-<service>      Restart a specific service (replace <service> with name)"
	@echo "  generate-compose       Generate docker compose"
	$(MAKE) list-services
