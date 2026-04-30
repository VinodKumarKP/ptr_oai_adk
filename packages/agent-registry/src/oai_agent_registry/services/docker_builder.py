import os
import time
import logging
import asyncio
import socket
from typing import Optional
from docker.errors import NotFound

logger = logging.getLogger(__name__)

def get_free_port() -> int:
    """Finds an available port on the host machine."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def run_agent_container(docker_client, agent_name: str):
    """Spins up the newly built agent image using a dynamic port."""
    port = get_free_port()
    container_name = f"agent-{agent_name.replace('_', '-')}"
    
    volumes = {
        os.path.abspath("./logs"): {"bind": "/tmp", "mode": "rw"},
        "/Users/vinodkumarkp/.aws": {"bind": "/home/agentuser/.aws", "mode": "ro"}
    }
    
    environment = {
        "AGENT_NAME": agent_name,
        "AWS_REGION": "us-east-1",
        "REDIS_HOST": "agent-valkey",
        "REDIS_PORT": "6379",
        "AGENT_AUTH_ENABLED": "true",
        "LOGGING_DB_HOST": "agent_logs_db",
        "LOGGING_DB_PORT": "5432",
        "LOGGING_DB_USER": "postgres",
        "LOGGING_DB_PASSWORD": "postgres",
        "LOGGING_DB_NAME": "agent_logs",
        "DB_LOGGING_ENABLED": "true",
        "DB_POOL_MAX_SIZE": "2",
        "DB_POOL_TIMEOUT": "60",
        "DB_POOL_MIN_SIZE": "1",
        "AGENT_BASE_URL": "http://192.168.1.114:8081",
        "AGENT_LOCAL_REGISTRY_URL": "http://host.docker.internal:8081"
    }

    healthcheck = {
        "test": ["CMD", "curl", "-f", f"http://localhost:{port}/health"],
        "interval": 30 * 1000000000, # 30s in nanoseconds
        "timeout": 10 * 1000000000,  # 10s
        "retries": 15
    }

    # Clean up existing container if it exists
    try:
        old_container = docker_client.containers.get(container_name)
        logger.info(f"Removing existing container {container_name}")
        old_container.remove(force=True)
    except NotFound:
        pass
    except Exception as e:
        logger.warning(f"Could not remove old container {container_name}: {e}")

    logger.info(f"Starting container {container_name} mapped to host port {port}...")
    try:
        container = docker_client.containers.run(
            image=f"{agent_name}:latest",
            name=container_name,
            ports={f"{port}/tcp": port},
            volumes=volumes,
            environment=environment,
            command=[f"--port", str(port)],
            restart_policy={"Name": "unless-stopped"},
            network="agent-server-network",
            healthcheck=healthcheck,
            detach=True
        )
        logger.info(f"Container {container_name} successfully started with ID {container.id[:12]}")
        return container
    except Exception as e:
        logger.error(f"Failed to start container {container_name}: {e}")
        raise

async def build_agent_image(docker_client, agent_name: str, github_url: str, framework: Optional[str] = None):
    """
    Executes a blocking Docker build process in a background thread to avoid blocking the event loop,
    and then spins up the container.
    """
    if not docker_client:
        logger.error("Cannot build image: Docker client not initialized.")
        return

    if not github_url:
        logger.error(f"Cannot build image for agent '{agent_name}': missing source.")
        return

    os.environ['DOCKER_BUILDKIT'] = '1'

    # Determine the directory containing the Dockerfile relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.abspath(os.path.join(current_dir, '..', 'resources', 'docker'))

    if not os.path.exists(os.path.join(build_dir, 'Dockerfile')):
        logger.error(f"Dockerfile not found in {build_dir}. Cannot build image.")
        return

    framework_lower = (framework or "").lower()
    base_image = "oai-adk-base-image:latest"
    if framework_lower == "crewai":
        base_image = "oai-adk-crewai-base-image:latest"
    elif framework_lower == "langgraph":
        base_image = "oai-adk-langgraph-base-image:latest"
    elif framework_lower == "openai":
        base_image = "oai-adk-openai-base-image:latest"
    elif framework_lower == "strands":
        base_image = "oai-adk-strands-base-image:latest"

    try:
        logger.info(f"Starting background Docker build for agent '{agent_name}' from {github_url} with base {base_image}")
        image, build_logs = await asyncio.to_thread(
            docker_client.images.build,
            path=build_dir,
            dockerfile="Dockerfile",
            tag=f"{agent_name}:latest",
            buildargs={
                "BASE_IMAGE": base_image,
                "GITHUB_URL": github_url,
                "CACHE_BUST": str(int(time.time()))
            },
            rm=True
        )
        for chunk in build_logs:
            if 'stream' in chunk:
                logger.debug(chunk['stream'].strip())
        logger.info(f"Docker build succeeded for agent '{agent_name}'. Tagged as {agent_name}:latest")

        # Start the container
        await asyncio.to_thread(run_agent_container, docker_client, agent_name)

    except Exception as e:
        logger.error(f"Docker build and run pipeline failed for agent '{agent_name}': {e}")
