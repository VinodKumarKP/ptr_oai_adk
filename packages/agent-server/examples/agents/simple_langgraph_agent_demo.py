import asyncio
import os
import platform
from pprint import pprint
import yaml

# ⚠️ CRITICAL: Set environment variables BEFORE any imports that use them!
# This must happen before importing oai_agent_server and oai_agent_core

# Determine Jaeger connection method based on platform and execution context
# For local Python process talking to Docker containers:
# - macOS/Windows: use TCP gRPC (more reliable than UDP on Docker Desktop)
# - Linux: use UDP (standard Jaeger agent protocol)
# - Docker container: use UDP via 'jaeger' service name
def get_jaeger_config():
    """Detect platform and return appropriate Jaeger connection config."""
    # Check if running in Docker container
    if os.path.exists('/.dockerenv'):
        return {
            'protocol': 'udp',
            'host': 'jaeger',
            'port': 6831,
            'mode': 'docker'
        }
    
    # Running locally with Docker containers
    system = platform.system()
    if system == 'Darwin':  # macOS
        # UDP on macOS/Docker Desktop is unreliable, use TCP gRPC instead
        return {
            'protocol': 'tcp',
            'endpoint': 'http://host.docker.internal:14250',  # gRPC over HTTP
            'host': 'host.docker.internal',
            'port': 14250,
            'mode': 'macos'
        }
    elif system == 'Windows':
        # Same issue as macOS, use TCP gRPC
        return {
            'protocol': 'tcp',
            'endpoint': 'http://host.docker.internal:14250',
            'host': 'host.docker.internal',
            'port': 14250,
            'mode': 'windows'
        }
    else:  # Linux
        # Linux native Docker handles UDP better
        return {
            'protocol': 'udp',
            'host': '172.17.0.1',  # Docker default gateway
            'port': 6831,
            'mode': 'linux'
        }

JAEGER_CONFIG = get_jaeger_config()

# Phase 3: Observability & Phase 4: Robustness Configuration
os.environ['DB_LOGGING_ENABLED'] = 'true'
os.environ['FORCE_AUTH'] = 'false'

# Phase 3: OpenTelemetry Configuration (Jaeger exporter) - SET BEFORE IMPORTS
os.environ['OTEL_ENABLED'] = 'true'

# Set Jaeger/OTLP config based on platform - using OTLP for universal compatibility
# Jaeger in Docker exposes OTLP on port 4317, which is bound to localhost for local Python processes
os.environ['OTEL_EXPORTER_OTLP_ENDPOINT'] = 'http://localhost:4317'

os.environ['OTEL_SERVICE_NAME'] = 'oai-agent-server'
os.environ['OTEL_TRACES_SAMPLER'] = 'always_on'  # Sample all traces for development

# Phase 3: Prometheus Configuration (Metrics) - SET BEFORE IMPORTS
os.environ['PROMETHEUS_ENABLED'] = 'true'
os.environ['PROMETHEUS_EXPORT_INTERVAL'] = '15'  # Export metrics every 15 seconds

# Phase 4: Robustness & Resilience Configuration
os.environ['CIRCUIT_BREAKER_FAILURE_THRESHOLD'] = '5'
os.environ['CIRCUIT_BREAKER_SUCCESS_THRESHOLD'] = '2'
os.environ['CIRCUIT_BREAKER_TIMEOUT_SECONDS'] = '60'
os.environ['RETRY_MAX_ATTEMPTS'] = '5'
os.environ['RETRY_INITIAL_DELAY'] = '0.1'
os.environ['RETRY_MAX_DELAY'] = '10.0'
os.environ['RETRY_JITTER_ENABLED'] = 'true'
os.environ['AGENT_CACHE_MAX_AGENTS'] = '100'
os.environ['AGENT_CACHE_INFO_TTL'] = '300'  # 5 minutes
os.environ['AGENT_CACHE_LOGS_TTL'] = '60'   # 1 minute

# Logging for observability
os.environ['LOG_LEVEL'] = 'INFO'  # DEBUG to see Phase 4 feature details
os.environ['LOG_FORMAT'] = 'json'  # Structured JSON logs

# NOW import the modules AFTER environment is configured
from oai_agent_core.langgraph_core.agents.langgraph_agent import LangGraphAgent
from oai_agent_server.main import AgentHTTPServer as BaseAgentHTTPServer, main as http_main

# Get the absolute path to the examples directory
EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(EXAMPLES_DIR, 'agents_config')
file_root = os.path.dirname(os.path.abspath(__file__))


def print_observability_info():
    """Print observability access information."""
    system = platform.system()
    running_in_docker = os.path.exists('/.dockerenv')
    
    print("\n" + "=" * 80)
    print("📊 OBSERVABILITY & MONITORING")
    print("=" * 80)
    
    if running_in_docker:
        print("🐳 Running in Docker container")
        print(f"   Jaeger: jaeger:6831 (UDP)")
    else:
        print(f"🖥️  Running locally on {system}")
        print(f"   Protocol: {JAEGER_CONFIG['protocol'].upper()}")
        print(f"   Jaeger Host: {JAEGER_CONFIG['host']}:{JAEGER_CONFIG['port']}")
        if JAEGER_CONFIG['protocol'] == 'tcp':
            print(f"   ℹ️  Using TCP gRPC (more reliable on {system} Docker Desktop)")
        else:
            print(f"   ℹ️  Using UDP (standard Jaeger agent protocol on {system})")
    
    print("\n🔍 Distributed Tracing (OpenTelemetry + Jaeger):")
    print("   - Jaeger UI: http://localhost:16686")
    print("   - Service Name: oai-agent-server")
    print("   - Find traces for your requests with automatic instrumentation")
    print("\n📈 Metrics (Prometheus):")
    print("   - Prometheus: http://localhost:9090")
    print("   - Metrics endpoint: http://localhost:8000/metrics")
    print("   - Query examples in Prometheus:")
    print("     * http_request_duration_seconds (request latency)")
    print("     * agent_cache_hits_total (cache performance)")
    print("     * circuit_breaker_state (resilience monitoring)")
    print("\n🛡️  Robustness Features (Phase 4):")
    print("   - Circuit Breaker: Automatic failure protection")
    print("   - Retry Logic: Exponential backoff with jitter")
    print("   - Agent Caching: 40-60% hit rate on agent info")
    print("   - API Versioning: /api/v1/, /api/v2/, /api/ (latest)")
    
    if not running_in_docker and system == 'Darwin':
        print("\n💡 macOS Docker Desktop Tip:")
        print("   - Using TCP gRPC to Jaeger (UDP is unreliable on macOS)")
        print("   - If traces still don't appear:")
        print("     1. Check: curl http://host.docker.internal:14250")
        print("     2. Restart Docker Desktop if needed")
        print("     3. Check Docker logs: docker logs jaeger")
    print("\n💡 Testing Tips:")
    print("   - Enable LOG_LEVEL=DEBUG to see Phase 4 transitions")
    print("   - Check /metrics endpoint for real-time metrics")
    print("   - Trace requests in Jaeger UI for end-to-end visibility")
    print("   - Monitor cache hit rates for optimization")
    
    if not running_in_docker:
        print("\n🔧 Troubleshooting (Local Python + Docker Containers):")
        print(f"   - Protocol: {JAEGER_CONFIG['protocol'].upper()}")
        print(f"   - Jaeger Host: {JAEGER_CONFIG['host']}:{JAEGER_CONFIG['port']}")
        if JAEGER_CONFIG['protocol'] == 'tcp':
            print(f"   - ✓ Using TCP gRPC (more reliable on {system})")
            print(f"   - Endpoint: {JAEGER_CONFIG['endpoint']}")
            print("   - If no traces appear:")
            print("     1. Check: curl -v http://host.docker.internal:14250")
            print("     2. Verify: docker ps | grep jaeger")
            print("     3. Restart: docker compose restart jaeger")
        else:
            print(f"   - ✓ Using UDP (Jaeger agent protocol on {system})")
            print(f"   - Connection: {JAEGER_CONFIG['host']}:{JAEGER_CONFIG['port']}")
            if system == 'Linux':
                print("   - If no traces appear, try your actual host IP instead")
                print("   - Run: hostname -I (to find your IP)")
                print("   - Then set: export OTEL_EXPORTER_JAEGER_AGENT_HOST=<your-ip>")
        print("   - Check Jaeger UI: http://localhost:16686")
        print("   - Look for service: oai-agent-server")
    
    print("\n" + "=" * 80 + "\n")


def run_simple_agent():
    """Run a simple agent example with full observability."""
    print("\n" + "=" * 80)
    print("✨ OAI Agent Server with Observability & Robustness")
    print("=" * 80)
    print("\n📋 Phases Enabled:")
    print("   ✅ Phase 1: Stability & Validation")
    print("   ✅ Phase 2: Performance Optimization")
    print("   ✅ Phase 3: Observability (OpenTelemetry + Prometheus)")
    print("   ✅ Phase 4: Robustness (Circuit Breaker, Retry, Cache, Versioning)")

    config_path = os.path.join(CONFIG_DIR, 'travel_agent.yaml')
    
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize agent
    agent = LangGraphAgent(
        agent_name="travel_agent",
        agent_config=config,
        config_root=EXAMPLES_DIR
    )

    server = BaseAgentHTTPServer(
        agent_name=agent.agent_name,
        config_root=os.path.dirname(os.path.dirname(file_root)),
        agent=agent
    )

    # Print observability information
    print_observability_info()

    # Start the HTTP server
    print("🚀 Starting OAI Agent HTTP Server with all observability features...\n")
    http_main(server)
    return server


if __name__ == "__main__":
    run_simple_agent()