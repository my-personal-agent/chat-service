# Application metrics
from prometheus_client import Counter, Gauge, Histogram, Info

from config.settings_config import get_settings

api_calls_counter = Counter(
    "chat_api_calls_total", "Total Chat API calls", ["endpoint", "method", "status"]
)
api_duration_histogram = Histogram(
    "chat_api_duration_seconds", "Chat API execution time", ["endpoint", "method"]
)
active_connections = Gauge(
    "chat_api_active_connections", "Number of active connections"
)
server_info = Info("chat_api_server_info", "Server info")

# System metrics
memory_usage = Gauge("chat_api_memory_usage_bytes", "Memory usage in bytes")
cpu_usage = Gauge("chat_api_cpu_usage_percent", "CPU usage percent")

# Set static metadata for server
server_info.info(
    {
        "version": get_settings().project_version,
        "name": get_settings().project_name,
        "framework": "FastAPI",
    }
)
