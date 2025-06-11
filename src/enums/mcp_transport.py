from enum import Enum


class McpTransport(str, Enum):
    STDIO = "stdio"
    SSE = "sse"
    STREAMABLE_HTTP = "streamable_http"
    WEBSOCKET = "websocket"
