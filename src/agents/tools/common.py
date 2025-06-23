from datetime import datetime, timezone, tzinfo
from zoneinfo import ZoneInfo

from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations."""
    try:
        allowed_names = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "pi": 3.14159265359,
            "e": 2.71828182846,
            "sqrt": lambda x: x**0.5,
            "sin": __import__("math").sin,
            "cos": __import__("math").cos,
            "tan": __import__("math").tan,
            "log": __import__("math").log,
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Calculation result: {expression} = {result}"
    except Exception as e:
        return f"Calculation failed: {str(e)}"


@tool
def get_current_time(timezone_name: str = "UTC") -> str:
    """Get current date/time in specific IANA timezone."""
    try:
        tz: tzinfo = ZoneInfo(timezone_name)
    except Exception:
        tz = timezone.utc  # timezone.utc is also a tzinfo
        timezone_name = "UTC"

    now = datetime.now(tz)
    return f"Current time ({timezone_name}): {now.strftime('%Y-%m-%d %H:%M:%S %Z')}"
