from enum import Enum


class ChatRole(str, Enum):
    USER = "user"
    BOT = "bot"
    SYSTEM = "system"
