from enum import Enum


class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    CONFIRMATION = "confirmation"


class StreamType(str, Enum):
    INIT = "init"
    START_THINKING = "start_thinking"
    THINKING = "thinking"
    END_THINKING = "end_thinking"
    START_MESSAGING = "start_messaging"
    MESSAGING = "messaging"
    END_MESSAGING = "end_messaging"
    CONFIRMATION = "confirmation"
    END_CONFIRMATION = "end_confirmation"
    ERROR = "error"
    CHECKING_TITLE = "checking_title"
    GENERATED_TITLE = "generated_title"


class ApproveType(str, Enum):
    ASKING = "asking"
    ACCEPT = "accept"
    UPDATE = "update"
    FEEDBACK = "feedback"
    CANCEL = "cancel"
