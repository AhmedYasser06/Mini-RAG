from enum import Enum

class LLMEnums(Enum):
    """
    Enum class for LLMs (Large Language Models).
    """
    OPENAI = "OPENAI"
    COHERE = "COHERE"
    
class OpenAIEnums(Enum):
    """
    Enum class for OpenAI specific models.
    """
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    
    