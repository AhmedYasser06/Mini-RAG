from enum import Enum

class LLMEnums(Enum):
    """
    Enum class for LLMs (Large Language Models).
    """
    OPENAI = "OPENAI"
    COHERE = "COHERE"
    GEMINI = "GEMINI"
    
class OpenAIEnums(Enum):
    """
    Enum class for OpenAI specific models.
    """
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    
class CoHereEnums(Enum):
    """
    Enum class for Cohere specific models.
    """
    SYSTEM = "SYSTEM"
    USER = "USER"
    ASSISTANT = "CHATBOT"
    
    DOCUMENT = "search_document"
    QUERY = "search_query"
    
class GeminiEnums(Enum):
    SYSTEM = "user"
    USER = "user"
    ASSISTANT = "model"

class DocumentTypeEnums(Enum):
    DOCUMENT = "document"
    QUERY = "query"
