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
    
class CoHereEnums(Enum):
    """
    Enum class for Cohere specific models.
    """
    SYSTEM = "SYSTEM"
    USER = "USER"
    ASSISTANT = "CHATBOT"
    
    DOCUMENT = "search_document"
    QUERY = "search_query"

class DocumentTypeEnums(Enum):
    DOCUMENT = "document"
    QUERY = "query"
