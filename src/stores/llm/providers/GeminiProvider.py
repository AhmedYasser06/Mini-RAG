from google import genai
from google.genai import types # import when using mini-rag-py310
import logging
import time

from ..LLMInterface import LLMInterface
from ..LLMEnums import GeminiEnums, DocumentTypeEnums


class GeminiProvider(LLMInterface):

    def __init__(self, api_key: str,
                 default_input_max_characters: int = 1024,
                 default_generation_max_output_tokens: int = 1024,
                 default_generation_temperature: float = 0.1):

        self.api_key = api_key
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        self.client = genai.Client(api_key=self.api_key)
        
        self.enums = GeminiEnums
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": GeminiEnums.ASSISTANT.value if role == "assistant" else GeminiEnums.USER.value,
            "parts": [self.process_text(prompt)],
        }

    def generate_text(self, prompt: str, chat_history: list = [],
                       max_output_tokens: int = None,
                       temperature: float = None):

        if not self.client:
            self.logger.error("Gemini client was not set")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model for Gemini was not set")
            return None

        max_output_tokens = max_output_tokens or self.default_generation_max_output_tokens
        temperature = temperature if temperature is not None else self.default_generation_temperature

        # Gemini's `contents` can just be prior turns + the new prompt as plain strings/dicts
        contents = self.process_text(prompt)

        try:
            response = self.client.models.generate_content(
                model=self.generation_model_id,
                contents=contents,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                ),
            )
        except Exception as e:
            self.logger.error(f"Error while generating text with Gemini: {e}")
            return None

        if not response or not response.text:
            self.logger.error("Error while generating text with Gemini")
            return None

        return response.text

    def embed_text(self, text: str, document_type: str = None):

        if not self.client:
            self.logger.error("Gemini client was not set")
            return None

        if not self.embedding_model_id:
            self.logger.error("Embedding model for Gemini was not set")
            return None

        task_type = "RETRIEVAL_DOCUMENT"
        if document_type == DocumentTypeEnums.QUERY.value:
            task_type = "RETRIEVAL_QUERY"

        is_batch = isinstance(text, list)
        texts = text if is_batch else [text]
        texts = [self.process_text(t) for t in texts]
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = self.client.models.embed_content(
                    model=self.embedding_model_id,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self.embedding_size,
                    ),
                )
                break
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) and attempt < max_retries - 1:
                    wait_time = 20 * (attempt + 1)  # 20s, then 40s
                    self.logger.warning(f"Rate limited, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                self.logger.error(f"Error while embedding text with Gemini: {e}")
                return None
        else:
            return None

        if not result or not result.embeddings:
            self.logger.error("No embeddings returned from Gemini")
            return None

        vectors = [e.values for e in result.embeddings]
        return vectors if is_batch else vectors[0]