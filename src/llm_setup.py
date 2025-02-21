from typing import Literal, Optional
from langchain_openai import ChatOpenAI
from langchain.llms.base import BaseLLM
import logging
import os

class LLMConfig:
    """Venice API configuration"""
    def __init__(self):
        # Venice-specific defaults
        self.model_name = os.getenv('VENICE_MODEL_NAME', 'llama-3.3-70b')
        self.temperature = float(os.getenv('VENICE_TEMPERATURE', '0.5'))
        self.base_url = os.getenv('VENICE_BASE_URL', 'https://api.venice.ai/v1')
        self.api_key = os.getenv('VENICE_API_KEY')

class LLMClient:
    """Simplified Venice API client"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = LLMConfig()
        self.models: dict[str, BaseLLM] = {}

    def init_venice(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> BaseLLM:
        """Initialize Venice API connection"""
        try:
            self.models['venice'] = ChatOpenAI(
                model=model_name or self.config.model_name,
                temperature=temperature or self.config.temperature,
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                **kwargs
            )
            self.logger.info(f"Connected to Venice API: {self.models['venice'].model_name}")
            return self.models['venice']
        except Exception as e:
            self.logger.error(f"Venice connection failed: {e}")
            raise

    def get_model(self, provider: str = "venice") -> BaseLLM:
        """Get initialized model"""
        if model := self.models.get(provider):
            return model
        raise ValueError(f"Model {provider} not initialized")

# Initialize logging
logging.basicConfig(level=logging.INFO)