from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # rag
    FAISS_DIR: str = "faiss"
    DATA_DIR: str = str(Path(__file__).parent.joinpath("data").absolute())
    EMBEDDING_PROVIDER: str = "huggingface"  # یا "openai", "cohere" و ...
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # model
    OPENROUTER_API_KEY: str = (
        # "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"
        # "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"
        "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"
        # "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"
    )
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    # MODEL_NAME: str = "deepseek/deepseek-r1-0528:free"
    # MODEL_NAME: str = "tngtech/deepseek-r1t2-chimera:free"
    MODEL_NAME: str = "deepseek/deepseek-chat-v3.1:free"
    # app
    APP_TITLE: str = "ai chat"
    APP_VERSION: float = 0.1


@lru_cache()
def get_settings():
    return Settings()
