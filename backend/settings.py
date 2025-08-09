from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # rag
    FAISS_DIR: str = "faiss"
    DATA_DIR: str = str(Path(__file__).parent.joinpath("data").absolute())
    # model
    OPENROUTER_API_KEY: str = (
        "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"
    )
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    MODEL_NAME: str = "deepseek/deepseek-chat-v3-0324:free"
    # app
    APP_TITLE: str = "ai chat"
    APP_VERSION: float = 0.1


@lru_cache()
def get_settings():
    return Settings()
