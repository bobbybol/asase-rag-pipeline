"""
Centralized Configuration
Uses pydantic-settings for validated environment variables.
"""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from functools import lru_cache

# Load .env into os.environ so LangChain/LangSmith SDK can read tracing config
load_dotenv()

class Settings(BaseSettings):

    # LLM Configuration
    google_api_key: str
    primary_model: str = "gemini-2.5-flash-lite"
    fallback_model: str = "gemini-2.5-flash-lite"

    # LangSmith
    langsmith_tracing: bool = True
    langsmith_api_key: str = ""
    langsmith_project: str = "production-api"
    langsmith_endpoint: str = "https://eu.api.smith.langchain.com"

    # Application
    app_env: str = "development"
    log_level: str = "INFO"
    rate_limit: str = "20/minute"
    cache_ttl_seconds: int = 300
    max_retries: int = 3

    # RAG / Vector store
    supabase_database_url: str = ""
    rag_collection_name: str = "minigrid_docs"
    rag_k: int = 5
    rag_relevance_threshold: float = 0.5

    model_config = {"env_file": ".env", "extra": "ignore"}

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

@lru_cache
def get_settings() -> Settings:
    """Cached settings instance - loaded once, reused everywhere."""
    return Settings()