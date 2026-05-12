from functools import lru_cache
from langchain_openrouter import ChatOpenRouter
from src.core.config import settings


class LLMFactory:
    """Фабрика для создания LLM на основе конфигурации"""

    @staticmethod
    def create_openrouter_llm() -> ChatOpenRouter:
        """Создаёт OpenRouter LLM"""


        return ChatOpenRouter(
            model=settings.openrouter_model,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            app_title=None,
            streaming=False
        )

llm: ChatOpenRouter | None = None

@lru_cache(maxsize=1)
def get_llm() -> ChatOpenRouter:
    """
    Возвращает singleton инстанс LLM.
    Кешируется для переиспользования.

    Returns:
        Настроенный LLM
    """

    global llm

    if llm is None:
        llm = LLMFactory.create_openrouter_llm()
    return llm

