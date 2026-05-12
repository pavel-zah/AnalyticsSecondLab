from functools import lru_cache
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langgraph.graph.state import CompiledStateGraph
from src.agent.schemas import ResponseFormat
from src.agent.prompts import SYSTEM_PROMPT
from src.agent.llm import get_llm

_agent: CompiledStateGraph | None = None

@lru_cache(maxsize=1)
def build_agent() -> CompiledStateGraph:
    """Создаёт агента (singleton)"""
    global _agent
    if _agent is None:
        _agent = create_agent(
            model=get_llm(),
            system_prompt=SYSTEM_PROMPT,
            response_format=ToolStrategy(ResponseFormat)
        )

    return _agent

