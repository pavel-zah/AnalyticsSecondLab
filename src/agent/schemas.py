from dataclasses import dataclass

@dataclass
class ResponseFormat:
    """Схема ответа для агента."""

    review_sentiment: str
    review_topic: str