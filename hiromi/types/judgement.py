from enum import Enum
from dataclasses import dataclass, field


class EPrediction(int, Enum):
    incorrect = 0  # hallucination
    correct = 1
    error = -1


@dataclass
class Judgement:
    question: str
    llm_answer: str
    prediction: EPrediction

    meta: dict = field(default_factory=dict)
