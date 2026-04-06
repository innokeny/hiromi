from __future__ import annotations

import re
from dataclasses import dataclass

from hiromi.judge.llm import LlmAsAJudge
from hiromi.types.judgement import EPrediction

from loguru import logger


@dataclass
class ChainOfThoughtJudge(LlmAsAJudge):
    """LLM-as-a-Judge with Chain-of-Thought reasoning.

    The model reasons step by step before delivering a verdict.
    The verdict is parsed from 'Verdict: correct' or 'Verdict: hallucination'
    anywhere in the response, falling back to base class parsing if not found.
    """

    def parse_response(self, response: str) -> EPrediction:
        match = re.search(r"verdict\s*:\s*(correct|hallucination)", response.lower())
        if match:
            verdict = match.group(1)
            if verdict == self.incorrect_alias:
                return EPrediction.incorrect
            elif verdict == self.correct_alias:
                return EPrediction.correct

        logger.debug(f"No 'Verdict:' found in CoT response, falling back to base parser: {response[:120]!r}")
        return super().parse_response(response)
