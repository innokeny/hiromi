from __future__ import annotations

import re
from dataclasses import dataclass

from openai import OpenAI

from hiromi.judge.llm import PromptTemplate
from hiromi.types.judgement import EPrediction, Judgement

from loguru import logger


@dataclass
class DecomposedJudge:
    """Decomposed evaluation strategy.

    Breaks evaluation into three independent sub-tasks:
    1. Identify factual claims in the answer.
    2. Verify those claims against established knowledge.
    3. Detect fabricated entities (invented people, dates, events).

    Final verdict: hallucination if claims are unsupported OR fabrication is detected.
    """

    client: OpenAI
    model: str
    factual_prompt: str | PromptTemplate
    verification_prompt: str | PromptTemplate
    fabrication_prompt: str | PromptTemplate
    temperature: float = 0.1

    def _request(self, prompt: str | PromptTemplate, **kwargs) -> str:
        if isinstance(prompt, PromptTemplate):
            content = prompt.format(**kwargs)
        else:
            content = prompt.format(**kwargs)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _parse_yes_no(response: str) -> bool:
        match = re.search(r"result\s*:\s*(yes|no)", response.lower())
        if match:
            return match.group(1) == "yes"
        lower = response.lower()
        logger.debug(f"No 'Result: yes/no' found, falling back to keyword search: {response[:80]!r}")
        return "yes" in lower and "no" not in lower

    @staticmethod
    def _parse_supported(response: str) -> bool:
        match = re.search(r"result\s*:\s*(supported|unsupported)", response.lower())
        if match:
            return match.group(1) == "supported"
        logger.debug(f"No 'Result: supported/unsupported' found, falling back: {response[:80]!r}")
        return "unsupported" not in response.lower()

    def predict(self, question: str, llm_answer: str, **kwargs) -> Judgement:
        # Step 1: Identify factual claims
        factual_response = self._request(
            self.factual_prompt,
            question=question,
            llm_answer=llm_answer,
        )
        has_claims = self._parse_yes_no(factual_response)

        # Step 2: Verify claims (skip if none found)
        if has_claims:
            verification_response = self._request(
                self.verification_prompt,
                question=question,
                llm_answer=llm_answer,
                factual_claims=factual_response,
            )
            claims_supported = self._parse_supported(verification_response)
        else:
            verification_response = "No factual claims identified — verification skipped."
            claims_supported = True

        # Step 3: Detect fabricated entities
        fabrication_response = self._request(
            self.fabrication_prompt,
            question=question,
            llm_answer=llm_answer,
        )
        has_fabrication = self._parse_yes_no(fabrication_response)

        is_hallucination = (not claims_supported) or has_fabrication
        prediction = EPrediction.incorrect if is_hallucination else EPrediction.correct

        return Judgement(
            question=question,
            llm_answer=llm_answer,
            prediction=prediction,
            meta={
                "model": self.model,
                "temperature": self.temperature,
                "factual_response": factual_response,
                "has_claims": has_claims,
                "verification_response": verification_response,
                "claims_supported": claims_supported,
                "fabrication_response": fabrication_response,
                "has_fabrication": has_fabrication,
                "full-model-response": (
                    f"claims={has_claims}, supported={claims_supported}, fabrication={has_fabrication}"
                ),
            },
        )
