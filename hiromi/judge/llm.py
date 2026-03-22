from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field

from openai import OpenAI

from hiromi.types.judgement import Judgement, EPrediction

from loguru import logger


@dataclass
class PromptTemplate:
    template: str

    @classmethod
    @logger.catch(message="error in reading file for prompt")
    def from_file(cls, file: str | Path) -> PromptTemplate:
        content = Path(file).read_text().strip()
        return cls(content)

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)


@dataclass
class LlmAsAJudge:
    prompt: str | PromptTemplate
    client: OpenAI
    model: str
    temperature: float = 0.1

    incorrect_alias: str = "hallucination"
    correct_alias: str = "correct"

    @logger.catch(message="error in model request")
    def _request(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # pyright: ignore[reportArgumentType]
            temperature=self.temperature,
        )
        content = response.choices[0].message.content
        if content is None:
            raise
        return content

    def parse_response(self, response: str) -> EPrediction:
        content = response.strip().lower()
        if content.find(self.incorrect_alias) != -1:
            return EPrediction.incorrect
        elif content.find(self.correct_alias) != -1:
            return EPrediction.correct

        logger.debug(f"unknown response: {response}")
        return EPrediction.error

    def predict(self, question: str, llm_answer: str, **kwargs) -> Judgement:
        """make a decision for llm_answer if it is correct or not

        :param str question: fact check question
        :param str llm_answer: generated answer by llm
        :param **kwargs: additional fields to add to prompt template
        :return Judgement: judgement for llm_answer
        """


        content = self.prompt.format(question=question, llm_answer=llm_answer, **kwargs)
        response = self._request([{"role": "user", "content": content}])
        prediction = self.parse_response(response)
        return Judgement(
            question,
            llm_answer,
            prediction,
            {
                "model": self.model,
                "temperature": self.temperature,
                "full-model-response": response,
            },
        )
