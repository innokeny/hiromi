from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from hiromi.judge.llm import LlmAsAJudge
from hiromi.types.judgement import EPrediction, Judgement

from loguru import logger


@dataclass
class SelfConsistencyJudge:
    """Self-consistency wrapper over LlmAsAJudge.

    Makes N independent calls with higher temperature and returns the majority vote.
    All N responses and vote distribution are stored in meta.
    """

    base_judge: LlmAsAJudge
    n_samples: int = 5
    temperature: float = 0.5

    def predict(self, question: str, llm_answer: str, **kwargs) -> Judgement:
        original_temp = self.base_judge.temperature
        self.base_judge.temperature = self.temperature

        predictions: list[EPrediction] = []
        responses: list[str] = []

        try:
            with ThreadPoolExecutor(max_workers=self.n_samples) as executor:
                futures = [
                    executor.submit(self.base_judge.predict, question=question, llm_answer=llm_answer, **kwargs)
                    for _ in range(self.n_samples)
                ]
                for future in as_completed(futures):
                    judgement = future.result()
                    predictions.append(judgement.prediction)
                    responses.append(judgement.meta.get("full-model-response", ""))
        finally:
            self.base_judge.temperature = original_temp

        valid_preds = [p for p in predictions if p != EPrediction.error]

        if not valid_preds:
            majority = EPrediction.error
            logger.warning(f"All {self.n_samples} samples returned error for question: {question[:80]!r}")
        else:
            majority = Counter(valid_preds).most_common(1)[0][0]

        vote_distribution = {int(k): v for k, v in Counter(predictions).items()}

        return Judgement(
            question=question,
            llm_answer=llm_answer,
            prediction=majority,
            meta={
                "model": self.base_judge.model,
                "temperature": self.temperature,
                "n_samples": self.n_samples,
                "all_responses": responses,
                "vote_distribution": vote_distribution,
                "full-model-response": f"votes={vote_distribution}, majority={int(majority)}",
            },
        )
