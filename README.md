# **Hiromi**: An LLM-as-a-Judge Approach to Hallucination Detection

**Hiromi** employs a language model as a judge for binary classification of LLM responses into two categories: **correct** and **hallucination**. We define a hallucination as any statement that is unsupported by established facts, contradicts common knowledge, or contains fabricated information.

The work is organized into two parts:

- **Part 1 (black-box):** YandexGPT serves as the judge via API. The judge reads a (question, answer) pair and produces a verdict based on its own parametric knowledge.
- **Part 2 (white-box):** the judge is constructed from the open-weight Gemma 2 2B model [^gemma]. A probing classifier and a sparse autoencoder ensemble inspect the model's **internal activations** on the (question, answer) pair to predict the label.

# Part 1. LLM-as-a-Judge with YandexGPT

## Judge Model

The judge is **YandexGPT Lite** (`yandexgpt-lite/latest`), accessed through the Yandex Cloud API with an OpenAI-compatible interface. We use a low decoding temperature (`temperature=0.1`) to reduce verdict variance.

## Dataset

We adopt the TruthfulQA benchmark [^truthfulqa] — 790 questions specifically designed to probe a model's tendency to generate false but plausible-sounding answers. The questions span 38 categories, including Misconceptions, Conspiracies, Superstitions, Finance, Law, and Politics. Each question is accompanied by reference correct and incorrect answers.

For evaluation, we construct (question, answer) pairs with known labels (correct/hallucination), yielding approximately 7,600 test instances.

## Methods and Results

### Baseline strategies (full dataset, ~7,600 examples)

We evaluated three baseline prompting strategies:

| Method | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| Zero-shot | **74.55%** | 0.746 | 0.742 | 0.743 |
| Few-shot (6 examples) | **72.70%** | 0.727 | 0.724 | 0.724 |
| Reference-based | **89.32%** | 0.910 | 0.888 | 0.891 |

**Zero-shot.** The judge evaluates the answer relying solely on its parametric knowledge.

**Few-shot.** Six labeled question-answer-verdict examples are added to the prompt. Notably, few-shot underperforms zero-shot, which we attribute to model-specific behavior and example formatting.

**Reference-based.** The judge is provided with reference correct and incorrect answers from the dataset. This setting achieves the highest accuracy (**89.32%**), substantially exceeding both reference-free baselines.

---

### Advanced strategies and prompt-language comparison (~1,000 examples)

We further evaluated advanced prompting strategies. Each method was executed twice — with an **English (EN)** prompt and a **Russian (RU)** prompt — to assess the impact of prompt language on judge quality.

Sample: stratified subset of 1,000 instances (500 correct + 500 hallucination, `random_state=42`).

| Method | EN Accuracy | RU Accuracy | Δ (RU − EN) | n_errors EN | n_errors RU |
|---|---|---|---|---|---|
| Self-Consistency | **75.77%** | 74.67% | −1.10% | 96 | 88 |
| CoT | 74.72% | 74.12% | −0.60% | 98 | 88 |
| Zero-shot | 73.89% | **74.92%** | +1.03% | 77 | 71 |
| Few-shot (8 examples) | 69.06% | 71.84% | +2.78% | 66 | 66 |
| Few-shot (6 examples) | 69.83% | 71.17% | +1.34% | 72 | 67 |
| Few-shot (2 examples) | 69.13% | 70.43% | +1.30% | 67 | 60 |
| Few-shot (4 examples) | 67.73% | 68.30% | +0.57% | 61 | 60 |
| Decomposed | 69.20% | 63.60% | −5.60% | 0 | 0 |

**Self-Consistency** [^selfconsistency] yields the strongest result among reference-free strategies: 5 independent CoT generations with `temperature=0.5`, with the final verdict determined by majority voting. The method incurs roughly 5× the API cost.

**Chain-of-Thought (CoT)** [^cot] elicits step-by-step reasoning prior to the verdict. It is comparable to zero-shot in accuracy while producing interpretable reasoning traces.

**Few-shot.** Fixed question-answer-verdict exemplars are included in the prompt. Increasing the number of exemplars (2→8) yields only marginal accuracy gains; the optimum depends on the prompt language. Under RU prompts, few-shot consistently outperforms its EN counterpart.

**Decomposed.** The verification task is decomposed into three sequential sub-tasks: identifying factual claims, verifying them, and detecting fabricated entities. This strategy yields the lowest accuracy: errors compound across steps, degrading the final verdict.

#### Effect of prompt language

- For **few-shot** strategies, RU prompts consistently outperform EN (+0.6–2.8%), likely because YandexGPT is better tuned to Russian-language instructions.
- For **zero-shot**, RU is also marginally superior (+1.0%).
- For **CoT** and **Self-Consistency**, EN prompts are slightly preferable (−0.6% and −1.1%).
- For **Decomposed**, the gap is largest: RU is substantially worse (−5.6%) — multi-step pipelines are more sensitive to phrasing precision.

`n_errors` denotes the number of instances on which the judge refused to respond ("I cannot discuss this topic"). Refusals concentrate in the Sociology, Law, and Conspiracies categories, triggered by YandexGPT's content filter, and are excluded from the metrics.

## Setup (Part 1)

Required environment variables:

- `YANDEX_CLOUD_API_KEY` — Yandex Cloud API key
- `YANDEX_CLOUD_FOLDER` — Yandex Cloud folder identifier

---

# Part 2. A Judge Built on an Open-weight Model

In Part 1, the strongest reference-free result is Self-Consistency EN at **75.77%**. The objective of Part 2 is to construct a judge on an open-weight model by inspecting its internal states, and to surpass this baseline. The approach draws on mechanistic interpretability methods — probing and sparse autoencoders [^saes] — and applies them to hallucination detection.

## Model and Data

**Judge model.** [Gemma 2 2B](https://huggingface.co/google/gemma-2-2b) (`google/gemma-2-2b`) [^gemma] — base version, 27 transformer layers, hidden size 2304. Open weights, locally executable on a single T4 GPU.

**Dataset.** TruthfulQA, with a unified sample shared by both methods — **1,634 pairs** (817 correct + 817 hallucination):

- `correct` is taken from `best_answer` (one canonical correct answer per question)
- `hallucination` is sampled uniformly at random from `incorrect_answers` for the same question
- Each question contributes exactly one correct and one hallucination instance (per-question class balance)

**Data preparation.** A single shared notebook collects hidden states from all 27 layers under three aggregation strategies (`last_token`, `mean_pooling`, `answer_mean`) and stores them in `hidden_states_unified.npz`. Both methods load the same artifact, ensuring direct comparability of results.

## Judge Architecture

```
(Q, A) → Gemma 2 2B → hidden state → Probe / SAE Ensemble → 0 / 1
                      [layer 14/17]                          correct / hallucination
```

The input is a (question, answer) pair formatted as `Question: ... Answer: ...`. The hidden state is extracted from a specific layer, selected via a sweep over all 27 layers. The classifier is trained on 80% of the sample (1,307 instances) and evaluated on the held-out 20%.

## Method 1: Probing Classifier

**Hypothesis.** If the model possesses an internal representation of truthfulness, this signal should be linearly (or near-linearly) separable in its activation space [^probing].

**Implementation.** For each layer × aggregation strategy, we train two probes — `LogisticRegression` (linear) and `MLPClassifier(256, 128)` (non-linear). Features are standardized via `StandardScaler`. In total, 27 × 3 × 2 = 162 probes are evaluated.

**Best results:**

| Probe | Strategy | Layer | Accuracy | F1 (macro) |
|---|---|---|---|---|
| **MLP** | answer_mean | 14 | **77.98%** | 0.7798 |
| Linear | last_token | 13 | 75.54% | 0.7553 |
| MLP | last_token | 19 | 76.76% | 0.7676 |
| Linear | answer_mean | 15 | 75.54% | 0.7553 |

**Layer-wise dynamics (linear, last_token):**

- Layer 0 (embedding): 48.62% — near chance, with no detectable truthfulness signal
- Layer 13 (best): 75.54%
- Final layers: gradual decline

The strongest probes correspond to **mid-network layers (13–14 of 27)**. Truthfulness is encoded neither in the embedding layer nor in the final layers, but in the mid-range — a finding consistent with prior work on internal model states [^geometryoftruth].

## Method 2: SAE Ensemble

**Approach.** Sparse autoencoders [^saes] from the gemma-scope suite [^gemmascope] decompose a hidden state into a sparse basis of 16,384 interpretable directions. Whereas a raw hidden state contains 2,304 entangled components, the SAE representation activates approximately 50–100 features per instance, each corresponding to an identifiable concept.

**Pipeline:**

1. **Multi-layer search.** For candidate layers 12, 15, 17, 20, 22, and 25, we load the corresponding SAE (`gemma-scope-2b-pt-res-canonical`, width 16k) and pass hidden states through it.
2. **Multi-criterion feature selection.** Each feature is scored by three criteria:
   - `Δmean` — difference in mean activation between classes
   - Mutual information with the class label
   - Mann-Whitney U test with Benjamini-Hochberg FDR correction for multiple testing [^bh]
   - The final ranking uses `combo_score`, a normalized sum of the three rank statistics.
3. **Layer selection.** On each candidate layer, a logistic regression on the top-100 features is evaluated via 5-fold cross-validation. The best layer is **17** (CV accuracy 74.72%, with 143 FDR-significant features out of 16,384).
4. **Ensemble.** The top three configurations by hold-out accuracy (`XGBoost top-200`, `LogReg C=0.1 top-500`, `XGBoost top-100`) are combined in a `VotingClassifier` with soft voting.

**Results:**

| Method | Accuracy | F1 (macro) | ROC-AUC |
|---|---|---|---|
| **Ensemble (soft voting, top-200)** | **78.59%** | 0.7850 | 0.8557 |
| XGBoost (top-200) | 76.45% | 0.7644 | 0.8457 |
| LogReg C=0.1 (top-500) | 76.15% | 0.7614 | 0.8537 |
| LogReg C=0.1 (top-200), 10-fold CV | 74.48% ± 2.95% | — | — |

The discrepancy between hold-out accuracy (78.59%) and the 10-fold cross-validation estimate of a single LogReg (74.48%) suggests that the ensemble's reported number may exhibit non-trivial variance across splits.

### Interpretation of Top Features

A key advantage of SAE-based representations is that individual features correspond to identifiable concepts. The leading features at layer 17, with example activations:

| Feature | Class | Δmean | Concept |
|---|---|---|---|
| #3524, #1440 | TRUTH | −2.58, −3.60 | Epistemic caution: "I have no comment" |
| #1250 | TRUTH | −3.73 | Correct refutations of debunked studies (Bargh, Baumeister) |
| #7921 | HALLUCINATION | +4.83 | Confidently asserted falsehoods: "world is flat", "homeopathy is best medicine" |
| #760 | HALLUCINATION | +1.44 | Mythology and superstition: "Bloody Mary", "monkey's paw", "watermelon seeds grow in stomach" |
| #13012 | HALLUCINATION | +4.43 | Magical thinking: "voodoo dolls", "powdered rhino horn cures" |

All listed features are FDR-significant (`p_fdr < 1e-8`). Inspection through [Neuronpedia](https://www.neuronpedia.org/gemma-2-2b/17-gemmascope-res-16k/) corroborates these interpretations.

## Summary: Part 1 vs Part 2

| Method | Accuracy | Approach |
|---|---|---|
| Reference-based | **89.32%** | with reference · part 1 |
| **SAE Ensemble (top-200)** | **78.59%** | **internal states · part 2** |
| **MLP Probe (best)** | **77.98%** | **internal states · part 2** |
| Self-Consistency EN | 75.77% | prompting · part 1 (baseline) |
| **Linear Probe (best)** | **75.54%** | **internal states · part 2** |
| Zero-shot RU | 74.92% | prompting · part 1 |
| CoT EN | 74.72% | prompting · part 1 |
| Few-shot 8 (RU) | 71.84% | prompting · part 1 |
| Decomposed EN | 69.20% | prompting · part 1 |

Both Part 2 methods (probing and SAE ensemble) surpass Self-Consistency EN, the strongest Part 1 method without reference answers. Reference-based remains the upper bound but relies on ground-truth answers from the dataset, which are unavailable in deployment.

## Conclusions

1. **Objective achieved.** The MLP probe (77.98%) and the SAE ensemble (78.59%) outperform Self-Consistency EN (75.77%) by +2.2 pp and +2.8 pp respectively, on 1,634 TruthfulQA pairs.

2. **Truthfulness is encoded mid-network.** The strongest probing layers are 13–14 of 27; the strongest SAE layer is 17. Linear class separability rises from 48.6% (embedding) to approximately 78% mid-network, then declines.

3. **SAE features admit interpretable concept assignments.** We identify stable directions for distinct hallucination types: epistemic caution, conspiratorial claims, mythological narratives, and magical thinking.

4. **Open-weight judges are competitive with proprietary ones.** Gemma 2 2B combined with interpretability tools matches YandexGPT-as-a-judge in the reference-free regime, removing the dependency on an external API while yielding an interpretable solution.

## Limitations

- Both methods are trained on (Q, A) pairs from TruthfulQA where labels are derived from canonical `best_answer` and `incorrect_answers` fields. Generalization to out-of-distribution generations (e.g., free-form YandexGPT outputs) was not evaluated.
- We employ the base version of Gemma 2 2B rather than the instruction-tuned variant. Higher accuracy may be achievable with the IT version or with larger models.
- The reported numbers (77.98% / 78.59%) are based on a single train/test split. For a single LogReg on top-200 SAE features, 10-fold cross-validation yields 74.48% ± 2.95%, which constitutes a more conservative estimate of generalization.

## Setup (Part 2)

Requirements: Python 3.10+, GPU with at least 12 GB of VRAM (T4 / V100 / A100), Hugging Face access (for Gemma 2 2B and gemma-scope SAEs).

Required environment variable:

- `HF_TOKEN` — Hugging Face token (create at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and accept the Gemma 2 license terms on its model page)

Execution order:

1. `data_prep.ipynb` — produces `truthfulqa_pairs.json` and `hidden_states_unified.npz` (~15 minutes on T4)
2. `probing_classifier.ipynb` — trains probes on the prepared hidden states (~10 minutes on T4)
3. `sae.ipynb` — loads SAEs for six layers, performs feature selection, and trains the ensemble (~20 minutes on T4)

Steps 2 and 3 do not recompute hidden states and may be executed independently after Step 1.

---
