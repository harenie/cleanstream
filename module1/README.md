# Module 1

Module 1 contains answer-understanding feature extraction:

- preprocessing
- concept generation from model answers with FLAN-T5, cached for review
- concept coverage with DeBERTa-v3-style NLI, plus legacy DistilBERT and weak-score fallbacks
- semantic similarity with Sentence-BERT by default, plus TF-IDF fallback
- reasoning checks, gated by question-level requirements and then scored from NLI concept support
- contradiction checks with NLI contradiction scores, plus the older rule fallback
- language-quality checks
- cross-question relevance flags
- the local browser tester

Run the main feature pipeline from the repository root:

```powershell
python module1\scripts\run_module1_features.py "data\raw\synthetic_dataset.xlsx" --model-answers-file "data\reference\model_answers.csv" --strict-model-answers --output module1\generated_outputs\module1_features.csv
```

The default path uses `google/flan-t5-small` to create `module1/generated_outputs/generated_concepts.csv`, `MoritzLaurer/deberta-v3-base-mnli-fever-anli` for concept/reasoning/contradiction NLI, and `sentence-transformers/all-MiniLM-L6-v2` for semantic similarity. CUDA is used automatically when PyTorch reports it is available.

Reasoning is controlled by `data/reference/question_requirements.csv`. The pipeline first checks whether a question requires reasoning, such as critical evaluation, comparison, or causal explanation. If reasoning is not required, the row is marked `reasoning_quality=not_applicable`; connective words such as `because` and `therefore` are kept only as explainability signals.

The browser demo exposes three processing paths:

- `NLI path`: generated concepts, NLI concept coverage, NLI reasoning, NLI contradiction, and Sentence-BERT similarity.
- `Legacy DistilBERT path`: manual/reference concepts with the earlier fine-tuned DistilBERT classifier.
- `Lightweight fallback`: reference concepts, weak-score concept labels, rule-based reasoning, rule-based contradiction, and TF-IDF similarity.
