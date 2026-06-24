# Module 1

Module 1 contains answer-understanding feature extraction:

- preprocessing
- concept coverage, with DistilBERT prioritized in auto mode and weak-score as backup
- semantic similarity
- reasoning checks, gated by question-level requirements and then scored by DistilBERT or a rule fallback
- contradiction checks
- language-quality checks
- cross-question relevance flags
- the local browser tester

Run the main feature pipeline from the repository root:

```powershell
python module1\scripts\run_module1_features.py "data\raw\synthetic_dataset.xlsx" --model-answers-file "data\reference\model_answers.csv" --strict-model-answers --output module1\generated_outputs\module1_features.csv
```

Reasoning is controlled by `data/reference/question_requirements.csv`. The pipeline first checks whether a question requires reasoning, such as critical evaluation, comparison, or causal explanation. If reasoning is not required, the row is marked `reasoning_quality=not_applicable`; connective words such as `because` and `therefore` are kept only as explainability signals.
