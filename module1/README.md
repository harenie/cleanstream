# Module 1

Module 1 contains answer-understanding feature extraction:

- preprocessing
- concept coverage, with DistilBERT prioritized in auto mode and weak-score as backup
- semantic similarity
- reasoning checks, either rule-based or DistilBERT-backed
- contradiction checks
- language-quality checks
- cross-question relevance flags
- the local browser tester

Run the main feature pipeline from the repository root:

```powershell
python module1\scripts\run_module1_features.py "data\raw\synthetic_dataset.xlsx" --model-answers-file "data\reference\model_answers.csv" --strict-model-answers --output module1\generated_outputs\module1_features.csv
```
