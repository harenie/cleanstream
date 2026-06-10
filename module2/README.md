# Module 2

Module 2 is reserved for score prediction and model training.

The current baseline trains a Ridge regression model from Module 1 features:

```powershell
python module2\scripts\run_training.py "data\raw\synthetic_dataset.xlsx" --model-answers-file "data\reference\model_answers.csv" --output-dir module2\training_results
```
