# Cleanstream

Preprocessing, concept coverage, semantic similarity, and first training baseline code for the synthetic answer dataset.

This repository is intentionally small so group members can review the early grading pipeline before it is connected to the full project.

## Files

- `preprocessing.py` - reusable preprocessing functions.
- `run_preprocessing.py` - command-line script to run preprocessing on an Excel or CSV dataset.
- `concept_coverage.py` - reusable concept extraction and concept coverage helpers.
- `run_concept_coverage.py` - command-line script to add concept coverage columns.
- `semantic_similarity.py` - reusable TF-IDF cosine semantic similarity helpers.
- `run_semantic_similarity.py` - command-line script to add semantic similarity columns.
- `run_training.py` - command-line script to train and evaluate the first score baseline.
- `requirements.txt` - Python packages required for preprocessing, similarity, and training.
- `README.md` - setup and usage instructions.

Generated output files are local checking artifacts and do not need to be committed.

## What The Code Does

The preprocessing step:

- standardizes column names into lowercase `snake_case`
- cleans noisy answer text
- creates cleaned text columns such as `synthetic_answer_clean`
- cleans `generated_answer`, `answer`, and `question` if those columns exist
- trims chapter and difficulty values
- prints a small summary with row count, columns, missing values, question count, and answer count
- saves a cleaned CSV for checking

The concept coverage step:

- compares each `student_answer` against the marking-schema `model_answer`
- extracts concept keywords from the model answer
- renames the dataset's student-answer field to `student_answer` in the concept output
- removes raw `generated_answer` columns from the concept output
- adds `model_answer`, `missing_model_answer`, `concepts`, `concepts_present`, `concepts_missing`, and `concepts_covered_ratio`

The semantic similarity step:

- compares each `student_answer` against the marking-schema `model_answer`
- uses a TF-IDF word n-gram vectorizer and cosine similarity
- adds `semantic_similarity_score`, `student_answer_length`, and `model_answer_length`
- marks rows without a model answer as `missing_model_answer`

The first training baseline:

- trains a Ridge regression model to predict `ai_score`
- uses `semantic_similarity_score`, `concepts_covered_ratio`, `student_answer_length`, and `model_answer_length`
- uses a fixed `2/3` train and `1/3` test split with `random_state=42`
- writes the split, predictions, metrics, and trained model to `training_results`

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
pip install pandas openpyxl
```

For semantic similarity and training, install all requirements:

```powershell
pip install -r requirements.txt
```

## Run

From inside this repository:

```powershell
python run_preprocessing.py "P:\Harine_Project\Synthetic Data - FOR PREPROCESS.xlsx" --output preprocessed_dataset.csv
```

The script prints a preprocessing summary and creates:

```text
preprocessed_dataset.csv
```

That output file is only for local checking. It does not need to be committed.

To add concept coverage columns:

```powershell
python run_concept_coverage.py "Synthetic Data - FOR PREPROCESS.xlsx" --model-answers-file "P:\Harine_Project\automated_answer_grader\data\reference\model_answers.csv" --output concept_coverage_output.xlsx
```

If the dataset already has a populated `model_answer` column, the `--model-answers-file` argument is optional.

To add semantic similarity columns:

```powershell
python run_semantic_similarity.py "Synthetic Data - FOR PREPROCESS.xlsx" --model-answers-file "P:\Harine_Project\automated_answer_grader\data\reference\model_answers.csv" --output semantic_similarity_output.xlsx
```

To train the first `ai_score` baseline:

```powershell
python run_training.py "Synthetic Data - FOR PREPROCESS.xlsx" --model-answers-file "P:\Harine_Project\automated_answer_grader\data\reference\model_answers.csv" --output-dir training_results
```

The current reference file has model answers for `Q1`, `Q2`, and `Q3`, so training currently uses `36` eligible rows. Add the remaining marking-schema answers to the reference file before treating the metrics as full-dataset results.

## Main Function For Review

The core cleaning function is:

```python
from preprocessing import clean_text

cleaned = clean_text(" Ubiquity means the internrt is everywhere [[ and good. ")
print(cleaned)
```

Expected style of output:

```text
ubiquity means the internrt is everywhere and good.
```
