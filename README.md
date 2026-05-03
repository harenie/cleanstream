# Cleanstream

Preprocessing-only code for the synthetic answer dataset.

This repository is intentionally small so group members can review just the preprocessing work before it is connected to the full grading pipeline.

## Files

- `preprocessing.py` - reusable preprocessing functions.
- `run_preprocessing.py` - command-line script to run preprocessing on an Excel or CSV dataset.
- `concept_coverage.py` - reusable concept extraction and concept coverage helpers.
- `run_concept_coverage.py` - command-line script to add concept coverage columns.
- `README.md` - setup and usage instructions.

No dataset files or generated output files are included in this repository.

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
