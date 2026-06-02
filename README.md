# Cleanstream

Preprocessing, Module 1 feature extraction, concept coverage, semantic similarity, and first training baseline code for the synthetic answer dataset.

This repository is intentionally small so group members can review the early grading pipeline before it is connected to the full project.

## Files

- `preprocessing/preprocessing.py` - reusable preprocessing functions.
- `concept_coverage/concept_coverage.py` - reference-concept coverage helpers.
- `concept_coverage/llm_concept_coverage.py` - transformer/LLM concept coverage inference.
- `concept_coverage/concepts_reference.py` - expected concept reference loading and generation.
- `semantic_similarity/semantic_similarity.py` - reusable TF-IDF cosine semantic similarity helpers.
- `language_quality/language_quality.py` - optional spelling and grammar quality checks.
- `reasoning/reasoning.py` - reasoning quality, contradiction, and noise-detection helpers.
- `model_answers/model_answers.py` - model-answer reference loading and attachment helpers.
- `module1/module1_features.py` - combined Module 1 feature extraction pipeline.
- `scripts/run_preprocessing.py` - command-line script to run preprocessing on an Excel or CSV dataset.
- `scripts/run_concept_coverage.py` - command-line script to add concept coverage columns.
- `scripts/run_semantic_similarity.py` - command-line script to add semantic similarity columns.
- `scripts/run_module1_features.py` - command-line script to build the full Module 1 output.
- `scripts/build_concept_reference.py` - command-line script to create expected concepts from marking-scheme bullet points.
- `scripts/prepare_concept_training_data.py` - command-line script to create concept-answer training pairs.
- `scripts/train_concept_model.py` - command-line script to fine-tune the concept coverage transformer.
- `scripts/run_training.py` - command-line script to train and evaluate the first score baseline.
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

- compares each `student_answer` against predefined expected concepts from `data/reference/concepts.csv`
- uses a trainable transformer classifier to label each concept as `missing`, `partial`, or `covered`
- provides a `weak-score` bootstrap backend before human concept-level labels are available
- renames the dataset's student-answer field to `student_answer` in the concept output
- removes raw `generated_answer` columns from the concept output
- adds `model_answer`, `missing_model_answer`, `concepts`, `concepts_present`, `concepts_partial`, `concepts_missing`, `concept_prediction_details`, and `concepts_covered_ratio`

The semantic similarity step:

- compares each `student_answer` against the marking-schema `model_answer`
- uses a TF-IDF word n-gram vectorizer and cosine similarity by default
- fits TF-IDF on marking-schema/model-answer text only, so evaluated student answers do not define their own similarity vocabulary
- supports optional Sentence-BERT with `--similarity-backend sentence-bert`
- adds `semantic_similarity_score`, `student_answer_length`, and `model_answer_length`
- marks rows without a model answer as `missing_model_answer`

The full Module 1 pipeline:

- combines concept coverage, semantic similarity, reasoning quality, scoped contradiction checks, language-quality indicators, and cross-question relevance flags
- treats reasoning quality as a rule-based signal from explicit explanation markers, not a final reasoning judge
- can run locally with `weak-score` concept coverage, or use the optional trained transformer concept model with `--concept-backend trained-llm`
- does not train the final scoring model; score prediction belongs to Module 2

The first training baseline:

- trains a Ridge regression model to predict `ai_score`
- uses `semantic_similarity_score`, `concepts_covered_ratio`, `student_answer_length`, and `model_answer_length`
- uses a fixed `2/3` train and `1/3` test split with `random_state=42`
- writes the split, predictions, metrics, and trained model to `training_results`

Contradiction detection is scoped by question type. It runs only for definition or concept-identification questions such as "define", "what is", or "explain the concept". It is skipped for advantages/disadvantages, pros/cons, benefits/drawbacks, comparison, challenge, and limitation questions because balanced answers may naturally contain opposing points.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv_clean
.\.venv_clean\Scripts\Activate.ps1
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
python scripts\run_preprocessing.py "data\raw\synthetic_dataset.xlsx" --output outputs\preprocessed_dataset.csv
```

The script prints a preprocessing summary and creates:

```text
preprocessed_dataset.csv
```

That output file is only for local checking. It does not need to be committed.

To add concept coverage columns:

```powershell
python scripts\run_concept_coverage.py "data\raw\synthetic_dataset.xlsx" --model-answers-file "data\reference\model_answers.csv" --concept-reference "data\reference\concepts.csv" --concept-backend trained-llm --concept-model-path "models\concept_coverage_model" --output outputs\concept_coverage_output.xlsx
```

If the dataset already has a populated `model_answer` column, the `--model-answers-file` argument is optional.

To rebuild the expected concept reference from marking-scheme bullet points:

```powershell
python scripts\build_concept_reference.py --model-answers data\reference\model_answers.csv --output data\reference\concepts.csv
```

To prepare weak concept-level training rows from existing answer scores:

```powershell
python scripts\prepare_concept_training_data.py data\raw\synthetic_dataset.xlsx --concept-reference data\reference\concepts.csv --model-answers-file data\reference\model_answers.csv --output data\training\concept_coverage_training.csv
```

To train the transformer concept coverage model:

```powershell
pip install -r requirements-llm.txt
python scripts\train_concept_model.py --training-data data\training\concept_coverage_training.csv --output-dir models\concept_coverage_model --base-model distilbert-base-uncased --epochs 1 --batch-size 8 --device auto
```

The current local model was trained from weak labels derived from `ai_score`. For stronger research results, manually review/update `data\training\concept_coverage_training.csv` labels before retraining.

To add semantic similarity columns:

```powershell
python scripts\run_semantic_similarity.py "data\raw\synthetic_dataset.xlsx" --model-answers-file "data\reference\model_answers.csv" --output outputs\semantic_similarity_output.xlsx
```

To test the optional Sentence-BERT similarity backend after installing its dependency:

```powershell
pip install sentence-transformers
python scripts\run_semantic_similarity.py "data\raw\synthetic_dataset.xlsx" --model-answers-file "data\reference\model_answers.csv" --similarity-backend sentence-bert --output outputs\semantic_similarity_sbert_output.xlsx
```

To build the full Module 1 feature file with lightweight language-quality checks:

```powershell
python scripts\run_module1_features.py "data\raw\synthetic_dataset.xlsx" --model-answers-file "data\reference\model_answers.csv" --strict-model-answers --output outputs\module1_features.csv
```

To use the trained concept coverage model in Module 1:

```powershell
python scripts\run_module1_features.py "data\raw\synthetic_dataset.xlsx" --model-answers-file "data\reference\model_answers.csv" --concept-backend trained-llm --concept-model-path models\concept_coverage_model --strict-model-answers --output outputs\module1_features.csv
```

To open the simple browser tester for one student answer and one schema/model answer:

```powershell
python scripts\run_module1_demo_server.py
```

Then open:

```text
http://127.0.0.1:8765
```

Language checks are optional:

```powershell
python scripts\run_module1_features.py "data\raw\synthetic_dataset.xlsx" --model-answers-file "data\reference\model_answers.csv" --language-check none --output outputs\module1_features.csv
python scripts\run_module1_features.py "data\raw\synthetic_dataset.xlsx" --model-answers-file "data\reference\model_answers.csv" --language-check simple --apply-language-penalty --output outputs\module1_features.csv
```

For stronger LanguageTool checks, install optional dependencies first:

```powershell
pip install -r requirements-optional.txt
python scripts\run_module1_features.py "data\raw\synthetic_dataset.xlsx" --model-answers-file "data\reference\model_answers.csv" --language-check languagetool --output outputs\module1_features.csv
```

To train the first `ai_score` baseline:

```powershell
python scripts\run_training.py "data\raw\synthetic_dataset.xlsx" --model-answers-file "data\reference\model_answers.csv" --output-dir outputs\training_results
```

The current reference file contains model answers for all 60 question IDs.

## Test

Run the Module 1 checks with:

```powershell
pytest
```

## Main Function For Review

The core cleaning function is:

```python
from preprocessing.preprocessing import clean_text

cleaned = clean_text(" Ubiquity means the internrt is everywhere [[ and good. ")
print(cleaned)
```

Expected style of output:

```text
ubiquity means the internrt is everywhere and good.
```
