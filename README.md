# Cleanstream

Preprocessing, Module 1 feature extraction, concept coverage, semantic similarity, and first training baseline code for the synthetic answer dataset.

This repository is intentionally small so group members can review the early grading pipeline before it is connected to the full project.

## Files

- `module1/preprocessing/preprocessing.py` - reusable preprocessing functions.
- `module1/concept_coverage/concept_coverage.py` - generated/reference concept coverage helpers.
- `module1/concept_coverage/concept_generation.py` - FLAN-T5 concept generation and cache helpers.
- `module1/concept_coverage/llm_concept_coverage.py` - NLI, transformer, and weak-score concept coverage inference.
- `module1/concept_coverage/concepts_reference.py` - expected concept reference loading and fallback generation.
- `module1/nli/nli.py` - shared DeBERTa-v3-style NLI scoring for concept, reasoning, and contradiction checks.
- `module1/semantic_similarity/semantic_similarity.py` - Sentence-BERT and TF-IDF semantic similarity helpers.
- `module1/language_quality/language_quality.py` - optional spelling and grammar quality checks.
- `module1/reasoning/reasoning.py` - reasoning quality, contradiction, and noise-detection helpers.
- `module1/model_answers/model_answers.py` - model-answer reference loading and attachment helpers.
- `module1/module1_features.py` - combined Module 1 feature extraction pipeline.
- `module1/scripts/run_preprocessing.py` - command-line script to run preprocessing on an Excel or CSV dataset.
- `module1/scripts/run_concept_coverage.py` - command-line script to add concept coverage columns.
- `module1/scripts/run_semantic_similarity.py` - command-line script to add semantic similarity columns.
- `module1/scripts/run_module1_features.py` - command-line script to build the full Module 1 output.
- `module1/scripts/build_concept_reference.py` - command-line script to create expected concepts from marking-scheme bullet points.
- `module1/scripts/prepare_concept_training_data.py` - command-line script to create concept-answer training pairs.
- `module1/scripts/train_concept_model.py` - command-line script to fine-tune the concept coverage transformer.
- `module2/scripts/run_training.py` - command-line script to train and evaluate the first score baseline.
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

- generates expected concept statements from marking-schema/model answers with FLAN-T5 and caches them in `module1/generated_outputs/generated_concepts.csv`
- keeps `data/reference/concepts.csv` as a reviewable/manual fallback rather than the default source of truth
- uses a DeBERTa-v3-style NLI model by default to label each concept as `missing`, `partial`, or `covered`
- keeps the earlier DistilBERT classifier and `weak-score` bootstrap backend as explicit fallback options
- renames the dataset's student-answer field to `student_answer` in the concept output
- removes raw `generated_answer` columns from the concept output
- adds `model_answer`, `missing_model_answer`, `concept_backend`, `concept_source`, `concepts`, `concepts_present`, `concepts_partial`, `concepts_missing`, `concept_prediction_details`, and `concepts_covered_ratio`

The semantic similarity step:

- compares each `student_answer` against the marking-schema `model_answer`
- uses Sentence-BERT by default for sentence-level semantic similarity
- keeps TF-IDF as a lightweight fallback with `--similarity-backend tfidf`
- adds `semantic_similarity_score`, `student_answer_length`, and `model_answer_length`
- marks rows without a model answer as `missing_model_answer`

The full Module 1 pipeline:

- combines concept coverage, semantic similarity, reasoning quality, scoped contradiction checks, language-quality indicators, and cross-question relevance flags
- uses FLAN-T5 generated concepts, NLI concept coverage, NLI contradiction checks, and Sentence-BERT similarity by default
- checks `data/reference/question_requirements.csv` before scoring reasoning; non-reasoning questions are marked `not_applicable`
- keeps DistilBERT, rule-based, weak-score, and TF-IDF paths for local comparison and fallback
- does not train the final scoring model; score prediction belongs to Module 2

The first training baseline:

- trains a Ridge regression model to predict `ai_score`
- uses `semantic_similarity_score`, `concepts_covered_ratio`, `student_answer_length`, and `model_answer_length`
- uses a fixed `2/3` train and `1/3` test split with `random_state=42`
- writes the split, predictions, metrics, and trained model to `module2/training_results`

Contradiction detection uses NLI by default, comparing the student answer against generated expected concepts and recording the strongest contradiction score. The older scoped pattern detector remains available with `--contradiction-backend rule-based`.

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

For the Module 1 NLI/Sentence-BERT/FLAN-T5 path, install the LLM requirements:

```powershell
pip install -r requirements-llm.txt
```

## Run

From inside this repository:

```powershell
python module1\scripts\run_preprocessing.py "data\raw\synthetic_dataset.xlsx" --output module1\outputs\preprocessed_dataset.csv
```

The script prints a preprocessing summary and creates:

```text
preprocessed_dataset.csv
```

That output file is only for local checking. It does not need to be committed.

To add concept coverage columns:

```powershell
python module1\scripts\run_concept_coverage.py "data\raw\synthetic_dataset.xlsx" --model-answers-file "data\reference\model_answers.csv" --concept-source generated --concept-backend nli --output module1\outputs\concept_coverage_output.xlsx
```

If the dataset already has a populated `model_answer` column, the `--model-answers-file` argument is optional.

To rebuild the expected concept reference from marking-scheme bullet points:

```powershell
python module1\scripts\build_concept_reference.py --model-answers data\reference\model_answers.csv --output data\reference\concepts.csv
```

The manual `data\reference\concepts.csv` file is now a fallback/review artifact. The default generated concept cache is `module1\generated_outputs\generated_concepts.csv`.

To prepare weak concept-level training rows from existing answer scores:

```powershell
python module1\scripts\prepare_concept_training_data.py data\raw\synthetic_dataset.xlsx --concept-reference data\reference\concepts.csv --model-answers-file data\reference\model_answers.csv --output data\training\concept_coverage_training.csv
```

To train the transformer concept coverage model:

```powershell
pip install -r requirements-llm.txt
python module1\scripts\train_concept_model.py --training-data data\training\concept_coverage_training.csv --output-dir module1\models\concept_coverage_model --base-model distilbert-base-uncased --epochs 1 --batch-size 8 --device auto
```

The current local model was trained from weak labels derived from `ai_score`. For stronger research results, manually review/update `data\training\concept_coverage_training.csv` labels before retraining.

To add semantic similarity columns with the default Sentence-BERT backend:

```powershell
python module1\scripts\run_semantic_similarity.py "data\raw\synthetic_dataset.xlsx" --model-answers-file "data\reference\model_answers.csv" --output module1\outputs\semantic_similarity_output.xlsx
```

To use the lightweight TF-IDF fallback:

```powershell
python module1\scripts\run_semantic_similarity.py "data\raw\synthetic_dataset.xlsx" --model-answers-file "data\reference\model_answers.csv" --similarity-backend tfidf --output module1\outputs\semantic_similarity_tfidf_output.xlsx
```

To build the full Module 1 feature file with lightweight language-quality checks:

```powershell
python module1\scripts\run_module1_features.py "data\raw\synthetic_dataset.xlsx" --model-answers-file "data\reference\model_answers.csv" --strict-model-answers --output module1\generated_outputs\module1_features.csv
```

To force the trained concept coverage model in Module 1:

```powershell
python module1\scripts\run_module1_features.py "data\raw\synthetic_dataset.xlsx" --model-answers-file "data\reference\model_answers.csv" --concept-source reference --concept-backend trained-llm --concept-model-path module1\models\concept_coverage_model --similarity-backend tfidf --contradiction-backend rule-based --strict-model-answers --output module1\generated_outputs\module1_features_legacy.csv
```

To force DistilBERT-backed reasoning quality as well:

```powershell
python module1\scripts\run_module1_features.py "data\raw\synthetic_dataset.xlsx" --model-answers-file "data\reference\model_answers.csv" --reasoning-backend trained-llm --reasoning-model-path module1\models\concept_coverage_model --strict-model-answers --output module1\generated_outputs\module1_features.csv
```

To open the simple browser tester for one student answer and one schema/model answer:

```powershell
python module1\scripts\run_module1_demo_server.py
```

Then open:

```text
http://127.0.0.1:8765
```

Language checks are optional:

```powershell
python module1\scripts\run_module1_features.py "data\raw\synthetic_dataset.xlsx" --model-answers-file "data\reference\model_answers.csv" --language-check none --output module1\generated_outputs\module1_features.csv
python module1\scripts\run_module1_features.py "data\raw\synthetic_dataset.xlsx" --model-answers-file "data\reference\model_answers.csv" --language-check simple --apply-language-penalty --output module1\generated_outputs\module1_features.csv
```

For stronger LanguageTool checks, install optional dependencies first:

```powershell
pip install -r requirements-optional.txt
python module1\scripts\run_module1_features.py "data\raw\synthetic_dataset.xlsx" --model-answers-file "data\reference\model_answers.csv" --language-check languagetool --output module1\generated_outputs\module1_features.csv
```

To train the first `ai_score` baseline:

```powershell
python module2\scripts\run_training.py "data\raw\synthetic_dataset.xlsx" --model-answers-file "data\reference\model_answers.csv" --output-dir module2\training_results
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
from module1.preprocessing.preprocessing import clean_text

cleaned = clean_text(" Ubiquity means the internrt is everywhere [[ and good. ")
print(cleaned)
```

Expected style of output:

```text
ubiquity means the internrt is everywhere and good.
```
