# Logbook entry: First submission notebook creation

**Date:** 30 October 2026  
**Notebook:** [name_that_llm_first_submission.ipynb](../name_that_llm_first_submission.ipynb)  
**Competition:** [Name That LLM](https://www.kaggle.com/competitions/name-that-llm)  
**Version:** Initial CPU baseline

## Objective and implementation

The first notebook implements a complete supervised text-classification workflow: read the competition files, inspect and validate the data, extract text features, evaluate a classifier, train on all labeled passages, and generate the required submission file.

The task is to identify which of 18 candidate language models generated a passage. The notebook uses the passage text as its only predictive input. Training labels come from the `model` column. Record IDs identify submission rows, while `prompt_id` controls validation splits; neither is used as a predictive feature.

The baseline runs on CPU using NumPy, pandas, scikit-learn, Matplotlib, and PyArrow, with IPython for displaying tables. It requires no pretrained weights, external training data, API calls, or downloads during execution. It is intended as a reproducible starting point for later experiments.

## Classification approach

The classification pipeline is:

```text
Passage text
    ├── Word TF-IDF features ─────── weight 0.7 ──┐
    └── Character TF-IDF features ── weight 1.0 ──┤
                                                ↓
                                  Concatenated sparse features
                                                ↓
                                  Class-balanced LinearSVC
                                                ↓
                                  One authoring-model label
```

### Text representations

TF-IDF (term frequency–inverse document frequency) converts text into numeric features. It weights terms according to their frequency within a passage and their rarity across the training documents. Two `TfidfVectorizer` instances provide complementary representations:

| Setting | Word branch | Character branch |
| --- | --- | --- |
| Extracted units | Individual words and pairs of adjacent words | Sequences of 2–5 characters, including punctuation and spaces |
| `ngram_range` | `(1, 2)` | `(2, 5)` |
| Maximum vocabulary size | 60,000 | 120,000 |
| Minimum document frequency | 2 documents | 2 documents |
| Feature weight in `FeatureUnion` | 0.7 | 1.0 |
| Numeric storage | `float32` | `float32` |

Both branches preserve case and accents and use sublinear term frequency, which reduces the influence of repeated occurrences through logarithmic scaling. Stop words are retained because common function words may help distinguish writing styles. Word features capture phrasing preferences; character features can capture typography, punctuation, spelling, and fragments spanning word boundaries.

`FeatureUnion` scales and concatenates the two sparse feature matrices. These are two feature representations feeding one classifier, rather than two independently voting classification models. The feature limits and weights are initial design choices, not values established by a competition-data tuning experiment.

### Classifier

The classifier is scikit-learn's **`LinearSVC`**, a linear support vector machine. Its default multiclass strategy fits one-versus-rest decision functions and selects the class with the highest decision score. These scores are margins, not calibrated probabilities.

| Parameter | Initial value | Purpose |
| --- | --- | --- |
| `C` | `1.0` | Controls the trade-off between regularization and penalizing training errors |
| `class_weight` | `'balanced'` | Gives greater weight to classes with fewer training examples |
| `dual` | `True` | Uses the dual optimization formulation |
| `max_iter` | `10_000` | Sets the optimizer's iteration limit |
| `tol` | `1e-4` | Sets the optimization stopping tolerance |
| `random_state` | `42` | Sets the classifier's random seed |

Class weighting is a starting choice suited to the competition's balanced-accuracy metric, which gives each class equal importance. It does not guarantee optimal balanced accuracy.

The candidate LLMs are the **target classes**, not models loaded or run by this notebook:

| Provider | Predicted labels |
| --- | --- |
| OpenAI | `gpt-5.5`, `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-6-astra` |
| Google | `gemini-3.1-pro-preview`, `gemini-3.6-flash`, `gemini-3.7-flash`, `gemini-3.8-flash` |
| Anthropic | `claude-fable-5`, `claude-opus-4-6`, `claude-opus-4-8`, `claude-opus-5`, `claude-sonnet-5` |
| Qwen | `qwen3.8-max` |
| ZAI | `glm-5.3`, `glm-5.3-flash` |
| Moonshot AI | `kimi-k3` |

## How each notebook section operates

### Introduction: requirements and scope

The opening Markdown records the requirements available from the official overview: balanced-accuracy evaluation, offline execution, a nine-hour runtime limit, and a file named `submission.parquet` containing `id` and `model` columns. It explains that Kaggle replaces the single interactive test example with hidden test rows during scoring, so inference must handle a variable number of rows.

It also records the research limitation: the full Rules page was blocked by reCAPTCHA, and the Data page did not expose the actual file contents during preparation. Full legal-rule verification and real-data execution were therefore not completed.

### Section 1: Settings and dependencies

This section imports the libraries and centralizes the main settings:

| Setting | Default |
| --- | --- |
| Random seed | `42` |
| Validation folds | `5` |
| Run cross-validation | `True` |
| SVM `C` | `1.0` |
| Inference batch size | `512` |
| Input directory | `/kaggle/input/competitions/name-that-llm` |
| Output directory | `/kaggle/working` if present; otherwise the current directory |
| Text column | Automatically select exactly one of `text` or `text_snippet` shared by train and test |

It defines the 18 permitted labels, checks that they are unique, seeds NumPy, and prints the pandas and scikit-learn versions. The notebook contains no package-installation cells; local execution requires installing its dependencies beforehand.

### Section 2: Read and validate the data

The loader expects `train.csv` and `test.csv`. If the default Kaggle directory is unavailable, it checks the alternative `/kaggle/input/name-that-llm` mount. It stops with an explanatory error if either file is missing.

The expected training schema is `id`, `prompt_id`, a passage column, and `model`. Test data requires `id` and the passage column. pandas reads every field as a string to preserve IDs with leading zeros. `keep_default_na=False` prevents literal strings such as `NA` and `null` from being interpreted as missing values. The CSV parser supports quoted commas and embedded newlines.

Validation checks required columns, nonempty datasets, null fields, blank or duplicate IDs, blank passages, and blank training prompt IDs. It requires training labels to match all 18 configured classes and checks that PyArrow is available for writing Parquet. Ambiguous text-column detection requires an explicit `TEXT_COLUMN` setting.

The section then prepares `X` (passages), `y` (authoring-model labels), and `groups` (prompt IDs). It displays training class counts, the number of prompts, passage-length statistics, and the number of repeated training passages. It does not lowercase, strip punctuation, or otherwise clean away potential style signals before vectorization.

### Section 3: Define the model

`make_model()` constructs the TF-IDF branches, their `FeatureUnion`, and the `LinearSVC` classifier inside a scikit-learn `Pipeline`.

Calling this function returns a fresh, unfitted pipeline. This matters during validation: each fold must learn its vocabulary, IDF weights, and classifier exclusively from that fold's training rows. No fitted feature extractor is shared across folds.

### Section 4: Validate on held-out prompts

With `RUN_CV=True`, `GroupKFold` creates five splits using `prompt_id` as the grouping variable. All responses to one prompt stay in the same fold. This evaluates transfer to other prompts and avoids placing answers to the same question in both training and validation.

For each fold, the code:

1. Checks that training and validation prompt groups do not overlap.
2. Rejects exact passage duplicates crossing the split. It stops for manual group correction rather than merging groups automatically.
3. Checks that all 18 classes appear in both training and validation.
4. Fits a fresh pipeline on the training subset.
5. Predicts the validation subset and stores out-of-fold predictions.
6. Prints fold balanced accuracy and elapsed time, then releases the fitted fold model.

Balanced accuracy is the average recall across the 18 classes. Recall for one model is the fraction of its passages correctly attributed to it. After all folds, the notebook reports pooled out-of-fold balanced accuracy, the fold-score mean and standard deviation, and the uniform-random expected reference of `1/18` (approximately 5.56%).

It also displays per-class precision, recall, F1 score, and support; writes `oof_predictions.csv`; and displays a row-normalized confusion matrix. The matrix's diagonal shows recall for each true authoring model. If cross-validation is disabled, this section reports that no local performance estimate is available.

### Section 5: Refit and predict every test row

A fresh pipeline is trained on all labeled training passages after validation. This final model supplies the submission predictions; the five validation models are not ensembled or retained for inference.

Test passages are processed in batches of up to 512 rows to limit the size of feature matrices created during inference. Predictions are concatenated in input order. The code prints elapsed refit/inference time and the predicted class counts.

Every passage receives an independent prediction. The notebook does not force equal numbers of predictions per class or assume one response from every model for each test prompt.

### Section 6: Write and read back the submission

The submission is constructed from the original test IDs and predicted labels. `validate_submission()` checks:

- Exactly two columns in order: `id`, `model`.
- One output row for every input test row.
- No null values, blank IDs, or duplicate IDs.
- IDs and row order exactly matching the test input.
- Every predicted model belonging to the configured valid-label list.

If `sample_submission.parquet` is present, its column names are checked as an additional schema reference. Its example IDs and row count do not control predictions.

The notebook writes a genuine Parquet file using PyArrow with `index=False`. It reads the file back, repeats the validation, and checks that the saved table equals the in-memory submission. It also writes `run_summary.json` containing the main settings, input row counts, library versions, and validation scores when available.

### Section 7: Submit on Kaggle

The final Markdown section describes the manual submission steps: review and accept the competition rules, import the notebook, attach the data, select CPU, disable Internet, run all cells, save a version using **Save & Run All**, and submit `submission.parquet` from the completed version's output.

The notebook does not accept rules or submit an entry automatically. This section also suggests later experiments with character n-gram ranges and `C`, assessed on the same prompt folds.

## Files produced when the notebook runs

| Output | Purpose |
| --- | --- |
| `submission.parquet` | Required competition submission with one predicted authoring model per test ID |
| `oof_predictions.csv` | True labels and held-out predictions for error analysis; written only when cross-validation runs |
| `run_summary.json` | Settings, row counts, versions, and available validation scores |

Tables and the confusion matrix appear in notebook output. The first version does not serialize the trained classifier or vectorizers; rerunning trains them again. The run summary is a compact record, not a complete environment lockfile or dataset-version record.

## Verification completed during creation

The notebook's structure was validated, and every Python code cell was checked for valid syntax. Its code was exercised sequentially using synthetic training data covering all 18 labels and grouped prompts. Integration checks passed for:

- A one-row test file using the `text` column.
- A 1,027-row test file using `text_snippet`, exercising multiple inference batches.
- Prompt-grouped cross-validation and final refitting.
- String IDs with leading zeros and CSV passages containing quotes and newlines.
- Actual Parquet writing, reading, and equality checks.
- Rejection of submissions containing duplicate IDs or an unknown model label.

These checks establish that the workflow operated on the synthetic fixtures. They do not establish accuracy, runtime, memory use, or successful submission on the real competition data. The delivered notebook was left unexecuted, without synthetic results embedded in its cells.

## Current limitations and next steps

No real-data validation score or Kaggle leaderboard score was available at creation. The feature weights, vocabulary caps, n-gram ranges, and SVM regularization are baseline settings. The code also deliberately stops if the expected schema, class coverage, or split integrity checks fail.

The next step is to run the notebook on the authorized Kaggle data and record the actual schema, fold scores, pooled balanced accuracy, per-model recall, execution time, and submission outcome in a new logbook entry. Subsequent experiments can compare feature settings and regularization against this baseline while keeping prompt groups fixed.
