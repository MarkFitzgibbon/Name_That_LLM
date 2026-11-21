# Logbook: explanation of the TPU experiment notebook

Date: 9 November 2026

Notebook reviewed: `name-that-llm-tpu-experiment.ipynb`.

This entry explains the current executable code and its saved outputs. It is a code review, not a new training run. Cell numbers below are zero-based positions in the notebook, including Markdown cells.

## Purpose and overall pipeline

The task is supervised, multiclass text classification: given a piece of generated text, predict which of 18 named LLMs wrote it. Training examples supply both `text` and the correct `model`; test examples supply `text` without its label.

The pipeline is:

`labelled texts -> character TF-IDF vectors -> small neural network -> predicted LLM label`

There is one classifier architecture, trained twice: first on a training split to select the number of epochs, then from scratch on all labelled rows to generate the submission. The 18 LLM names are the possible output classes. The notebook does not run, fine-tune, query, or ensemble those LLMs.

Despite the filename, this implementation uses a CUDA GPU, not a TPU. Its saved run reports a Tesla T4. No TPU device setup or PyTorch/XLA code is present.

## Opening description: cell 0

The introductory Markdown describes an offline Kaggle baseline, accelerator setup, submission workflow, and balanced accuracy as the evaluation metric. It states that text feature extraction happens on the CPU and neural-network calculations happen on the GPU.

Some explanatory text is out of date: the later executable code adds an external open-pool dataset, although the opening says that no external datasets are needed and the data-loading description says none is used. The later pretrained-text-encoder suggestion is a possible future experiment, not part of this implementation.

## Section 1: settings and immediate GPU check — cells 1–2

The code imports data-processing, feature-extraction, evaluation, and PyTorch libraries. It defines these main settings:

| Setting | Value | Purpose |
| --- | --- | --- |
| `SEED` | 42 | Seeds Python, NumPy, and PyTorch random generators. |
| `REQUIRE_GPU` | `True` | Raises an error if CUDA is unavailable. |
| `MAX_FEATURES` | 30,000 | Caps the character-feature vocabulary. |
| `HIDDEN_DIM` | 128 | Sets the neural network's hidden-layer width. |
| `BATCH_SIZE` | 128 | Processes up to 128 examples per training/prediction batch. |
| `EPOCHS` | 12 | Maximum epochs during validation-based training. |
| `PATIENCE` | 3 | Stops after three consecutive epochs without a better validation score. |
| `LEARNING_RATE` | 0.003 | Controls the AdamW parameter-update scale. |
| `WEIGHT_DECAY` | 0.001 | Configures AdamW's weight regularisation. |
| `VALID_SIZE` | 0.20 | Holds out approximately 20% of prompt groups when grouping is available. |

`DATA_DIR` can override the competition CSV location. Outputs go to `/kaggle/working` when that directory exists, otherwise to a local `baseline_outputs` directory.

`seed_everything()` helps repeat runs, but does not explicitly enforce fully deterministic CUDA operations. The selected device is `cuda:0` when CUDA is available, otherwise CPU if the GPU requirement has been disabled. A small matrix multiplication verifies that GPU calculations actually work before data loading begins. The code also checks that batch size, epoch limit, and patience are positive and creates the output directory.

## Section 2: load and check data — cells 3–5

### Open-pool data: cell 4

The code recursively searches `/kaggle/input` for `ntl_open_pool.parquet`, loads the first match with pandas, and prints its size and label counts. The stored output shows 198,507 rows across the 18 classes.

This cell assumes at least one match exists; otherwise `files[0]` fails. Unlike the later CSV loader, it has no local-path fallback. A copy of the Parquet file in the project directory alone therefore does not make the notebook locally runnable without adapting this cell.

### Competition data and schema alignment: cell 5

The code searches an explicit `DATA_DIR`, or a list containing two Kaggle paths, the working directory, and its parent, for `train.csv` and `test.csv` together. CSV IDs are read as strings.

Before combining the datasets, it copies the pool and conditionally renames `prompt_group` to `prompt_id`. It requires the pool to contain every competition training column, then selects those columns in the same order. Extra pool metadata is discarded. The saved run excludes fields such as provenance, similarity measures, and risk flags; none becomes a classifier input or a filtering condition.

`pd.concat()` appends all pool rows to the competition training rows. The saved output shows 201,206 combined training examples: 2,699 competition examples plus 198,507 pool examples. The preview test file has 3 rows. The notebook does not deduplicate examples by text or filter near-duplicate prompts before concatenating.

Validation checks require the expected columns, nonempty tables, unique non-null IDs, and nonempty text. Missing labels or a label set other than the explicitly listed 18 names cause an error. Duplicate training IDs cause an error rather than being removed automatically.

`LabelEncoder` assigns an integer from 0 to 17 to each model name, producing `y`, the neural network's targets. Its inverse mapping later converts predicted integers back to names. These integers are categories, not an ordering of model quality.

Only `text` is used to construct predictive features. `model` is the target, `prompt_id` controls validation grouping, and `id` identifies rows in the output.

## Section 3: hold out prompts and extract features — cells 6–7

### Validation split

When `prompt_id` exists, `GroupShuffleSplit` assigns entire prompt groups to either training or validation. An assertion checks that no exact prompt ID appears on both sides. This avoids training on one response to a prompt while validating on another response with that same prompt ID.

The 20% setting applies to groups, not an exact fraction of rows. The saved run contains 152,308 training rows and 48,898 validation rows. If prompt IDs are absent, the fallback is a stratified row split that aims to preserve class proportions. Both sides must contain every class.

This grouping protects against shared prompt IDs; it does not prove that differently identified prompts or texts in the external pool are unrelated. The pool's similarity/risk flags are not used to strengthen that protection.

### Character TF-IDF

`make_vectorizer()` creates a `TfidfVectorizer` with these settings:

- `analyzer="char"`, `ngram_range=(3, 5)`: extracts overlapping sequences of 3, 4, and 5 characters, including patterns involving punctuation and whitespace.
- `lowercase=False`: preserves capitalisation.
- `min_df=2`: ignores character sequences appearing in fewer than two fitting documents.
- `max_features=30_000`: retains at most 30,000 features, selected by corpus term frequency after filtering.
- `sublinear_tf=True`: uses `1 + log(count)` for nonzero within-document counts, reducing the influence of repetition.
- `dtype=np.float32`: stores feature values as 32-bit floats.

TF-IDF combines within-document frequency with inverse document frequency, so character sequences common across many documents receive less weight. The vectorizer also uses scikit-learn's default L2 normalisation for each document vector. For example, `hello` contains the character trigrams `hel`, `ell`, and `llo`.

These features can capture recurring spelling, formatting, punctuation, and phrasing patterns associated with different generators. They do not provide a pretrained language model's contextual understanding.

The vectorizer is fitted only on the training split. Validation text is transformed with that fitted vocabulary and IDF values, avoiding feature-fitting leakage from validation. Most feature values are zero, so the matrices stay sparse in CPU memory. The saved training matrix has shape `(152308, 30000)`.

## Section 4: define the classifier and training helpers — cells 8–9

### Neural-network architecture

`TextClassifier` is a feed-forward multilayer perceptron (MLP) with two trainable linear layers:

`up to 30,000 TF-IDF features -> Linear(128) -> ReLU -> Dropout(0.20) -> Linear(18)`

The first linear layer learns combinations of character features. ReLU adds nonlinearity by replacing negative activations with zero. During training, dropout randomly zeroes 20% of hidden activations to discourage over-reliance on particular units. The final layer produces one raw score, or logit, per LLM class.

At 30,000 inputs, the network has 3,842,450 trainable parameters, matching the saved output. It starts with randomly initialised weights and does not use pretrained embeddings, a transformer, or downloaded LLM weights.

The section's Markdown mentions probabilities, but `forward()` actually returns logits. Softmax could convert these scores into probabilities, but the notebook does not explicitly compute or save probabilities. Selecting the largest logit gives the same class as selecting the largest softmax probability.

### Loss, optimisation, and batches

`feature_batch()` converts only the requested sparse rows into a dense array, wraps that array as a PyTorch tensor, and transfers it to the selected device. This avoids allocating the entire dense feature matrix. A full 128-row batch with 30,000 float32 features occupies about 14.6 MiB for its input values alone.

`make_loss()` builds weighted cross-entropy. For class `c`, its weight is `N / (18 * count_c)`, where `N` and `count_c` refer to the labels supplied to that training stage. Less frequent classes therefore receive higher weights. PyTorch's cross-entropy accepts logits and internally performs the required log-softmax calculation.

`train_one_epoch()` enables training mode, shuffles the rows, and processes mini-batches. Each batch clears previous gradients, computes predictions and loss, checks that the loss is finite, backpropagates gradients, and updates the parameters. It returns a batch-size-weighted loss diagnostic; with weighted cross-entropy, this is not an exact single global weighted-loss calculation.

The optimiser is AdamW, using learning rate 0.003 and weight decay 0.001.

`predict_classes()` disables gradient tracking, switches to evaluation mode so dropout is inactive, and predicts in batches. `argmax` selects the highest-scoring class for each row. The classifier and optimiser are then instantiated for validation training and their device placement is checked.

## Section 5: train and evaluate — cells 10–11

The training loop performs one pass over the training split per epoch, predicts the validation labels, and calculates balanced accuracy. A strict improvement saves a CPU copy of the network weights and resets the patience counter. Three consecutive non-improving epochs stop training.

Balanced accuracy is the mean recall across the 18 classes: calculate the fraction of each class correctly identified, then average those 18 fractions equally. This prevents common classes from dominating the metric simply because they have more examples. It differs from ordinary accuracy, which is the fraction of all validation rows classified correctly.

The stored run reports:

| Epoch | Training loss | Validation balanced accuracy |
| --- | --- | --- |
| 1 | 1.8241 | 0.4036 |
| 2 | 1.3818 | 0.4339 |
| 3 | 1.1508 | 0.4289 |
| 4 | 0.9577 | 0.4278 |
| 5 | 0.8009 | 0.4219 |

Epoch 2 is selected. Training stops after epoch 5, then restores the epoch-2 weights. Decreasing training loss alongside declining validation performance is consistent with overfitting after the selected epoch.

The saved classification report gives ordinary accuracy of approximately 0.405 and balanced accuracy of 0.4339. Thus 43.39% describes average class recall, not the proportion of all examples classified correctly. Per-class precision, recall, F1, and support help identify which classes are difficult. For example, saved recall is 0.738 for `gemini-3.1-pro-preview` and 0.150 for `glm-5.3-flash`.

The code saves per-epoch results to `validation_history.csv` and prints elapsed time and peak allocated CUDA memory. The saved values are 55.4 seconds and about 134.5 MiB for this stage. These are historical notebook outputs, not measurements from this review. Since validation selects the epoch count, its score is a model-selection estimate rather than a separate untouched test evaluation.

## Section 6: refit on all labelled data and save artifacts — cells 12–13

The notebook deletes the validation-stage model and feature matrices, collects garbage, clears unused CUDA cache, and resets random seeds.

It fits a new TF-IDF vectorizer on all 201,206 labelled texts and creates a fresh neural network. The vocabulary and IDF values may differ from those used for validation, so the new network is trained from scratch with the new feature representation. Class weights are recalculated from all labels.

Training now lasts exactly `best_epoch` epochs: two in the saved run. There is no validation-based early stopping during this stage. This uses all labelled examples for the final classifier while carrying over the epoch count selected earlier. The 0.4339 score belongs to the earlier validation-stage model; it is not an independently measured score for this final model.

Saved artifacts are:

| File | Contents |
| --- | --- |
| `tfidf.joblib` | Fitted final text vectorizer, including vocabulary and IDF values. |
| `classifier.pt` | Final network weights, input dimension, hidden dimension, and class names. |
| `run_summary.json` | Selected epoch, validation score, row counts, device, and selected settings. |
| `validation_history.csv` | Earlier validation-stage learning history. |

The vectorizer and checkpoint support later inference when loaded with the compatible model definition. Test text is never used to fit either the vocabulary or the network.

## Section 7: classify test rows and write the submission — cells 14–15

The final vectorizer transforms each test text into the same feature space used by the final network. Previously unseen character sequences outside its vocabulary contribute no features.

`predict_classes(final_model, X_test)` produces one integer class per row. `encoder.inverse_transform()` converts those integers to LLM names. The submission contains exactly two columns, `id` and `model`, preserving the original test IDs and row order.

Assertions check column order, row count, missing values, unique IDs, ID ordering, and membership in the allowed label set. The notebook writes `submission.parquet` using PyArrow, reads it back, and checks that the recovered table exactly matches the intended submission. This verifies file integrity and formatting, not prediction correctness.

The saved preview predictions are:

| Test ID | Predicted LLM |
| --- | --- |
| `00353fb1c541` | `claude-sonnet-5` |
| `005f5a9a6139` | `gpt-5.6-sol` |
| `0062a53d5487` | `gemini-3.1-pro-preview` |

No test ground-truth labels are loaded, so the notebook cannot calculate test accuracy. Its Markdown describes Kaggle rerunning the notebook on hidden test data during scoring; the executable code predicts whichever test rows are supplied for that run, in batches.

## Closing guidance: cell 16

The final Markdown recommends recording the baseline score, reducing batch size or feature count if memory is limited, and changing one setting at a time. It explains that CPU feature preparation can leave the GPU intermittently idle. It also proposes a pretrained text encoder as future work; no such encoder is implemented here.

## Interpretation and limits

This experiment learns statistical differences in the generated texts rather than identifying their authors with certainty. Character patterns may reflect generator style, response content, or differences between the sources used to assemble the training data. Closely related generators can produce similar patterns and remain hard to separate.

The exact prompt-group split is a useful protection, but pool metadata about near-training prompts and routing risks is dropped rather than used. The stored validation score should therefore be interpreted with the composition of this augmented dataset in mind.

The existing `Logbooks/Day1.txt` separately records an initial leaderboard score of 0.429, validation balanced accuracy of 0.4339 with character n-grams `(3, 5)`, and 0.4103 with `(1, 3)`. The current notebook code uses `(3, 5)` and stores the 0.4339 validation result. Its saved outputs alone do not establish that the separate leaderboard score came from this exact augmented-data run.

Review validation: all 17 notebook cells and their saved outputs were inspected. The notebook was not modified or executed for this documentation task.
