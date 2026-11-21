# Name That LLM: starter GPU baseline

Open or import [name_that_llm_gpu_baseline.ipynb](name_that_llm_gpu_baseline.ipynb) in Kaggle.

1. Accept the [competition rules](https://www.kaggle.com/competitions/name-that-llm/rules).
2. Attach the **Name That LLM** competition data under Input.
3. Choose a **GPU** accelerator in notebook settings, and keep **Internet off**.
4. Run all cells. The first code cell checks CUDA and performs a GPU matrix multiplication.
5. Confirm the final cell writes `/kaggle/working/submission.parquet`.
6. Save a version with **Save & Run All**, then submit that notebook version.

The notebook uses CPU character TF-IDF extraction and a small PyTorch neural classifier trained on the GPU. It needs no external model weights, packages to install on Kaggle, or API keys. It validates on held-out prompt groups, reports balanced accuracy, selects an epoch count, and retrains on all labeled rows before predicting. It supports both the single interactive test example and batched hidden-test inference.

Settings are in the first code cell. Defaults use 30,000 features, 128 hidden units, batches of 128, and at most 12 validation-training epochs with early stopping. A missing GPU gives a clear error; for an intentional local CPU run, set `REQUIRE_GPU = False`. Only one GPU is used.

For local execution, use an environment with `torch`, `numpy`, `pandas`, `scikit-learn`, `joblib`, `pyarrow`, and `ipython`. Set `DATA_DIR` if the competition CSVs are not in the current folder or its parent. Local outputs go into `baseline_outputs/`.

The [competition overview](https://www.kaggle.com/competitions/name-that-llm/overview/description) specifies balanced accuracy, offline notebook execution, and a Parquet submission containing `id` and `model`. The interactive test file is a small preview; scoring reruns the notebook with the hidden test data. This baseline prioritizes a working pipeline over a strong leaderboard result.

Validated locally on 31 October 2026 using the existing competition CSVs: 2,699 training rows, 540 held-out validation rows, and 3 preview test rows. All code cells ran through training, full-data refitting, saving the model, predicting, and writing/reading the Parquet submission. Held-out balanced accuracy was **0.3019**, with epoch 10 selected. Multi-batch inference, a partial final batch, notebook schema, and the missing-GPU guard also passed. This machine has CPU-only PyTorch, so the local execution explicitly used CPU debugging mode; CUDA execution still needs verification on Kaggle. The delivered notebook defaults to requiring CUDA.
