# Kaggle Run Guide — Disaster Decision Support Pipeline

> **Every code cell imports and calls functions from the existing project files.**
> No code is duplicated or rewritten — all logic lives in `src/`.

---

## Prerequisites

1. **Create a Kaggle notebook** with **GPU T4 x2** accelerator
2. **Add your dataset**: `karthiksekar1821/humaid` — contains the raw `train.parquet`, `val.parquet`, `test.parquet`
3. **Add saved notebook output as input**:
   - Click **Add Input** → **Notebook Output**
   - Search for `main-project` (your training notebook)
   - This mounts the trained model outputs at `/kaggle/input/notebooks/karthiksekar1821/main-project/`
4. The notebook will clone the GitHub repo automatically in Cell 1

---

## Important Path Information

| Variable | Path | Contents |
|----------|------|----------|
| `BASE_INPUT` | `/kaggle/input/notebooks/karthiksekar1821/main-project` | Saved notebook output root |
| `OUTPUT_DIR` | `{BASE_INPUT}/output` | All trained model checkpoints and predictions |
| `DATA_DIR` | `/kaggle/input/datasets/karthiksekar1821/humaid` | Raw HumAID parquet files |
| `PROCESSED_DIR` | `{BASE_INPUT}/disaster_decision_support/data/processed` | Processed parquets (cleaned text, 5 classes, label mapping) |
| `RESULTS_DIR` | `/kaggle/working/results` | All evaluation outputs (writable) |

### Model Checkpoint Paths

Transformer models are saved at:
```
{OUTPUT_DIR}/{model_key}/seed_42/best_model/
```
or (for CNN/BiLSTM):
```
{OUTPUT_DIR}/{model_key}/best_model/
```

The `load_predictions` function in Cell 3 automatically checks both `seed_42` and non-seed paths.

---

## Notebook Structure — 9 Cells

### Cell 1 — Clone Repo & Install Dependencies

Clones the GitHub repository and installs `gradio` and `joblib`. Does **NOT** install `captum` here (installed in Cell 7 to avoid numpy binary incompatibility).

**Must always run first.**

### Cell 2 — Configure All Paths

Defines `BASE_INPUT`, `OUTPUT_DIR`, `DATA_DIR`, `PROCESSED_DIR`, `RESULTS_DIR` and adds all `src/` directories to `sys.path`. Creates `RESULTS_DIR`.

**Must always run second.**

### Cell 3 — Load All Predictions + Train Texts

- Defines `load_predictions()` which checks both `seed_42` and non-seed paths
- Loads val and test predictions from all 8 models
- Loads train/val/test texts from **processed** parquet files (cleaned text, 5 classes only)
- Defines `class_names`, `train_labels`, `val_labels`, `test_labels`

### Cell 4 — Model Characterisation (Tweet Styles)

- Classifies all val/test tweets into 4 styles (URGENT, FORMAL, EYEWITNESS, INFORMATIONAL)
- Computes per-model per-style Macro F1 performance matrix
- Saves style heatmap to `RESULTS_DIR/evaluation_results/`

### Cell 5 — Dynamic Ensemble (Novelty 1)

- Trains MLP meta-learner on validation set using softmax probabilities + context features + style features + confidence gaps
- Predicts on test set
- Evaluates and reports ensemble Macro F1
- Saves meta-learner, scaler, and predictions to `RESULTS_DIR/ensemble/`

### Cell 6 — Novelty 2: Class-Adaptive Confidence Thresholds

- Sweeps per-class thresholds on validation set
- Applies thresholds to test set for selective prediction
- Reports coverage, selective Macro F1, and F1 improvement
- Saves results to `RESULTS_DIR/confidence_results/`

### Cell 7 — Novelty 3: Attribution Filter

- **Installs captum at the very top** (must stay here permanently)
- Loads RoBERTa model for Integrated Gradients attribution computation
- Computes attributions for up to 500 test samples
- Flags unreliable predictions (high confidence but irrelevant attributions)
- Combines with Novelty 2 for dual-signal abstention

### Cell 8 — Comprehensive Evaluation Report

- Runs full evaluation: individual model results, baselines (TF-IDF+SVM, majority class), ensemble, selective prediction, McNemar's test
- Generates all plots: confusion matrices, per-class F1 bars, macro F1 summary, training curves, calibration diagram, confidence distributions, style heatmap
- All PNG plots saved to `RESULTS_DIR/evaluation_results/`
- All JSON results saved to `RESULTS_DIR/`

### Cell 9 — Launch Gradio Dashboard

- Creates and launches the crisis dashboard using all loaded models
- Uses models from `OUTPUT_DIR`, label mapping from `PROCESSED_DIR`, and ensemble artifacts from `RESULTS_DIR/ensemble/`
- Dashboard provides real-time tweet classification with all 8 models

---

## Execution Order

1. **Cell 1** — Clone + install (run once)
2. **Cell 2** — Configure paths (run once)
3. **Cells 3–8** — Run in order, skip none
4. **Cell 9** — Launch dashboard (optional, run last)

> **Important:** Do NOT skip cells or run them out of order. Each cell depends on variables defined in previous cells.

---

## Interpreting Results

### Key Metrics

| Metric | Expected Value | Meaning |
|--------|---------------|---------|
| Ensemble Macro F1 | ~0.8166 | Dynamic ensemble performance on test set |
| Selective Macro F1 | ~0.8773 | F1 on accepted predictions (Novelty 2) |
| Selective Coverage | ~61.6% | Percentage of predictions accepted (rest abstained) |
| Combined Macro F1 | ~0.8771 | F1 with both confidence + attribution filters |
| Combined Coverage | ~59.3% | Coverage with dual-signal abstention |
| TF-IDF+SVM Baseline | ~0.76 | Traditional ML baseline for comparison |

### Plots Generated (in `RESULTS_DIR/evaluation_results/`)

| File | Description |
|------|-------------|
| `macro_f1_summary.png` | Bar chart comparing all models + baselines + ensemble |
| `per_class_f1_comparison.png` | Per-class F1 across all models |
| `confusion_matrix_*.png` | Confusion matrices for each model |
| `ensemble_confusion_matrix.png` | Ensemble confusion matrix |
| `calibration_diagram.png` | Reliability/calibration plot |
| `confidence_distributions.png` | Per-model confidence histograms |
| `model_style_heatmap.png` | Model × tweet style performance matrix |
| `training_curves_*.png` | Training loss/F1 curves per model |

---

## Common Errors and Fixes

### Error: `FileNotFoundError: No predictions found for {model_key}`

**Cause:** The model predictions are not in the expected path.

**Fix:** Check that you've added the correct notebook output as input. The predictions should be at:
```
/kaggle/input/notebooks/karthiksekar1821/main-project/output/{model_key}/seed_42/predictions/
```
or:
```
/kaggle/input/notebooks/karthiksekar1821/main-project/output/{model_key}/predictions/
```

### Error: `ModuleNotFoundError: No module named 'xxx'`

**Cause:** Cell 2 was not run, or `sys.path` was not configured.

**Fix:** Run Cell 2 first — it adds all `src/` directories to `sys.path`.

### Error: `captum` import fails

**Cause:** captum is not installed yet (it's installed in Cell 7).

**Fix:** Make sure to run Cell 7 before any cell that uses captum. Cell 7 installs it at the top.

### Error: SVM baseline returning very low F1 (~0.20)

**Cause:** This was a known bug where uncleaned raw texts were passed to the SVM. It has been fixed — the SVM function now internally cleans texts.

**Fix:** Make sure you're using the latest version of `evaluation.py` from the repository.

### Error: `No models produced predictions` in dashboard

**Cause:** Model paths are incorrect or models failed to load.

**Fix:** Check the startup print statements in Cell 9 — they show all paths being searched and which models loaded successfully.

---

## File Reference

| Step | Project File Used | Location |
|------|------------------|----------|
| Data Prep | `prepare_data.py`, `split.py`, `label_utils.py`, `preprocessing.py` | `src/data/` |
| Training | `train_model.py`, `config.py` | `src/training/` |
| CNN | `cnn_classifier.py` | `src/analysis/` |
| BiLSTM | `bilstm_classifier.py` | `src/analysis/` |
| Characterisation | `model_characterisation.py`, `disaster_vocab.py` | `src/analysis/` |
| Ensemble | `dynamic_ensemble.py`, `context_features.py` | `src/analysis/` |
| Confidence | `adaptive_confidence.py` | `src/analysis/` |
| Attribution | `attribution_filter.py`, `disaster_vocab.py` | `src/analysis/` |
| Evaluation | `evaluation.py` | `src/evaluation/` |
| Dashboard | `crisis_dashboard.py` | `src/app/` |
