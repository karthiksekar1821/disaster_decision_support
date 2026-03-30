# PROJECT DOCUMENTATION
## Disaster Decision Support System — B.Tech Final Year Project

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [File-by-File Documentation](#file-by-file-documentation)
3. [End-to-End Pipeline](#end-to-end-pipeline)
4. [Key Concepts](#key-concepts)
5. [Results Summary](#results-summary)
6. [Design Decisions](#design-decisions)
7. [Review Questions](#review-questions)

---

## Project Overview

This project classifies disaster-related tweets into **5 humanitarian categories** using the **HumAID dataset**. It employs an ensemble of **8 models** (6 transformers + 2 non-transformer) combined through a **Context-Conditioned Dynamic Ensemble** with **class-adaptive confidence thresholds** and **attribution-based abstention**.

### 5-Class Schema

| ID | Class Name | Description |
|----|-----------|-------------|
| 0  | `infrastructure_and_utility_damage` | Damage to buildings, roads, bridges, utilities |
| 1  | `injured_or_dead_people` | Casualties, injuries, missing persons |
| 2  | `not_humanitarian` | Non-humanitarian content |
| 3  | `other_relevant_information` | General disaster info, warnings, updates |
| 4  | `rescue_volunteering_or_donation_effort` | Rescue ops, donations, volunteering |

### Models

| Model | Type | HuggingFace String | Key Characteristic |
|-------|------|-------------------|-------------------|
| RoBERTa | Transformer | `roberta-base` | Dynamic masking, robust pre-training |
| DeBERTa | Transformer | `microsoft/deberta-base` | Disentangled attention mechanism |
| ELECTRA | Transformer | `google/electra-base-discriminator` | Replaced token detection |
| BERT | Transformer | `bert-base-uncased` | Original bidirectional transformer |
| XLNet | Transformer | `xlnet-base-cased` | Autoregressive, left padding required |
| XtremeDistil | Transformer | `microsoft/xtremedistil-l6-h256-uncased` | 6-layer distilled, efficient |
| TextCNN | Non-transformer | N/A | Conv filters [2,3,4] × 128, random embeddings |
| BiLSTM | Non-transformer | N/A | 256 hidden, dot-product attention, GloVe 100d |

### Three Novelties

1. **Context-Conditioned Dynamic Ensemble** — Meta-learner produces per-tweet weights conditioned on model outputs, linguistic features, tweet style, and confidence gaps
2. **Class-Adaptive Confidence Thresholds** — Per-class abstention thresholds swept on validation data
3. **Decision-Influencing Explainability** — Integrated Gradients flag unreliable predictions despite high softmax confidence

---

## File-by-File Documentation

### `src/data/preprocessing.py`
- **Purpose:** Text cleaning functions for tweets
- **Functions:**
  - `clean_text(text)` — removes URLs, mentions, hashtag symbols, normalises whitespace
  - `is_low_information(text, min_tokens=3)` — flags tweets with fewer than `min_tokens` words
- **Dependencies:** `re`

### `src/data/split.py`
- **Purpose:** Load raw HumAID data and preprocess splits
- **Functions:**
  - `load_local_humaid()` — loads train/val/test parquets, renames `class_label` → `label`
  - `preprocess_split(split_dataset)` — cleans text, flags low-info, removes empty
  - `save_processed_splits(dataset)` — saves to `data/processed/`
- **Dependencies:** `datasets`, `preprocessing`

### `src/data/label_utils.py`
- **Purpose:** Label mapping and class weight computation
- **Key variables:** `TARGET_CLASSES` — 5 sorted class names
- **Functions:**
  - `filter_to_target_classes(dataset_split)` — drops non-target labels
  - `create_label_mapping()` — returns `label2id`, `id2label`
  - `encode_labels(split_dataset, label2id)` — maps string labels to integers
  - `compute_and_save_class_weights(labels, num_classes)` — balanced class weights
- **Dependencies:** `sklearn.utils.class_weight`, `torch`

### `src/data/prepare_data.py`
- **Purpose:** End-to-end data preparation script
- **Pipeline:** Load → clean → deduplicate → filter to 5 classes → encode → compute weights → save
- **Dependencies:** `split`, `label_utils`

---

### `src/training/config.py`
- **Purpose:** Centralised training configuration
- **Key variables:**
  - `NUM_LABELS = 5`
  - `SEED = 42` — single seed for all training
  - `TRAINING_ARGS` — shared hyperparameters (5 epochs, LR 2e-5, batch 16, etc.)
  - `MODEL_CONFIGS` — 6 transformer model entries with `model_name` and `max_length`
  - Data paths: `TRAIN_FILE`, `VAL_FILE`, `TEST_FILE`, `LABEL_MAPPING_FILE`

### `src/training/train_model.py`
- **Purpose:** Unified training script for all 6 transformer models
- **Classes:**
  - `WeightedTrainer(Trainer)` — overrides `compute_loss` with class-weighted cross-entropy
  - `TrainingLossCallback(TrainerCallback)` — captures per-epoch train loss, val loss, val Macro F1; saves `training_history.json`
- **Functions:**
  - `compute_metrics(eval_pred)` — returns Macro F1 and accuracy
  - `load_data()` / `load_label_mapping()` — load parquets and label mappings
  - `tokenize_dataset(dataset, tokenizer, max_length, model_key)` — tokenises with XLNet-specific left padding and token_type_ids removal
  - `load_best_hyperparams(model_key)` — loads Optuna-tuned params if `best_hyperparams_{model_key}.json` exists
  - `train_single_model(model_key, seed, output_dir, ...)` — trains one model, saves predictions as `val_predictions.npz` / `test_predictions.npz`

**XLNet Special Handling:**
- `tokenizer.padding_side = "left"` set before tokenisation
- `token_type_ids` removed from tokenised output (XLNet handles them differently)

**Training History Format** (`training_history.json`):
```json
{
  "epoch": [1, 2, 3, 4, 5],
  "train_loss": [0.85, 0.62, 0.51, 0.44, 0.40],
  "val_loss": [0.71, 0.58, 0.53, 0.50, 0.49],
  "val_macro_f1": [0.72, 0.78, 0.81, 0.83, 0.84]
}
```

### `src/training/hyperparameter_tuning.py`
- **Purpose:** Optuna-based Bayesian hyperparameter optimisation per model
- **Search space per model:**
  - `learning_rate`: log-uniform in [1e-5, 5e-5]
  - `warmup_ratio`: uniform in [0.0, 0.2]
  - `weight_decay`: uniform in [0.0, 0.1]
  - `per_device_train_batch_size`: categorical {8, 16, 32}
- **Design:**
  - 20 trials per model using TPE sampler
  - Each trial trains on 30% of training data for 3 epochs
  - Optimises validation Macro F1
  - Saves best params to `data/processed/best_hyperparams_{model_key}.json`
  - Trial checkpoints are cleaned up after each trial to save disk space
- **Functions:**
  - `create_objective(model_key, ...)` — creates the Optuna objective function
  - `run_hyperparameter_tuning(model_key, output_dir, n_trials)` — callable from notebooks
  - `run_all_hyperparameter_tuning(output_dir)` — tunes all models sequentially

---

### `src/analysis/disaster_vocab.py`
- **Purpose:** Curated per-class disaster vocabulary for attribution verification
- **Key variables:**
  - `DISASTER_VOCAB` — dict of class_name → set of relevant tokens
  - `IRRELEVANT_TOKENS` — stopwords, punctuation, social media artifacts, special tokens
- **Functions:**
  - `get_disaster_vocab_for_class(class_name)` — returns vocab set for a class
  - `is_disaster_relevant(token, class_name)` — checks if token is relevant
  - `is_irrelevant(token)` — checks if token is a stopword

### `src/analysis/context_features.py`
- **Purpose:** Per-tweet linguistic feature extraction for the meta-learner
- **Features extracted:** char_count, word_count, avg_word_length, exclamation_count, question_count, uppercase_ratio, has_numbers, disaster_keyword_count/ratio, urgency_keyword_count
- **Functions:**
  - `extract_features(tweet_text)` — returns dict of features for one tweet
  - `extract_features_batch(tweet_texts)` — returns numpy matrix (N, num_features)

### `src/analysis/model_characterisation.py`
- **Purpose:** Explainability-driven model characterisation with 4 tweet styles
- **Tweet Style Categories:**

| Style | Signals | Example |
|-------|---------|---------|
| URGENT | All-caps, exclamation marks, SOS/HELP/TRAPPED | "HELP! Building collapsed!" |
| FORMAL | Organisation names, percentages, third-person reporting | "FEMA reports 30% of infrastructure damaged" |
| EYEWITNESS | First-person pronouns, present tense, "I see", "near me" | "I can see flooding from my window right now" |
| INFORMATIONAL | Past tense, statistics, factual reporting | "Hurricane was downgraded to Category 2" |

- **Functions:**
  - `classify_tweet_style(text)` — returns best style + scores
  - `classify_tweets_batch(texts)` — batch classification
  - `style_to_onehot(style_labels)` — converts to one-hot (N, 4)
  - `compute_style_performance_matrix(...)` — Macro F1 per model per style → 6×4 matrix
  - `plot_style_performance_heatmap(...)` — saves heatmap PNG
  - `compute_attribution_profiles(...)` — average IG attribution per style-vocabulary category
  - `run_attribution_style_verification(...)` — runs attribution profiles for all models
  - `plot_attribution_profiles(...)` — saves grouped bar chart

### `src/analysis/dynamic_ensemble.py`
- **Purpose:** Context-Conditioned Dynamic Ensemble (Novelty 1)
- **Meta-features (per sample):**
  - Model softmax probabilities: 8 models × 5 classes = 40 features
  - Context features: 10 linguistic features
  - Confidence gaps: max_prob − second_max_prob per model = 8 features
  - Tweet style one-hot: 4 features
  - **Total: ~62 features**
- **Meta-learner:** MLP with hidden layers (128, 64), trained on validation set
- **Functions:**
  - `compute_confidence_gaps(model_probs_list)` — returns (N, M) gap matrix
  - `build_meta_features(model_probs_list, tweet_texts, style_labels)` — concatenates all feature groups
  - `train_meta_learner(...)` — trains and returns sklearn model + scaler
  - `predict_with_dynamic_ensemble(...)` — returns ensemble probs and preds
  - `evaluate_dynamic_ensemble(...)` — prints Macro F1 and classification report
  - `save_ensemble_artifacts(...)` — saves meta-learner, scaler, predictions

### `src/analysis/adaptive_confidence.py`
- **Purpose:** Class-Adaptive Confidence Thresholds (Novelty 2)
- **Method:** For each class, sweep threshold from 0 to 1 on validation set; choose threshold maximising α·F1 + (1−α)·coverage
- **Functions:**
  - `sweep_per_class_thresholds(probs, preds, labels, ...)` — returns per-class thresholds
  - `apply_per_class_thresholds(probs, preds, ...)` — returns accepted mask
  - `evaluate_selective_prediction(...)` — reports coverage, F1, accuracy

### `src/analysis/attribution_filter.py`
- **Purpose:** Decision-Influencing Explainability via Attribution Filtering (Novelty 3)
- **Supports:** RoBERTa, DeBERTa, ELECTRA, BERT, XLNet, XtremeDistil (embedding layer auto-detection)
- **Functions:**
  - `compute_attributions_for_batch(...)` — LayerIntegratedGradients on embedding layer
  - `compute_disaster_relevance_score(...)` — fraction of top-K tokens that are disaster-relevant
  - `flag_unreliable_predictions(...)` — flags high-confidence but irrelevant-attribution predictions
  - `combined_abstention(...)` — combines confidence + attribution signals

### `src/analysis/cnn_classifier.py`
- **Purpose:** TextCNN ensemble member
- **Architecture:** Embedding(vocab, 128) → Conv1d[2,3,4]×128 → MaxPool → Dropout(0.5) → Linear(384, 5)
- **Training:** Adam, ReduceLROnPlateau, 10 epochs, class-weighted CE, seed=42
- **Functions:**
  - `train_cnn(...)` — trains and saves predictions
  - `predict_cnn(model, vocab, texts)` — inference

### `src/analysis/bilstm_classifier.py`
- **Purpose:** BiLSTM with attention ensemble member
- **Architecture:** GloVe 100d → BiLSTM(256) → Dot-Product Attention → Linear(512, 5)
- **GloVe loading:** Auto-downloads `glove.6B.zip` from Stanford NLP, caches in `~/.cache/glove/`
- **Training:** Adam, gradient clipping (max_norm=1.0), 15 epochs, class-weighted CE, seed=42
- **Functions:**
  - `train_bilstm(...)` — trains and saves predictions
  - `predict_bilstm(model, vocab, texts)` — inference

---

### `src/evaluation/evaluation.py`
- **Purpose:** Comprehensive evaluation across all 8 models + ensemble
- **Produces:**
  - Per-model confusion matrices (normalised)
  - Per-class F1 comparison bar chart (all 8 models + ensemble)
  - Macro F1 summary (baselines + all models + ensemble)
  - Training curves (train loss, val loss, val Macro F1 per epoch)
  - Calibration diagram (reliability plot)
  - Confidence distribution histograms
  - Model × style performance heatmap
  - Consolidated results table
  - McNemar's test for pairwise statistical significance
  - TF-IDF + SVM and majority class baselines

### `src/app/crisis_dashboard.py`
- **Purpose:** Gradio-based crisis dashboard for real-time tweet classification
- **Features:**
  - Loads all 8 models (6 transformers + CNN + BiLSTM)
  - Runs dynamic ensemble with style features
  - Displays: predicted category, confidence, tweet style, per-model predictions
  - Example tweets for each style category
- **Functions:**
  - `load_all_models(model_dir, device)` — loads all available models
  - `classify_tweet(text, models, tokenizers, meta_learner, scaler, class_names, device)` — full pipeline
  - `create_dashboard(model_dir, ...)` — creates and returns Gradio Blocks app

---

## End-to-End Pipeline

```
1. DATA PREPARATION (src/data/prepare_data.py)
   Raw HumAID Parquets → Clean text → Deduplicate → Filter to 5 classes
   → Encode labels → Compute class weights → Save processed parquets

2. HYPERPARAMETER TUNING (src/training/hyperparameter_tuning.py) [Optional]
   For each model: 20 Optuna trials on 30% data × 3 epochs
   → Save best_hyperparams_{model_key}.json

3. TRANSFORMER TRAINING (src/training/train_model.py)
   For each of 6 transformers (seed=42):
     Load data → Tokenize (XLNet: left pad, no token_type_ids)
     → Train with WeightedTrainer + TrainingLossCallback
     → Save val_predictions.npz, test_predictions.npz, training_history.json, best_model/

4. NON-TRANSFORMER TRAINING
   CNN: src/analysis/cnn_classifier.py (random embeddings, Conv1d filters)
   BiLSTM: src/analysis/bilstm_classifier.py (GloVe 100d, dot-product attention)
   → Save val_predictions.npz, test_predictions.npz

5. MODEL CHARACTERISATION (src/analysis/model_characterisation.py)
   Classify tweet styles → Compute per-model per-style F1 matrix
   → Attribution-based style verification

6. DYNAMIC ENSEMBLE (src/analysis/dynamic_ensemble.py)
   Build meta-features: 8×5 probs + context + confidence gaps + style one-hot
   → Train MLP meta-learner on validation set → Predict on test set

7. SELECTIVE PREDICTION
   Novelty 2: sweep per-class thresholds on validation
   Novelty 3: flag unreliable predictions via attribution analysis
   Combined: accept only if BOTH signals agree

8. EVALUATION (src/evaluation/evaluation.py)
   All plots, tables, statistical tests, training curves

9. DASHBOARD (src/app/crisis_dashboard.py)
   Gradio app for real-time classification with all 8 models
```

---

## Key Concepts

### Why 8 Models?
- **RoBERTa** — Strong general-purpose performance, dynamic masking improves robustness
- **DeBERTa** — Disentangled attention captures fine-grained position-dependent semantics
- **ELECTRA** — Replaced token detection is sample-efficient, good for noisy social media
- **BERT** — Baseline bidirectional transformer, established benchmark
- **XLNet** — Autoregressive pre-training captures longer-range dependencies
- **XtremeDistil** — 6-layer distilled model tests whether smaller models suffice
- **TextCNN** — Fast, strong at local n-gram patterns, complements transformer global attention
- **BiLSTM** — Captures sequential dependencies differently from transformers; GloVe embeddings provide pre-trained word semantics without fine-tuning

### Why Tweet Style Classification?
Different models may excel at different tweet styles. The style classification provides:
1. **Interpretable routing** — the meta-learner can learn to trust different models for different styles
2. **Evidence for model selection** — the style performance heatmap shows which model handles which style best
3. **Explainability** — attribution profiles verify that models attend to style-appropriate vocabulary

### Why Confidence Gap Features?
The gap between the highest and second-highest softmax probability indicates how "decisive" a model is. A large gap suggests the model is confident and discriminating; a small gap suggests uncertainty. Feeding this per-model to the meta-learner allows it to weight decisive models more heavily.

### Class-Weighted Training
Disaster datasets are inherently imbalanced (e.g., `not_humanitarian` dominates). Class weights computed via sklearn's `balanced` strategy ensure minority classes (e.g., `injured_or_dead_people`) receive higher loss weight during training.

---

## Results Summary

> **Note:** Results below are placeholders. Run the full pipeline on Kaggle to fill in actual values.

### Individual Model Performance (Test Set)

| Model | Macro F1 | Accuracy |
|-------|----------|----------|
| RoBERTa | — | — |
| DeBERTa | — | — |
| ELECTRA | — | — |
| BERT | — | — |
| XLNet | — | — |
| XtremeDistil | — | — |
| CNN | — | — |
| BiLSTM | — | — |
| **Dynamic Ensemble** | **—** | **—** |

### Selective Prediction Results

| Metric | Without Selection | With Novelty 2 | With Novelty 2+3 |
|--------|------------------|----------------|------------------|
| Coverage | 100% | — | — |
| Macro F1 | — | — | — |
| Accuracy | — | — | — |

---

## Design Decisions

### Why These 6 Transformer Models?
1. **RoBERTa/DeBERTa/ELECTRA** — three different pre-training objectives provide complementary representations
2. **BERT** — canonical baseline expected by reviewers
3. **XLNet** — explores autoregressive pre-training which handles word order differently
4. **XtremeDistil** — tests if a 6-layer distilled model can achieve competitive performance with much lower compute

### Why Single Seed Instead of Multi-Seed?
The panel requested removing multi-seed variance reporting. With 8 models, the computational cost of 3 seeds × 8 models = 24 training runs is prohibitive. Single seed (42) is standard in the literature and sufficient for comparing model architectures.

### Why Optuna for Hyperparameter Tuning?
- **Bayesian optimisation** (TPE) is more sample-efficient than grid/random search
- **30% data + 3 epochs** keeps trial time manageable (~5-10 min per trial on T4)
- **20 trials** balances exploration vs. compute budget
- Per-model tuning allows each architecture to find its optimal configuration

### Why TextCNN and BiLSTM?
- Non-transformer baselines provide diversity in the ensemble
- CNN captures local n-gram patterns; BiLSTM captures sequential semantics
- If transformers overfit to similar patterns, non-transformers provide an alternative signal
- Demonstrates the ensemble can integrate heterogeneous model families

### Why Tweet Style Features in the Ensemble?
- The style classification provides human-interpretable routing signals
- The meta-learner can learn patterns like "trust ELECTRA more for URGENT tweets"
- This creates a direct link between explainability analysis and ensemble decision-making

---

## Review Questions

### Data & Preprocessing
1. **Why 5 classes instead of 10?** — Reduces class ambiguity, improves per-class sample sizes, and focuses on the most actionable humanitarian categories.
2. **How are duplicates handled?** — Majority vote for conflicting labels, then deduplicate.
3. **What text cleaning is applied?** — URL removal, mention removal, hashtag symbol removal, whitespace normalisation.

### Models & Training
4. **Why use class-weighted loss?** — Disaster datasets are imbalanced; `not_humanitarian` dominates. Balanced weights ensure minority classes receive proportional attention.
5. **How does XLNet differ from other transformers?** — Uses left padding (all others use right padding) and does not use token_type_ids in the standard way.
6. **What is XtremeDistil and why include it?** — A 6-layer distilled BERT variant from Microsoft. Tests whether model compression degrades performance on this task.
7. **How does hyperparameter tuning work?** — Optuna samples from defined ranges using Bayesian optimisation (TPE), trains on 30% data for 3 epochs, and selects the configuration maximising validation Macro F1.
8. **Why use training_history.json?** — Records per-epoch loss curves for visualisation, helping detect overfitting and compare convergence patterns across models.

### Ensemble & Novelties
9. **What features does the meta-learner use?** — 40 softmax probabilities (8×5), 10 linguistic features, 8 confidence gaps, 4 style one-hot features = ~62 total.
10. **Why train the meta-learner on validation data, not training data?** — Prevents the meta-learner from memorising training set patterns; forces it to learn genuine cross-model complementarity.
11. **What is the confidence gap feature?** — max_prob − second_max_prob per model. A large gap indicates a decisive model; a small gap indicates uncertainty.
12. **How do per-class thresholds differ from a global threshold?** — Different classes have different difficulty levels. A global threshold would over-abstain on easy classes and under-abstain on hard ones.
13. **What does the attribution filter catch?** — High-confidence predictions where the most-attributed tokens are stopwords/irrelevant, suggesting the model is confident for the wrong reasons.

### Explainability & Characterisation
14. **What are the 4 tweet styles?** — URGENT (SOS, all-caps), FORMAL (organisations, statistics), EYEWITNESS (first person, present tense), INFORMATIONAL (factual, past tense).
15. **How does tweet style classification work?** — Rule-based scoring using regex patterns and keyword sets. Each tweet gets 4 scores; the highest determines the style.
16. **What does the style performance heatmap show?** — A 6×4 matrix of Macro F1 per transformer per tweet style, revealing which models excel at which styles.
17. **What are attribution profiles?** — Average Integrated Gradients attribution scores grouped by vocabulary category (urgent/formal/eyewitness/stopword) per model per style.

### CNN & BiLSTM
18. **Why random embeddings for CNN instead of pre-trained?** — Keeps the CNN lightweight and tests whether simple n-gram patterns suffice without pre-trained semantics.
19. **Why GloVe for BiLSTM?** — GloVe provides static word semantics that complement the BiLSTM's sequential modelling, without requiring fine-tuning a large model.
20. **How do CNN/BiLSTM predictions integrate with the ensemble?** — Saved in identical `.npz` format; their softmax probabilities feed into the meta-learner alongside transformer outputs.

### Evaluation
21. **What baselines are compared?** — Majority class and TF-IDF + LinearSVC.
22. **How is statistical significance tested?** — McNemar's test with continuity correction, comparing pairwise models and ensemble vs. best individual.
23. **What plots are generated?** — Confusion matrices, per-class F1 bars, Macro F1 summary, training curves, calibration diagrams, confidence distributions, style performance heatmap.
