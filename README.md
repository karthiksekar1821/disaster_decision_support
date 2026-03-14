# Disaster Decision Support System
B.Tech Final Year Project

## Project Overview
Transformer ensemble for humanitarian tweet classification
using the HumAID dataset.

### 5-Class Schema
| Class | Description |
|-------|-------------|
| `infrastructure_and_utility_damage` | Tweets about damaged buildings, roads, utilities |
| `rescue_volunteering_or_donation_effort` | Tweets about rescue, volunteering, or donations |
| `affected_individuals` | Merged from: injured/dead people, displaced/evacuated, missing/found |
| `other_relevant_information` | General disaster info, warnings, caution/advice, requests, sympathy |
| `not_humanitarian` | Non-humanitarian tweets |

## Three Novelties

### Novelty 1: Context-Conditioned Dynamic Ensemble
Instead of fixed ensemble weights, a meta-learner (MLP) produces
**per-tweet ensemble weights** conditioned on each tweet's linguistic
profile (word count, disaster keyword overlap, urgency markers, etc.)
and the individual model softmax outputs. Trained on the validation set.

### Novelty 2: Class-Adaptive Confidence Thresholds
Per-class abstention thresholds swept independently on the validation set.
Each class gets its own optimal threshold based on its difficulty.
"Adaptive" = threshold adapts to each class, not a global constant.

### Novelty 3: Decision-Influencing Explainability
Integrated Gradients attributions are used as a **decision signal**:
if a prediction is high-confidence but the top-attributed tokens are
stopwords (not disaster vocabulary), the prediction is flagged as
unreliable. This creates a second abstention signal independent of
softmax confidence, combined with Novelty 2 for dual-signal prediction.

## Architecture
- **Models**: RoBERTa-base, DeBERTa-v3-base, ELECTRA-base
- **Training**: Class-weighted cross-entropy, multi-seed (3 seeds), consistent hyperparameters
- **Ensemble**: Meta-learner on validation set (Novelty 1)
- **Selective Prediction**: Per-class thresholds (Novelty 2) + attribution check (Novelty 3)

## Environment
All training and inference runs on Google Colab (T4 GPU) / Kaggle.
Models saved to Google Drive at:
/MyDrive/disaster_project/disaster_models/

## Run Order
1. `src/data/prepare_data.py` — preprocess & merge to 5 classes
2. `src/training/train_model.py --model roberta` — train RoBERTa (3 seeds)
3. `src/training/train_model.py --model deberta` — train DeBERTa (3 seeds)
4. `src/training/train_model.py --model electra` — train ELECTRA (3 seeds)
5. `src/analysis/dynamic_ensemble.py` — Novelty 1: train meta-learner
6. `src/analysis/adaptive_confidence.py` — Novelty 2: per-class thresholds
7. `src/analysis/attribution_filter.py` — Novelty 3: attribution check
8. `src/evaluation/evaluation.py` — comprehensive evaluation

## Dependencies
```
pip install -r requirements.txt
```