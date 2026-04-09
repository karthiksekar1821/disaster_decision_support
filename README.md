# Disaster Decision Support System
B.Tech Final Year Project

## Project Overview
Transformer and hybrid ensemble for humanitarian tweet classification
using the HumAID dataset filtered to 5 classes.

### 5-Class Schema
| Class | Description |
|-------|-------------|
| `infrastructure_and_utility_damage` | Tweets about damaged buildings, roads, utilities |
| `rescue_volunteering_or_donation_effort` | Tweets about rescue, volunteering, or donations |
| `injured_or_dead_people` | Tweets about casualties and injuries |
| `other_relevant_information` | General disaster info, warnings, caution/advice |
| `not_humanitarian` | Non-humanitarian tweets |

## Models (8 Total)

### 6 Transformer Models
- **RoBERTa-base** — Robust pre-training with dynamic masking
- **DeBERTa-base** — Disentangled attention with enhanced decoding
- **ELECTRA-base** — Replaced token detection pre-training
- **BERT-base-uncased** — Original bidirectional transformer
- **BERTweet** — RoBERTa pre-trained on 850M English tweets (domain-specific)
- **XtremeDistil-L6-H256** — Distilled BERT for efficiency

### 2 Non-Transformer Models
- **TextCNN** — CNN with filter sizes [2,3,4] × 128 filters, random embeddings
- **BiLSTM + Attention** — Bidirectional LSTM with dot-product attention, GloVe 100d

## Three Novelties

### Novelty 1: Context-Conditioned Dynamic Ensemble
Meta-learner (MLP) produces **per-tweet ensemble weights** conditioned on:
- Individual model softmax outputs (8 models)
- Tweet linguistic profile (word count, disaster keywords, urgency)
- Tweet style classification (URGENT/FORMAL/EYEWITNESS/INFORMATIONAL)
- Per-model confidence gaps (max_prob - second_max_prob)

### Novelty 2: Class-Adaptive Confidence Thresholds
Per-class abstention thresholds swept on the validation set.

### Novelty 3: Decision-Influencing Explainability
Integrated Gradients attributions flag unreliable high-confidence predictions.

## Architecture
- **Training**: Class-weighted cross-entropy, seed=42
- **Ensemble**: 8-model meta-learner with style features (Novelty 1)
- **Selective Prediction**: Per-class thresholds (Novelty 2) + attribution check (Novelty 3)

## Environment
All training runs on Kaggle (T4 GPU). See `KAGGLE_RUN_GUIDE.md`.

## Run Order
1. `src/data/prepare_data.py` — preprocess & filter to 5 classes
2. `src/training/train_model.py --model roberta` — train each transformer (seed=42)
3. `src/analysis/cnn_classifier.py` — train TextCNN
4. `src/analysis/bilstm_classifier.py` — train BiLSTM
5. `src/analysis/model_characterisation.py` — tweet style analysis
6. `src/analysis/dynamic_ensemble.py` — train meta-learner (8 models)
7. `src/analysis/adaptive_confidence.py` — per-class thresholds
8. `src/analysis/attribution_filter.py` — attribution reliability check
9. `src/evaluation/evaluation.py` — comprehensive evaluation
10. `src/app/crisis_dashboard.py` — Gradio dashboard

## Dependencies
```
pip install -r requirements.txt
```