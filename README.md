# Disaster Decision Support System
B.Tech Final Year Project

## Project Overview
Transformer ensemble for humanitarian tweet classification
using the HumAID dataset (10 categories).

## Three Novelties
1. Weighted Complementarity-Based Ensemble (RoBERTa + DeBERTa + ELECTRA)
2. Adaptive Confidence-Based Selective Prediction (threshold: 0.74)
3. Explainable Classification via Integrated Gradients

## Results
| Model    | Macro F1 | Accuracy |
|----------|----------|----------|
| RoBERTa  | 0.7604   | 0.7832   |
| DeBERTa  | 0.7600   | 0.7817   |
| ELECTRA  | 0.7553   | 0.7782   |
| Ensemble | 0.7690   | 0.7909   |
| Selective (72% coverage) | 0.8385 | — |

## Environment
All training and inference runs on Google Colab (T4 GPU).
Models saved to Google Drive at:
/MyDrive/disaster_project/disaster_models/

## Run Order
1. src/data/prepare_data.py
2. src/training/train_roberta.py (Kaggle)
3. src/training/train_deberta.py (Kaggle)
4. src/training/train_electra.py (Kaggle)
5. src/analysis/disagreement_analysis.py (Colab)
6. src/analysis/ensemble.py (Colab)
7. src/analysis/confidence.py (Colab)
8. src/analysis/explainability.py (Colab, GPU)
9. src/evaluation/evaluation_analysis.py (Colab)
10. src/app/crisis_dashboard.py (Colab, GPU)