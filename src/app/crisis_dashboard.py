"""
Crisis Dashboard — Standalone Python script (Addition 7).

Provides the inference pipeline and Gradio interface for the
disaster tweet classification system using all 8 models.

Usage (from Kaggle/Colab):
    from crisis_dashboard import create_dashboard, classify_tweet
    app = create_dashboard(model_dir="/path/to/models")
    app.launch()
"""

import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ── Model Loading ────────────────────────────────────────────────────────────

TRANSFORMER_KEYS = ["roberta", "deberta", "electra", "bert", "bertweet", "xtremedistil"]
ALL_MODEL_KEYS = TRANSFORMER_KEYS + ["cnn", "bilstm"]

DISPLAY_NAMES = {
    "roberta": "RoBERTa",
    "deberta": "DeBERTa",
    "electra": "ELECTRA",
    "bert": "BERT",
    "bertweet": "BERTweet",
    "xtremedistil": "XtremeDistil",
    "cnn": "CNN",
    "bilstm": "BiLSTM",
}


def load_transformer_model(model_dir, model_key, device="cpu"):
    """Load a trained transformer model and tokenizer."""
    model_path = os.path.join(model_dir, model_key, "best_model")
    if not os.path.exists(model_path):
        print(f"  Warning: {model_path} not found, skipping {model_key}")
        return None, None

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    return model, tokenizer


def load_all_models(model_dir, device="cpu"):
    """Load all available models."""
    models = {}
    tokenizers = {}

    # Load transformer models
    for key in TRANSFORMER_KEYS:
        model, tokenizer = load_transformer_model(model_dir, key, device)
        if model is not None:
            models[key] = model
            tokenizers[key] = tokenizer
            print(f"  Loaded {DISPLAY_NAMES[key]}")

    # Load CNN and BiLSTM if available
    cnn_model_path = os.path.join(model_dir, "cnn", "best_model", "model.pt")
    if os.path.exists(cnn_model_path):
        try:
            from cnn_classifier import TextCNN
            vocab = torch.load(os.path.join(model_dir, "cnn", "best_model", "vocab.pt"))
            cnn_model = TextCNN(vocab_size=len(vocab), num_classes=5)
            cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device))
            cnn_model.to(device)
            cnn_model.eval()
            models["cnn"] = cnn_model
            tokenizers["cnn"] = vocab  # vocab serves as tokenizer for CNN
            print(f"  Loaded CNN")
        except Exception as e:
            print(f"  Warning: Could not load CNN: {e}")

    bilstm_model_path = os.path.join(model_dir, "bilstm", "best_model", "model.pt")
    if os.path.exists(bilstm_model_path):
        try:
            from bilstm_classifier import BiLSTMAttention
            vocab = torch.load(os.path.join(model_dir, "bilstm", "best_model", "vocab.pt"))
            bilstm_model = BiLSTMAttention(vocab_size=len(vocab), num_classes=5)
            bilstm_model.load_state_dict(torch.load(bilstm_model_path, map_location=device))
            bilstm_model.to(device)
            bilstm_model.eval()
            models["bilstm"] = bilstm_model
            tokenizers["bilstm"] = vocab
            print(f"  Loaded BiLSTM")
        except Exception as e:
            print(f"  Warning: Could not load BiLSTM: {e}")

    return models, tokenizers


# ── Inference ────────────────────────────────────────────────────────────────

def predict_single_tweet(text, models, tokenizers, device="cpu"):
    """
    Run inference on a single tweet through all loaded models.

    Returns:
        model_probs: dict of model_key -> probability array of shape (num_classes,)
    """
    model_probs = {}

    for key, model in models.items():
        if key in TRANSFORMER_KEYS:
            tokenizer = tokenizers[key]
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True,
                padding="max_length", max_length=128,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            model_probs[key] = probs

        elif key in ("cnn", "bilstm"):
            vocab = tokenizers[key]
            encoded = vocab.encode(text)
            tensor = torch.tensor([encoded], dtype=torch.long).to(device)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            model_probs[key] = probs

    return model_probs


def classify_tweet(
    text,
    models,
    tokenizers,
    meta_learner=None,
    scaler=None,
    class_names=None,
    device="cpu",
):
    """
    Classify a single tweet using the dynamic ensemble.

    Returns:
        result dict with predictions, probabilities, style, and per-model details
    """
    from model_characterisation import classify_tweet_style

    # Get tweet style
    style, style_scores = classify_tweet_style(text)

    # Get per-model predictions
    model_probs = predict_single_tweet(text, models, tokenizers, device)

    if not model_probs:
        return {"error": "No models loaded"}

    # If meta-learner is available, use dynamic ensemble
    if meta_learner is not None and scaler is not None:
        from dynamic_ensemble import build_meta_features
        from model_characterisation import STYLE_CATEGORIES

        # Build probs list in correct order
        ordered_keys = [k for k in ALL_MODEL_KEYS if k in model_probs]
        probs_list = [model_probs[k].reshape(1, -1) for k in ordered_keys]

        meta_features, _ = build_meta_features(
            probs_list, [text], style_labels=[style]
        )
        meta_features_scaled = scaler.transform(meta_features)
        ensemble_probs = meta_learner.predict_proba(meta_features_scaled)[0]
        ensemble_pred = int(np.argmax(ensemble_probs))
    else:
        # Simple average ensemble
        all_probs = np.stack(list(model_probs.values()))
        ensemble_probs = all_probs.mean(axis=0)
        ensemble_pred = int(np.argmax(ensemble_probs))

    # Build result
    pred_class = class_names[ensemble_pred] if class_names else str(ensemble_pred)

    result = {
        "predicted_class": pred_class,
        "confidence": float(ensemble_probs[ensemble_pred]),
        "tweet_style": style,
        "style_scores": style_scores,
        "ensemble_probabilities": {
            class_names[i] if class_names else str(i): float(p)
            for i, p in enumerate(ensemble_probs)
        },
        "per_model": {
            DISPLAY_NAMES.get(k, k): {
                "predicted_class": class_names[int(np.argmax(p))] if class_names else str(np.argmax(p)),
                "confidence": float(np.max(p)),
            }
            for k, p in model_probs.items()
        },
    }

    return result


# ── Gradio Dashboard ─────────────────────────────────────────────────────────

def create_dashboard(
    model_dir,
    label_mapping_path=None,
    ensemble_dir=None,
    device=None,
):
    """
    Create a Gradio-based crisis dashboard.

    Args:
        model_dir: directory containing all model outputs
        label_mapping_path: path to label_mapping.json
        ensemble_dir: directory containing meta_learner.pkl and scaler.pkl
        device: 'cuda' or 'cpu'

    Returns:
        Gradio Blocks app
    """
    import gradio as gr
    import joblib

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models
    print("Loading models...")
    models, tokenizers = load_all_models(model_dir, device)
    print(f"  Loaded {len(models)} models: {list(models.keys())}")

    # Load label mapping
    class_names = None
    if label_mapping_path and os.path.exists(label_mapping_path):
        with open(label_mapping_path) as f:
            mapping = json.load(f)
        id2label = {int(k): v for k, v in mapping["id2label"].items()}
        class_names = [id2label[i] for i in range(len(id2label))]

    # Load meta-learner if available
    meta_learner = None
    scaler = None
    if ensemble_dir:
        ml_path = os.path.join(ensemble_dir, "meta_learner.pkl")
        sc_path = os.path.join(ensemble_dir, "scaler.pkl")
        if os.path.exists(ml_path) and os.path.exists(sc_path):
            meta_learner = joblib.load(ml_path)
            scaler = joblib.load(sc_path)
            print("  Loaded meta-learner and scaler")

    def predict_fn(tweet_text):
        if not tweet_text.strip():
            return "Please enter a tweet.", "", ""

        result = classify_tweet(
            tweet_text, models, tokenizers,
            meta_learner, scaler, class_names, device,
        )

        if "error" in result:
            return result["error"], "", ""

        # Format main prediction
        main_output = (
            f"**Predicted Category:** {result['predicted_class']}\n\n"
            f"**Confidence:** {result['confidence']:.4f}\n\n"
            f"**Tweet Style:** {result['tweet_style']}"
        )

        # Format ensemble probabilities
        prob_output = "\n".join([
            f"- {cls}: {prob:.4f}"
            for cls, prob in sorted(
                result['ensemble_probabilities'].items(),
                key=lambda x: -x[1]
            )
        ])

        # Format per-model results
        model_output = "\n".join([
            f"- **{name}:** {info['predicted_class']} ({info['confidence']:.4f})"
            for name, info in result['per_model'].items()
        ])

        return main_output, prob_output, model_output

    # Build Gradio interface
    with gr.Blocks(
        title="Crisis Tweet Classification Dashboard",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown("# 🚨 Crisis Tweet Classification Dashboard")
        gr.Markdown(
            "Classify disaster tweets using an 8-model dynamic ensemble "
            "(6 transformers + CNN + BiLSTM) with tweet style analysis."
        )

        with gr.Row():
            tweet_input = gr.Textbox(
                label="Enter Tweet",
                placeholder="Type or paste a disaster-related tweet here...",
                lines=3,
            )

        classify_btn = gr.Button("🔍 Classify Tweet", variant="primary")

        with gr.Row():
            with gr.Column():
                main_output = gr.Markdown(label="Classification Result")
            with gr.Column():
                prob_output = gr.Markdown(label="Class Probabilities")
            with gr.Column():
                model_output = gr.Markdown(label="Per-Model Predictions")

        classify_btn.click(
            fn=predict_fn,
            inputs=[tweet_input],
            outputs=[main_output, prob_output, model_output],
        )

        # Example tweets
        gr.Examples(
            examples=[
                ["HELP! Building collapsed, people trapped inside! Send rescue NOW!"],
                ["The Red Cross has set up 3 relief camps in the affected district."],
                ["I can see flooding from my window, water is rising fast near the bridge."],
                ["Hurricane Maria has been downgraded to Category 2 as it moves inland."],
                ["Just watched a great movie tonight, loved the special effects!"],
            ],
            inputs=[tweet_input],
        )

    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--label_mapping", default=None)
    parser.add_argument("--ensemble_dir", default=None)
    args = parser.parse_args()

    app = create_dashboard(
        model_dir=args.model_dir,
        label_mapping_path=args.label_mapping,
        ensemble_dir=args.ensemble_dir,
    )
    app.launch(share=True)
