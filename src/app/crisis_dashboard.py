"""
Crisis Dashboard — Standalone Gradio application.

Provides the inference pipeline and a polished Gradio interface for the
disaster tweet classification system using all 8 models.

Usage (from Kaggle/Colab):
    from crisis_dashboard import create_dashboard
    app = create_dashboard(model_dir="/path/to/models")
    app.launch()
"""

import os
import json
import traceback
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ── Constants ────────────────────────────────────────────────────────────────

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

# Operational guidance per predicted class
CLASS_GUIDANCE = {
    "infrastructure_and_utility_damage": "🏗️ Dispatch structural assessment teams. Check road, bridge, and utility status.",
    "injured_or_dead_people": "🚑 Deploy medical response teams immediately. Activate casualty management protocol.",
    "not_humanitarian": "ℹ️ No humanitarian action required. Monitor for updates.",
    "other_relevant_information": "📋 Log for situational awareness. Share with coordination teams.",
    "rescue_volunteering_or_donation_effort": "🤝 Coordinate with volunteer groups. Activate donation logistics pipeline.",
}


# ── Model Loading ────────────────────────────────────────────────────────────

def load_transformer_model(model_dir, model_key, device="cpu"):
    """Load a trained transformer model and tokenizer from local checkpoint."""
    # Try with seed_42 subfolder first, then without
    path_with_seed = os.path.join(model_dir, model_key, "seed_42", "best_model")
    path_without_seed = os.path.join(model_dir, model_key, "best_model")

    if os.path.exists(path_with_seed):
        model_path = path_with_seed
    elif os.path.exists(path_without_seed):
        model_path = path_without_seed
    else:
        print(f"  ⚠ {model_key}: no checkpoint at {path_with_seed} or {path_without_seed}")
        return None, None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
        print(f"  ✅ {DISPLAY_NAMES[model_key]} loaded from {model_path}")
        return model, tokenizer
    except Exception as e:
        print(f"  ❌ {model_key} failed to load: {e}")
        return None, None


def load_all_models(model_dir, device="cpu"):
    """Load all available models (transformers + CNN + BiLSTM)."""
    models = {}
    tokenizers = {}

    for key in TRANSFORMER_KEYS:
        model, tokenizer = load_transformer_model(model_dir, key, device)
        if model is not None:
            models[key] = model
            tokenizers[key] = tokenizer

    # CNN
    cnn_model_path = os.path.join(model_dir, "cnn", "best_model", "model.pt")
    if os.path.exists(cnn_model_path):
        try:
            from cnn_classifier import TextCNN
            vocab = torch.load(os.path.join(model_dir, "cnn", "best_model", "vocab.pt"),
                               map_location=device)
            cnn_model = TextCNN(vocab_size=len(vocab), num_classes=5)
            cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device))
            cnn_model.to(device).eval()
            models["cnn"] = cnn_model
            tokenizers["cnn"] = vocab
            print(f"  ✅ CNN loaded")
        except Exception as e:
            print(f"  ❌ CNN failed: {e}")

    # BiLSTM
    bilstm_model_path = os.path.join(model_dir, "bilstm", "best_model", "model.pt")
    if os.path.exists(bilstm_model_path):
        try:
            from bilstm_classifier import BiLSTMAttention
            vocab = torch.load(os.path.join(model_dir, "bilstm", "best_model", "vocab.pt"),
                               map_location=device)
            bilstm_model = BiLSTMAttention(vocab_size=len(vocab), num_classes=5)
            bilstm_model.load_state_dict(torch.load(bilstm_model_path, map_location=device))
            bilstm_model.to(device).eval()
            models["bilstm"] = bilstm_model
            tokenizers["bilstm"] = vocab
            print(f"  ✅ BiLSTM loaded")
        except Exception as e:
            print(f"  ❌ BiLSTM failed: {e}")

    return models, tokenizers


# ── Inference ────────────────────────────────────────────────────────────────

def predict_single_tweet(text, models, tokenizers, device="cpu"):
    """Run inference on a single tweet through all loaded models."""
    model_probs = {}

    for key, model in models.items():
        try:
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
        except Exception as e:
            print(f"  ⚠ Inference failed for {key}: {e}")

    return model_probs


def classify_tweet(
    text, models, tokenizers,
    meta_learner=None, scaler=None,
    class_names=None, device="cpu",
):
    """
    Classify a single tweet using the dynamic ensemble.

    Returns a result dict or a dict with 'error' key on failure.
    """
    try:
        from model_characterisation import classify_tweet_style

        style, style_scores = classify_tweet_style(text)
        model_probs = predict_single_tweet(text, models, tokenizers, device)

        if not model_probs:
            return {"error": "No models produced predictions. Check model loading logs."}

        if meta_learner is not None and scaler is not None:
            from dynamic_ensemble import build_meta_features

            ordered_keys = [k for k in ALL_MODEL_KEYS if k in model_probs]
            probs_list = [model_probs[k].reshape(1, -1) for k in ordered_keys]

            meta_features, _ = build_meta_features(
                probs_list, [text], style_labels=[style]
            )
            meta_features_scaled = scaler.transform(meta_features)
            ensemble_probs = meta_learner.predict_proba(meta_features_scaled)[0]
            ensemble_pred = int(np.argmax(ensemble_probs))
        else:
            all_probs = np.stack(list(model_probs.values()))
            ensemble_probs = all_probs.mean(axis=0)
            ensemble_pred = int(np.argmax(ensemble_probs))

        pred_class = class_names[ensemble_pred] if class_names else str(ensemble_pred)

        return {
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
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Classification error:\n{tb}")
        return {"error": f"Classification failed: {str(e)}"}


# ── HTML Helpers ─────────────────────────────────────────────────────────────

def _confidence_color(conf):
    if conf >= 0.8:
        return "#22c55e"  # green
    elif conf >= 0.6:
        return "#f59e0b"  # orange
    return "#ef4444"      # red


def _build_prediction_card(result, class_names):
    """Build an HTML card showing the main prediction."""
    pred = result["predicted_class"]
    conf = result["confidence"]
    style = result["tweet_style"]
    guidance = CLASS_GUIDANCE.get(pred, "")
    color = _confidence_color(conf)

    return f"""
    <div style="background:#fff;border-radius:12px;padding:24px;box-shadow:0 2px 8px rgba(0,0,0,0.1);">
      <div style="font-size:13px;color:#6b7280;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">
        Predicted Category
      </div>
      <div style="font-size:20px;font-weight:700;color:#1a1a2e;margin-bottom:12px;">
        {pred.replace('_', ' ').title()}
      </div>
      <div style="display:inline-block;background:{color};color:#fff;padding:4px 14px;
                  border-radius:20px;font-size:14px;font-weight:600;margin-bottom:16px;">
        Confidence: {conf:.1%}
      </div>
      <div style="margin-top:12px;font-size:14px;color:#374151;">
        <strong>Tweet Style:</strong> {style}
      </div>
      <div style="margin-top:16px;padding:12px;background:#f0fdf4;border-left:4px solid #22c55e;
                  border-radius:0 8px 8px 0;font-size:13px;color:#166534;">
        <strong>Action:</strong> {guidance}
      </div>
    </div>
    """


def _build_probability_bars(result, class_names):
    """Build styled HTML probability bars for each class."""
    probs = result["ensemble_probabilities"]
    sorted_probs = sorted(probs.items(), key=lambda x: -x[1])

    bars_html = ""
    for cls, prob in sorted_probs:
        pct = prob * 100
        label = cls.replace("_", " ").title()
        color = _confidence_color(prob) if prob > 0.3 else "#94a3b8"
        bars_html += f"""
        <div style="margin-bottom:10px;">
          <div style="display:flex;justify-content:space-between;font-size:12px;color:#374151;margin-bottom:3px;">
            <span>{label}</span>
            <span style="font-weight:600;">{prob:.4f}</span>
          </div>
          <div style="background:#e5e7eb;border-radius:6px;height:10px;overflow:hidden;">
            <div style="width:{pct}%;background:{color};height:100%;border-radius:6px;
                        transition:width 0.5s ease;"></div>
          </div>
        </div>
        """

    return f"""
    <div style="background:#fff;border-radius:12px;padding:24px;box-shadow:0 2px 8px rgba(0,0,0,0.1);">
      <div style="font-size:13px;color:#6b7280;text-transform:uppercase;letter-spacing:1px;margin-bottom:16px;">
        Class Probabilities
      </div>
      {bars_html}
    </div>
    """


def _build_model_table(result):
    """Build an HTML table showing per-model predictions."""
    rows_html = ""
    for name, info in result["per_model"].items():
        pred = info["predicted_class"].replace("_", " ").title()
        conf = info["confidence"]
        color = _confidence_color(conf)
        rows_html += f"""
        <tr>
          <td style="padding:8px 12px;font-weight:500;border-bottom:1px solid #f3f4f6;">{name}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #f3f4f6;font-size:13px;">{pred}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #f3f4f6;text-align:right;">
            <span style="color:{color};font-weight:600;">{conf:.4f}</span>
          </td>
        </tr>
        """

    return f"""
    <div style="background:#fff;border-radius:12px;padding:24px;box-shadow:0 2px 8px rgba(0,0,0,0.1);">
      <div style="font-size:13px;color:#6b7280;text-transform:uppercase;letter-spacing:1px;margin-bottom:16px;">
        Per-Model Predictions
      </div>
      <table style="width:100%;border-collapse:collapse;">
        <thead>
          <tr style="border-bottom:2px solid #e5e7eb;">
            <th style="padding:8px 12px;text-align:left;font-size:12px;color:#6b7280;">Model</th>
            <th style="padding:8px 12px;text-align:left;font-size:12px;color:#6b7280;">Prediction</th>
            <th style="padding:8px 12px;text-align:right;font-size:12px;color:#6b7280;">Confidence</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>
    """


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

    # ── Load models ──
    print("=" * 60)
    print("CRISIS DASHBOARD — Loading Models")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Model directory: {model_dir}")

    models, tokenizers = load_all_models(model_dir, device)
    print(f"\n  Total loaded: {len(models)} models — {list(models.keys())}")

    if not models:
        print("  ❌ FATAL: No models loaded. Check model_dir path.")

    # ── Load label mapping ──
    class_names = None
    if label_mapping_path and os.path.exists(label_mapping_path):
        with open(label_mapping_path) as f:
            mapping = json.load(f)
        id2label = {int(k): v for k, v in mapping["id2label"].items()}
        class_names = [id2label[i] for i in range(len(id2label))]
        print(f"  Label mapping loaded: {class_names}")
    else:
        print(f"  ⚠ label_mapping_path not found: {label_mapping_path}")

    # ── Load meta-learner ──
    meta_learner = None
    scaler = None
    if ensemble_dir:
        ml_path = os.path.join(ensemble_dir, "meta_learner.pkl")
        sc_path = os.path.join(ensemble_dir, "scaler.pkl")
        if os.path.exists(ml_path) and os.path.exists(sc_path):
            meta_learner = joblib.load(ml_path)
            scaler = joblib.load(sc_path)
            print(f"  ✅ Meta-learner loaded from {ml_path}")
            print(f"  ✅ Scaler loaded from {sc_path}")
        else:
            print(f"  ⚠ Ensemble artifacts not found at {ensemble_dir}")
    print("=" * 60)

    # ── Prediction function ──
    def predict_fn(tweet_text):
        empty = ('<div style="padding:20px;color:#9ca3af;text-align:center;">'
                 'Enter a tweet and click Classify.</div>')
        if not tweet_text or not tweet_text.strip():
            return empty, empty, empty

        result = classify_tweet(
            tweet_text, models, tokenizers,
            meta_learner, scaler, class_names, device,
        )

        if "error" in result:
            err_html = (f'<div style="padding:20px;color:#ef4444;font-weight:600;">'
                        f'Error: {result["error"]}</div>')
            return err_html, "", ""

        pred_card = _build_prediction_card(result, class_names)
        prob_bars = _build_probability_bars(result, class_names)
        model_table = _build_model_table(result)
        return pred_card, prob_bars, model_table

    # ── Custom CSS ──
    custom_css = """
    .gradio-container { max-width: 1100px !important; margin: auto; }
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white; padding: 32px 28px; border-radius: 16px;
        margin-bottom: 24px; text-align: center;
    }
    .main-header h1 { font-size: 28px; margin: 0 0 6px 0; font-weight: 700; }
    .main-header p { font-size: 14px; color: #94a3b8; margin: 0; }
    """

    # ── Build interface ──
    with gr.Blocks(
        title="Crisis Tweet Classification",
        theme=gr.themes.Soft(),
        css=custom_css,
    ) as app:

        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🚨 Crisis Tweet Classification Dashboard</h1>
            <p>8-model dynamic ensemble · tweet style analysis · attribution-based reliability</p>
        </div>
        """)

        # Input area
        with gr.Row():
            with gr.Column(scale=3):
                tweet_input = gr.Textbox(
                    label="Tweet Text",
                    placeholder="Enter a disaster-related tweet here, e.g. Roads flooded in downtown area, rescue teams needed immediately",
                    lines=4,
                    max_lines=6,
                )
            with gr.Column(scale=1, min_width=160):
                gr.HTML("<div style='height:8px'></div>")
                classify_btn = gr.Button("🔍 Classify", variant="primary", size="lg")
                clear_btn = gr.ClearButton(components=[tweet_input], value="🗑 Clear", size="lg")

        gr.HTML("<div style='margin:16px 0'></div>")

        # Results — two-column layout
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                pred_output = gr.HTML(
                    value='<div style="padding:20px;color:#9ca3af;text-align:center;">Results will appear here.</div>'
                )
            with gr.Column(scale=1):
                prob_output = gr.HTML(value="")

        gr.HTML("<div style='margin:12px 0'></div>")

        with gr.Row():
            model_output = gr.HTML(value="")

        classify_btn.click(
            fn=predict_fn,
            inputs=[tweet_input],
            outputs=[pred_output, prob_output, model_output],
        )

        gr.HTML("<div style='margin:20px 0'></div>")

        # Example tweets — one per class
        gr.Examples(
            examples=[
                ["HELP! Building collapsed, people trapped inside! Send rescue NOW!"],
                ["3 confirmed dead and dozens injured after the earthquake hit the coastal town this morning"],
                ["The Red Cross has set up 3 relief camps in the affected district, donations needed urgently"],
                ["Hurricane has been downgraded to Category 2 as it moves inland, power outages reported"],
                ["Just watched a great movie tonight, loved the special effects!"],
            ],
            inputs=[tweet_input],
            label="Example Tweets (one per class)",
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
