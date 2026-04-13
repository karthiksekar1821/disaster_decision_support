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
    "infrastructure_and_utility_damage": "🏗️ Deploy engineering and utility restoration teams",
    "injured_or_dead_people": "🚑 Deploy medical response teams immediately",
    "not_humanitarian": "ℹ️ No immediate action required",
    "other_relevant_information": "📋 Monitor and log for situational awareness",
    "rescue_volunteering_or_donation_effort": "🤝 Coordinate with volunteer and donation networks",
}

# Color per class for prediction badges
CLASS_COLORS = {
    "infrastructure_and_utility_damage": "#e07c3a",
    "injured_or_dead_people": "#e05555",
    "not_humanitarian": "#6b7a8a",
    "other_relevant_information": "#5b8dd9",
    "rescue_volunteering_or_donation_effort": "#00c47c",
}

DEFAULT_CLASS_NAMES = [
    "infrastructure_and_utility_damage",
    "injured_or_dead_people",
    "not_humanitarian",
    "other_relevant_information",
    "rescue_volunteering_or_donation_effort",
]


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
        print(f"  ✅ Loaded {model_key} from {model_path}")
        return model, tokenizer
    except Exception as e:
        print(f"  ❌ Failed to load {model_key}: {e}")
        raise


def load_all_models(model_dir, device="cpu"):
    """Load all available models (transformers + CNN + BiLSTM)."""
    models = {}
    tokenizers = {}

    for key in TRANSFORMER_KEYS:
        try:
            model, tokenizer = load_transformer_model(model_dir, key, device)
            if model is not None:
                models[key] = model
                tokenizers[key] = tokenizer
        except Exception as e:
            print(f"  ❌ Skipping {key} due to load error: {e}")

    # CNN
    cnn_model_path = os.path.join(model_dir, "cnn", "best_model", "model.pt")
    print(f"  Looking for CNN at: {cnn_model_path}")
    if os.path.exists(cnn_model_path):
        try:
            from cnn_classifier import TextCNN
            vocab = torch.load(os.path.join(model_dir, "cnn", "best_model", "vocab.pt"),
                               map_location=device, weights_only=False)
            cnn_model = TextCNN(vocab_size=len(vocab), num_classes=5)
            cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device, weights_only=False))
            cnn_model.to(device).eval()
            models["cnn"] = cnn_model
            tokenizers["cnn"] = vocab
            print(f"  ✅ Loaded CNN from {cnn_model_path}")
        except Exception as e:
            print(f"  ❌ Failed to load CNN: {e}")
            raise
    else:
        print(f"  ⚠ CNN model.pt not found at {cnn_model_path}")

    # BiLSTM
    bilstm_model_path = os.path.join(model_dir, "bilstm", "best_model", "model.pt")
    print(f"  Looking for BiLSTM at: {bilstm_model_path}")
    if os.path.exists(bilstm_model_path):
        try:
            from bilstm_classifier import BiLSTMAttention
            vocab = torch.load(os.path.join(model_dir, "bilstm", "best_model", "vocab.pt"),
                               map_location=device, weights_only=False)
            bilstm_model = BiLSTMAttention(vocab_size=len(vocab), num_classes=5)
            bilstm_model.load_state_dict(torch.load(bilstm_model_path, map_location=device, weights_only=False))
            bilstm_model.to(device).eval()
            models["bilstm"] = bilstm_model
            tokenizers["bilstm"] = vocab
            print(f"  ✅ Loaded BiLSTM from {bilstm_model_path}")
        except Exception as e:
            print(f"  ❌ Failed to load BiLSTM: {e}")
            raise
    else:
        print(f"  ⚠ BiLSTM model.pt not found at {bilstm_model_path}")

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


def _confidence_label(conf):
    if conf >= 0.8:
        return "High Confidence"
    elif conf >= 0.6:
        return "Medium Confidence"
    return "Low Confidence"


def _build_prediction_card(result, class_names):
    """Build an HTML card showing the main prediction with colored badge."""
    pred = result["predicted_class"]
    conf = result["confidence"]
    style = result["tweet_style"]
    guidance = CLASS_GUIDANCE.get(pred, "")
    conf_color = _confidence_color(conf)
    conf_label = _confidence_label(conf)
    class_color = CLASS_COLORS.get(pred, "#5b8dd9")

    return f"""
    <div style="background:#fff;border-radius:16px;padding:28px;
                box-shadow:0 2px 12px rgba(0,0,0,0.08);margin-bottom:12px;">
      <div style="font-size:12px;color:#333333;text-transform:uppercase;
                  letter-spacing:1.5px;margin-bottom:10px;font-weight:500;">
        Predicted Category
      </div>
      <div style="display:inline-block;background:{class_color};color:#fff;
                  padding:8px 20px;border-radius:24px;font-size:16px;
                  font-weight:700;margin-bottom:16px;">
        {pred.replace('_', ' ').title()}
      </div>
      <div style="margin-top:12px;">
        <span style="display:inline-block;background:{conf_color};color:#fff;
                     padding:5px 16px;border-radius:20px;font-size:13px;
                     font-weight:600;">
          {conf_label}: {conf:.1%}
        </span>
      </div>
      <div style="margin-top:14px;font-size:14px;color:#333333;">
        <strong>Tweet Style:</strong> {style}
      </div>
      <div style="margin-top:16px;padding:14px;background:#f0fdf4;
                  border-left:4px solid #22c55e;border-radius:0 10px 10px 0;
                  font-size:13px;color:#1a1a1a;">
        <strong>Operational Action:</strong> {guidance}
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
        bar_color = CLASS_COLORS.get(cls, "#94a3b8")
        bars_html += f"""
        <div style="margin-bottom:12px;">
          <div style="display:flex;justify-content:space-between;font-size:12px;
                      color:#1a1a1a;margin-bottom:4px;">
            <span>{label}</span>
            <span style="font-weight:600;color:#1a1a1a;">{prob:.4f}</span>
          </div>
          <div style="background:#e5e7eb;border-radius:8px;height:12px;overflow:hidden;">
            <div style="width:{pct}%;background:{bar_color};height:100%;
                        border-radius:8px;transition:width 0.5s ease;"></div>
          </div>
        </div>
        """

    return f"""
    <div style="background:#fff;border-radius:16px;padding:28px;
                box-shadow:0 2px 12px rgba(0,0,0,0.08);margin-bottom:12px;">
      <div style="font-size:12px;color:#333333;text-transform:uppercase;
                  letter-spacing:1.5px;margin-bottom:18px;font-weight:500;">
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
          <td style="padding:10px 14px;font-weight:500;color:#1a1a1a;
                     border-bottom:1px solid #f3f4f6;">{name}</td>
          <td style="padding:10px 14px;border-bottom:1px solid #f3f4f6;
                     font-size:13px;color:#1a1a1a;">{pred}</td>
          <td style="padding:10px 14px;border-bottom:1px solid #f3f4f6;
                     text-align:right;">
            <span style="color:{color};font-weight:600;">{conf:.4f}</span>
          </td>
        </tr>
        """

    return f"""
    <div style="background:#fff;border-radius:16px;padding:28px;
                box-shadow:0 2px 12px rgba(0,0,0,0.08);">
      <div style="font-size:12px;color:#333333;text-transform:uppercase;
                  letter-spacing:1.5px;margin-bottom:18px;font-weight:500;">
        Per-Model Predictions
      </div>
      <table style="width:100%;border-collapse:collapse;">
        <thead>
          <tr style="border-bottom:2px solid #e5e7eb;">
            <th style="padding:10px 14px;text-align:left;font-size:12px;
                       color:#333333;font-weight:700;">Model</th>
            <th style="padding:10px 14px;text-align:left;font-size:12px;
                       color:#333333;font-weight:700;">Prediction</th>
            <th style="padding:10px 14px;text-align:right;font-size:12px;
                       color:#333333;font-weight:700;">Confidence</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>
    """


def _build_attribution_html(text, model, tokenizer, pred_class, device="cpu"):
    """Build token attribution HTML with green/red highlighting."""
    try:
        from captum.attr import LayerIntegratedGradients

        # Detect embedding layer
        if hasattr(model, "roberta"):
            embed_layer = model.roberta.embeddings.word_embeddings
        elif hasattr(model, "deberta"):
            embed_layer = model.deberta.embeddings.word_embeddings
        elif hasattr(model, "electra"):
            embed_layer = model.electra.embeddings.word_embeddings
        elif hasattr(model, "bert"):
            embed_layer = model.bert.embeddings.word_embeddings
        elif hasattr(model, "distilbert"):
            embed_layer = model.distilbert.embeddings.word_embeddings
        else:
            return "<div style='color:#6b7280;font-size:13px;padding:12px;'>Attribution not available for this model architecture.</div>"

        def forward_fn(input_ids, attention_mask):
            return model(input_ids=input_ids, attention_mask=attention_mask).logits

        lig = LayerIntegratedGradients(forward_fn, embed_layer)

        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                          padding="max_length", max_length=128)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        baseline = torch.zeros_like(input_ids)

        attrs = lig.attribute(
            inputs=input_ids, baselines=baseline,
            additional_forward_args=(attention_mask,),
            target=int(pred_class), n_steps=30,
        )
        scores = attrs.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        seq_len = attention_mask.squeeze(0).sum().item()
        tokens = tokenizer.convert_ids_to_tokens(
            input_ids.squeeze(0).cpu().tolist()
        )[:seq_len]
        scores = scores[:seq_len]

        mx = np.abs(scores).max()
        if mx > 0:
            scores = scores / mx

        skip = {tokenizer.cls_token, tokenizer.sep_token,
                tokenizer.pad_token, "<s>", "</s>", "<pad>"}
        html = ""
        for tok, score in zip(tokens, scores):
            if tok in skip:
                continue
            display_tok = tok.replace("Ġ", " ").replace("Ċ", " ").replace("##", "")
            alpha = min(0.15 + abs(score) * 0.7, 0.9)
            if score > 0:
                bg = f"rgba(34,197,94,{alpha:.2f})"
            else:
                bg = f"rgba(239,68,68,{alpha:.2f})"
            html += (f'<span style="background:{bg};padding:3px 2px;margin:1px;'
                     f'border-radius:3px;font-size:13px;line-height:2.2;'
                     f'color:#1a1a1a;cursor:default;" title="score:{score:.3f}">'
                     f'{display_tok}</span>')

        legend = """
        <div style="display:flex;gap:16px;margin-top:10px;padding-top:8px;
                    border-top:1px solid #e5e7eb;">
          <div style="display:flex;align-items:center;gap:5px;font-size:11px;color:#333333;">
            <div style="width:12px;height:12px;border-radius:3px;
                        background:rgba(34,197,94,0.7);"></div>Supports prediction
          </div>
          <div style="display:flex;align-items:center;gap:5px;font-size:11px;color:#333333;">
            <div style="width:12px;height:12px;border-radius:3px;
                        background:rgba(239,68,68,0.7);"></div>Opposes prediction
          </div>
        </div>
        """

        return f"""
        <div style="background:#fff;border-radius:16px;padding:28px;
                    box-shadow:0 2px 12px rgba(0,0,0,0.08);margin-top:12px;">
          <div style="font-size:12px;color:#333333;text-transform:uppercase;
                      letter-spacing:1.5px;margin-bottom:14px;font-weight:500;">
            Token Attribution (Integrated Gradients)
          </div>
          <div style="line-height:2.2;word-wrap:break-word;">
            {html}
          </div>
          {legend}
        </div>
        """

    except ImportError:
        return "<div style='color:#6b7280;font-size:13px;padding:12px;'>Install captum for token attributions: pip install captum</div>"
    except Exception as e:
        return f"<div style='color:#ef4444;font-size:13px;padding:12px;'>Attribution error: {str(e)}</div>"


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

    # ── Print all paths being used ──
    print("=" * 60)
    print("CRISIS DASHBOARD — Startup Configuration")
    print("=" * 60)
    print(f"  Device:             {device}")
    print(f"  Model directory:    {model_dir}")
    print(f"  Label mapping:      {label_mapping_path}")
    print(f"  Ensemble directory: {ensemble_dir}")
    print("=" * 60)

    # ── Load models ──
    print("\n--- Loading Models ---")
    models, tokenizers = load_all_models(model_dir, device)
    print(f"\n  Total loaded: {len(models)} models — {list(models.keys())}")

    if not models:
        print("  ❌ FATAL: No models loaded. Check model_dir path.")

    # ── Load label mapping ──
    class_names = None
    if label_mapping_path and os.path.exists(label_mapping_path):
        try:
            with open(label_mapping_path) as f:
                mapping = json.load(f)
            id2label = {int(k): v for k, v in mapping["id2label"].items()}
            class_names = [id2label[i] for i in range(len(id2label))]
            print(f"  ✅ Label mapping loaded: {class_names}")
        except Exception as e:
            print(f"  ❌ Failed to load label mapping: {e}")
            raise
    else:
        print(f"  ⚠ label_mapping_path not found: {label_mapping_path}")
        class_names = DEFAULT_CLASS_NAMES
        print(f"  Using default class names: {class_names}")

    # ── Load meta-learner ──
    meta_learner = None
    scaler = None
    if ensemble_dir:
        ml_path = os.path.join(ensemble_dir, "meta_learner.pkl")
        sc_path = os.path.join(ensemble_dir, "scaler.pkl")
        print(f"  Looking for meta_learner at: {ml_path}")
        print(f"  Looking for scaler at: {sc_path}")
        if os.path.exists(ml_path) and os.path.exists(sc_path):
            try:
                meta_learner = joblib.load(ml_path)
                scaler = joblib.load(sc_path)
                print(f"  ✅ Meta-learner loaded from {ml_path}")
                print(f"  ✅ Scaler loaded from {sc_path}")
            except Exception as e:
                print(f"  ❌ Failed to load ensemble artifacts: {e}")
                raise
        else:
            print(f"  ⚠ Ensemble artifacts not found at {ensemble_dir}")

    # ── Test inference ──
    print("\n--- Test Inference ---")
    test_tweet = "People are trapped under rubble after the earthquake, urgent rescue needed immediately"
    test_result = classify_tweet(test_tweet, models, tokenizers,
                                 meta_learner, scaler, class_names, device)
    if "error" in test_result:
        print(f"  ❌ Test inference FAILED: {test_result['error']}")
    else:
        print(f"  ✅ Test tweet: '{test_tweet[:60]}...'")
        print(f"     Predicted: {test_result['predicted_class']}")
        print(f"     Confidence: {test_result['confidence']:.4f}")
    print("=" * 60)

    # Store first loaded transformer for attributions
    first_transformer_key = None
    first_transformer_model = None
    first_transformer_tokenizer = None
    for k in TRANSFORMER_KEYS:
        if k in models:
            first_transformer_key = k
            first_transformer_model = models[k]
            first_transformer_tokenizer = tokenizers[k]
            break

    # ── Prediction function ──
    def predict_fn(tweet_text):
        empty = ('<div style="padding:24px;color:#9ca3af;text-align:center;'
                 'font-size:15px;">Enter a tweet and click Classify.</div>')
        if not tweet_text or not tweet_text.strip():
            return empty, empty, empty, ""

        result = classify_tweet(
            tweet_text, models, tokenizers,
            meta_learner, scaler, class_names, device,
        )

        if "error" in result:
            err_html = (f'<div style="padding:24px;color:#ef4444;font-weight:600;'
                        f'font-size:15px;">Error: {result["error"]}</div>')
            return err_html, "", "", ""

        pred_card = _build_prediction_card(result, class_names)
        prob_bars = _build_probability_bars(result, class_names)
        model_table = _build_model_table(result)

        # Token attributions
        attr_html = ""
        if first_transformer_model is not None:
            pred_idx = list(result["ensemble_probabilities"].values())
            pred_class_idx = int(np.argmax(pred_idx))
            attr_html = _build_attribution_html(
                tweet_text, first_transformer_model,
                first_transformer_tokenizer, pred_class_idx, device,
            )

        return pred_card, prob_bars, model_table, attr_html

    # ── Custom CSS ──
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto;
        padding: 20px !important;
    }
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white; padding: 36px 32px; border-radius: 20px;
        margin-bottom: 28px; text-align: center;
    }
    .main-header h1 { font-size: 30px; margin: 0 0 8px 0; font-weight: 700; }
    .main-header p { font-size: 14px; color: #94a3b8; margin: 0; }
    """

    # ── Build interface ──
    with gr.Blocks(
        title="Crisis Tweet Classification Dashboard",
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

        # Input area with padding
        with gr.Row():
            with gr.Column(scale=4):
                tweet_input = gr.Textbox(
                    label="Tweet Text",
                    placeholder="Enter a disaster-related tweet here e.g. Roads flooded in downtown area, rescue teams needed immediately",
                    lines=4,
                    max_lines=8,
                )
            with gr.Column(scale=1, min_width=180):
                gr.HTML("<div style='height:12px'></div>")
                classify_btn = gr.Button(
                    "🔍 Classify", variant="primary", size="lg",
                )
                clear_btn = gr.ClearButton(
                    components=[tweet_input], value="🗑 Clear", size="lg",
                )

        gr.HTML("<div style='margin:20px 0'></div>")

        # Results — two-column layout
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                pred_output = gr.HTML(
                    value='<div style="padding:24px;color:#9ca3af;text-align:center;font-size:15px;">Results will appear here.</div>'
                )
            with gr.Column(scale=1):
                prob_output = gr.HTML(value="")

        gr.HTML("<div style='margin:16px 0'></div>")

        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                model_output = gr.HTML(value="")
            with gr.Column(scale=1):
                attr_output = gr.HTML(value="")

        classify_btn.click(
            fn=predict_fn,
            inputs=[tweet_input],
            outputs=[pred_output, prob_output, model_output, attr_output],
        )

        gr.HTML("<div style='margin:24px 0'></div>")

        # Example tweets — one per class
        gr.Examples(
            examples=[
                ["Massive flooding has destroyed the main bridge connecting the two districts, power lines down everywhere"],
                ["3 confirmed dead and dozens injured after the earthquake hit the coastal town this morning"],
                ["Just watched a great movie tonight, loved the special effects!"],
                ["Hurricane has been downgraded to Category 2 as it moves inland, power outages reported across 5 counties"],
                ["The Red Cross has set up 3 relief camps in the affected district, donations needed urgently for food and medicine"],
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
