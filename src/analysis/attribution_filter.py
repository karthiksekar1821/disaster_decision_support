"""
Decision-Influencing Explainability via Attribution Filtering (Novelty 3).

Instead of just visualizing attributions post-hoc, this module uses
Integrated Gradients attributions to FLAG suspicious predictions:

If a prediction is HIGH CONFIDENCE but the top attributed tokens are
stopwords or irrelevant words (not disaster-related vocabulary), the
prediction is flagged as UNRELIABLE despite the high softmax confidence.

This creates a SECOND abstention signal independent of softmax confidence,
which combines with Novelty 2 (class-adaptive thresholds) to form a
dual-signal selective prediction system.
"""

import numpy as np
import torch
from captum.attr import IntegratedGradients, LayerIntegratedGradients

from disaster_vocab import (
    get_disaster_vocab_for_class,
    is_disaster_relevant,
    is_irrelevant,
    IRRELEVANT_TOKENS,
)


def compute_attributions_for_batch(
    model,
    tokenizer,
    texts,
    predicted_classes,
    device="cuda",
    n_steps=50,
):
    """
    Compute Integrated Gradients attributions for a batch of texts.

    Args:
        model: the transformer model
        tokenizer: the tokenizer
        texts: list of tweet strings
        predicted_classes: array of predicted class indices
        device: 'cuda' or 'cpu'
        n_steps: number of steps for IG approximation

    Returns:
        list of dicts, each containing:
            - tokens: list of tokens
            - attributions: list of attribution scores (per token)
            - predicted_class: int
    """
    model.eval()
    model.to(device)

    # Use LayerIntegratedGradients on the embedding layer
    def forward_func(input_embeds, attention_mask):
        outputs = model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
        )
        return outputs.logits

    # Get the embedding layer
    if hasattr(model, "roberta"):
        embedding_layer = model.roberta.embeddings
    elif hasattr(model, "deberta"):
        embedding_layer = model.deberta.embeddings
    elif hasattr(model, "electra"):
        embedding_layer = model.electra.embeddings
    else:
        # Generic fallback
        embedding_layer = list(model.children())[0].embeddings

    lig = LayerIntegratedGradients(forward_func, embedding_layer)

    results = []

    for idx, (text, pred_class) in enumerate(zip(texts, predicted_classes)):
        # Tokenize
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True,
            padding="max_length", max_length=128,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Compute attributions for the predicted class
        try:
            attrs = lig.attribute(
                inputs=(input_ids,),
                additional_forward_args=(attention_mask,),
                target=int(pred_class),
                n_steps=n_steps,
            )

            # Sum attributions across embedding dimensions
            # attrs shape: (1, seq_len, hidden_dim)
            token_attrs = attrs.sum(dim=-1).squeeze(0)  # (seq_len,)
            token_attrs = token_attrs.detach().cpu().numpy()

        except Exception as e:
            # Fallback: use zero attributions
            token_attrs = np.zeros(input_ids.shape[1])

        # Decode tokens
        tokens = tokenizer.convert_ids_to_tokens(
            input_ids.squeeze(0).cpu().tolist()
        )

        results.append({
            "tokens": tokens,
            "attributions": token_attrs.tolist(),
            "predicted_class": int(pred_class),
        })

    return results


def compute_disaster_relevance_score(
    tokens,
    attributions,
    predicted_class_name,
    top_k=5,
):
    """
    Compute how disaster-relevant the top attributed tokens are.

    Args:
        tokens: list of token strings
        attributions: list of attribution scores
        predicted_class_name: string name of the predicted class
        top_k: number of top-attributed tokens to examine

    Returns:
        relevance_score: float in [0, 1], fraction of top-K tokens
                         that are disaster-relevant
        top_tokens: list of (token, attribution, is_relevant) tuples
        irrelevance_score: float in [0, 1], fraction of top-K tokens
                           that are stopwords/irrelevant
    """
    # Get absolute attribution values
    abs_attrs = np.abs(attributions)

    # Filter out special tokens and padding
    valid_indices = [
        i for i, t in enumerate(tokens)
        if t not in {"[PAD]", "[CLS]", "[SEP]", "[UNK]",
                     "<s>", "</s>", "<pad>", "<unk>"}
        and not t.startswith("▁") or len(t.strip("▁")) > 0
    ]

    if not valid_indices:
        return 0.0, [], 1.0

    # Sort by attribution magnitude
    sorted_indices = sorted(valid_indices, key=lambda i: abs_attrs[i], reverse=True)
    top_indices = sorted_indices[:top_k]

    vocab = get_disaster_vocab_for_class(predicted_class_name)

    top_tokens = []
    relevant_count = 0
    irrelevant_count = 0

    for i in top_indices:
        token = tokens[i]
        # Clean token (remove subword markers)
        clean_token = token.strip("▁").strip("Ġ").strip("##").lower()

        is_rel = clean_token in vocab
        is_irr = is_irrelevant(clean_token)

        if is_rel:
            relevant_count += 1
        if is_irr:
            irrelevant_count += 1

        top_tokens.append({
            "token": token,
            "clean_token": clean_token,
            "attribution": float(attributions[i]),
            "is_disaster_relevant": is_rel,
            "is_irrelevant": is_irr,
        })

    relevance_score = relevant_count / max(len(top_indices), 1)
    irrelevance_score = irrelevant_count / max(len(top_indices), 1)

    return relevance_score, top_tokens, irrelevance_score


def flag_unreliable_predictions(
    attribution_results,
    probs,
    preds,
    class_names,
    per_class_thresholds=None,
    top_k=5,
    irrelevance_threshold=0.6,
):
    """
    Flag predictions that are high-confidence but have irrelevant attributions.

    A prediction is flagged as UNRELIABLE if:
    1. It passes the confidence threshold (high softmax confidence), AND
    2. The fraction of top-K attributed tokens that are stopwords/irrelevant
       exceeds irrelevance_threshold

    Args:
        attribution_results: list of dicts from compute_attributions_for_batch
        probs: (N, C) softmax probabilities
        preds: (N,) predicted class indices
        class_names: list of class name strings
        per_class_thresholds: dict from Novelty 2 (optional)
        top_k: number of top tokens to examine
        irrelevance_threshold: if > this fraction of top-K are irrelevant, flag

    Returns:
        reliability_mask: boolean array, True = reliable, False = unreliable
        reliability_details: list of per-sample details
    """
    N = len(preds)
    reliability_mask = np.ones(N, dtype=bool)  # Default: all reliable
    reliability_details = []

    for i in range(N):
        pred_class = preds[i]
        pred_class_name = class_names[pred_class]
        confidence = float(probs[i, pred_class])

        # Check if this prediction is "confident"
        if per_class_thresholds is not None:
            threshold = per_class_thresholds.get(pred_class, 0.5)
        else:
            threshold = 0.5

        is_confident = confidence >= threshold

        # Compute relevance score
        attr_result = attribution_results[i]
        relevance_score, top_tokens, irrelevance_score = (
            compute_disaster_relevance_score(
                attr_result["tokens"],
                attr_result["attributions"],
                pred_class_name,
                top_k=top_k,
            )
        )

        # Flag as unreliable if confident but attributions are irrelevant
        is_unreliable = (
            is_confident
            and irrelevance_score >= irrelevance_threshold
            and relevance_score == 0.0  # No disaster-relevant tokens at all
        )

        if is_unreliable:
            reliability_mask[i] = False

        detail = {
            "sample_idx": i,
            "predicted_class": pred_class_name,
            "confidence": confidence,
            "is_confident": is_confident,
            "relevance_score": relevance_score,
            "irrelevance_score": irrelevance_score,
            "is_reliable": not is_unreliable,
            "top_tokens": top_tokens,
        }
        reliability_details.append(detail)

    return reliability_mask, reliability_details


def combined_abstention(
    confidence_accepted_mask,
    reliability_mask,
    preds,
    labels,
    class_names,
):
    """
    Combine Novelty 2 (confidence) and Novelty 3 (attribution) abstention.

    A prediction is accepted only if BOTH signals agree:
    1. Confidence threshold says accept (from Novelty 2)
    2. Attribution check says reliable (from Novelty 3)

    Returns:
        combined_mask: boolean array of accepted predictions
        results: dict with combined metrics
    """
    combined_mask = confidence_accepted_mask & reliability_mask

    total = len(preds)

    # Metrics for confidence-only
    conf_count = confidence_accepted_mask.sum()
    if conf_count > 0:
        conf_f1 = float(np.mean(
            preds[confidence_accepted_mask] == labels[confidence_accepted_mask]
        ))
    else:
        conf_f1 = 0.0

    # Metrics for combined
    combined_count = combined_mask.sum()
    combined_coverage = combined_count / total

    if combined_count > 0:
        from sklearn.metrics import f1_score as sklearn_f1
        combined_f1 = sklearn_f1(
            labels[combined_mask], preds[combined_mask], average="macro"
        )
        combined_acc = float(np.mean(
            preds[combined_mask] == labels[combined_mask]
        ))
    else:
        combined_f1 = 0.0
        combined_acc = 0.0

    # Additional flags from attribution
    attr_flagged = (~reliability_mask).sum()

    print(f"\n{'='*60}")
    print(f"COMBINED ABSTENTION (Novelty 2 + Novelty 3)")
    print(f"{'='*60}")
    print(f"  Confidence-only accepted: {conf_count}/{total}")
    print(f"  Attribution-flagged (unreliable): {attr_flagged}/{total}")
    print(f"  Combined accepted: {combined_count}/{total}")
    print(f"  Combined coverage: {combined_coverage:.4f}")
    print(f"  Combined Macro F1: {combined_f1:.4f}")
    print(f"  Combined Accuracy: {combined_acc:.4f}")

    results = {
        "confidence_only_accepted": int(conf_count),
        "attribution_flagged": int(attr_flagged),
        "combined_accepted": int(combined_count),
        "combined_coverage": float(combined_coverage),
        "combined_macro_f1": float(combined_f1),
        "combined_accuracy": float(combined_acc),
    }

    return combined_mask, results
