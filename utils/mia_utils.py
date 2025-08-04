#!/usr/bin/env python3
"""
Shared MIA (Membership Inference Attack) utilities
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_recall_fscore_support, roc_curve
)
import joblib


def get_confidence_scores(model, X, framework="sklearn"):
    """Get confidence scores for MIA"""
    if framework == "sklearn":
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            return np.max(probs, axis=1)
        else:
            raise ValueError("Sklearn model does not support predict_proba")
    else:
        raise ValueError("Unknown framework")


def get_entropy_scores(model, X, framework="sklearn"):
    """Get prediction entropy for MIA"""
    if framework == "sklearn":
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
            return entropy
        else:
            raise ValueError("Sklearn model does not support predict_proba")
    else:
        raise ValueError("Unknown framework")


def get_loss_scores(model, X, y, framework="sklearn"):
    """Get per-sample loss scores (better than confidence for MIA)"""
    print(f"[DEBUG] Getting loss scores using framework: {framework}")

    if framework == "sklearn":
        # For sklearn, use negative log-likelihood
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            # Convert y to class indices if needed
            if len(y.shape) > 1:
                y_indices = np.argmax(y, axis=1)
            else:
                y_indices = y.astype(int)

            # Get probability for true class
            true_class_probs = probs[np.arange(len(y_indices)), y_indices]
            # Return negative log-likelihood
            return -np.log(true_class_probs + 1e-10)
        else:
            raise ValueError("Sklearn model does not support predict_proba")
    else:
        raise ValueError("Unknown framework")


def evaluate_attack_performance(y_true, y_scores, attack_name):
    """Evaluate attack performance with threshold sweeping"""
    # Sweep thresholds
    thresholds = np.linspace(np.min(y_scores), np.max(y_scores), 101)
    best_acc, best_thresh = 0, 0

    for t in thresholds:
        y_pred = (y_scores > t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    # Final prediction with best threshold
    y_pred = (y_scores > best_thresh).astype(int)

    # Calculate metrics
    auc = roc_auc_score(y_true, y_scores)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    return {
        "attack_name": attack_name,
        "best_thresh": best_thresh,
        "accuracy": best_acc,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc": (fpr, tpr),
        "scores": y_scores
    }


def run_comprehensive_attack(model_path, X_member, X_nonmember, y_member=None, y_nonmember=None, framework="sklearn"):
    """Run comprehensive MIA attack with multiple strategies"""
    print(f"[DEBUG] Running comprehensive MIA on: {os.path.basename(model_path)}")

    # Load model
    if framework == "sklearn":
        model = joblib.load(model_path)
    else:
        raise ValueError("Unknown framework")

    results = {}

    # Strategy 1: Confidence-based attack
    try:
        member_conf = get_confidence_scores(model, X_member, framework)
        nonmember_conf = get_confidence_scores(model, X_nonmember, framework)

        y_true = np.concatenate([np.ones(len(X_member)), np.zeros(len(X_nonmember))])
        y_scores = np.concatenate([member_conf, nonmember_conf])

        results['confidence'] = evaluate_attack_performance(y_true, y_scores, "Confidence")
    except Exception as e:
        print(f"[ERROR] Confidence attack failed: {e}")
        results['confidence'] = None

    # Strategy 2: Entropy-based attack
    try:
        member_entropy = get_entropy_scores(model, X_member, framework)
        nonmember_entropy = get_entropy_scores(model, X_nonmember, framework)

        # For entropy, we want LOWER scores for members, so we negate
        y_scores = np.concatenate([-member_entropy, -nonmember_entropy])

        results['entropy'] = evaluate_attack_performance(y_true, y_scores, "Entropy")
    except Exception as e:
        print(f"[ERROR] Entropy attack failed: {e}")
        results['entropy'] = None

    # Strategy 3: Loss-based attack (requires labels)
    if y_member is not None and y_nonmember is not None:
        try:
            member_loss = get_loss_scores(model, X_member, y_member, framework)
            nonmember_loss = get_loss_scores(model, X_nonmember, y_nonmember, framework)

            # For loss, we want LOWER scores for members, so we negate
            y_scores = np.concatenate([-member_loss, -nonmember_loss])

            results['loss'] = evaluate_attack_performance(y_true, y_scores, "Loss")
        except Exception as e:
            print(f"[ERROR] Loss attack failed: {e}")
            results['loss'] = None
    else:
        print("[WARNING] Labels not provided, skipping loss-based attack")
        results['loss'] = None

    return results 