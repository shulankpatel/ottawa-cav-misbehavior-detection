#!/usr/bin/env python3
"""
ML-Based Misbehavior Detection for Ottawa CAV Dataset
Project: 5G Vehicle Misbehavior Detection - SYSC5804/ITEC5910

Trains and evaluates 3 ML models:
  1. Random Forest
  2. Support Vector Machine (SVM)
  3. Multi-Layer Perceptron (MLP) Neural Network

Usage:
    python3 ml_misbehavior_detection.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
import joblib
import json

DATASET_PATH = os.path.expanduser("~/ottawa-cav-project/datasets/ottawa_cav_dataset.csv")
RESULTS_DIR = os.path.expanduser("~/ottawa-cav-project/results")
MODELS_DIR = os.path.expanduser("~/ottawa-cav-project/models")
PLOTS_DIR = os.path.expanduser("~/ottawa-cav-project/results/plots")

ML_FEATURES = [
    "posX", "posY", "speed", "acceleration",
    "spdX", "spdY", "aclX", "aclY", "hedX", "hedY",
    "pos_delta", "time_delta", "implied_speed",
    "speed_consistency", "speed_delta", "accel_plausible",
    "msg_frequency", "pos_noise_mag", "spd_noise_mag",
]

TARGET = "is_attacker"


def load_and_prepare_data():
    print("[STEP 1] Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"  Loaded {len(df)} messages from {df['sender_id'].nunique()} vehicles")
    print(f"  Attackers: {df['is_attacker'].sum()} messages ({df['is_attacker'].mean()*100:.1f}%)")
    print(f"  Normal:    {(~df['is_attacker'].astype(bool)).sum()} messages")

    available = [f for f in ML_FEATURES if f in df.columns]
    missing = [f for f in ML_FEATURES if f not in df.columns]
    if missing:
        print(f"  [WARNING] Missing features: {missing}")
    print(f"  Using {len(available)} features")

    X = df[available].copy()
    y = df[TARGET].astype(int).copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    return X, y, available, df


def main():
    print("=" * 60)
    print("  ML Misbehavior Detection - Ottawa CAV Dataset")
    print("  SYSC5804/ITEC5910 - 5G Vehicle Misbehavior Detection")
    print("=" * 60)

    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset not found: {DATASET_PATH}")
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    X, y, feature_names, df = load_and_prepare_data()

    print("\n[INFO] Splitting data: 80% train, 20% test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)} samples ({y_train.sum()} attackers)")
    print(f"  Test:  {len(X_test)} samples ({y_test.sum()} attackers)")

    # Create smaller subset for SVM (too slow on full data)
    X_train_small, _, y_train_small, _ = train_test_split(
        X_train, y_train, train_size=50000, random_state=42, stratify=y_train
    )
    print(f"  SVM subset: {len(X_train_small)} samples")

    # Scale features
    print("\n[STEP 2] Scaling features...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    X_small_sc = scaler.transform(X_train_small)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
        ),
        "SVM": SVC(
            kernel="rbf", C=1.0, gamma="scale", random_state=42,
            probability=True, max_iter=1000, cache_size=1000
        ),
        "MLP Neural Network": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), max_iter=300,
            random_state=42, early_stopping=True, validation_fraction=0.1
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\n[STEP 3] Training {name}...")

        if name == "Random Forest":
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        elif name == "SVM":
            model.fit(X_small_sc, y_train_small)
            y_pred = model.predict(X_test_sc)
            y_prob = model.predict_proba(X_test_sc)[:, 1]
        else:
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)
            y_prob = model.predict_proba(X_test_sc)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.0

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "auc_roc": round(auc, 4),
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  AUC-ROC:   {auc:.4f}")

        joblib.dump(model, os.path.join(MODELS_DIR, f"{name.replace(' ', '_')}.pkl"))

        print(f"\n  Classification Report ({name}):")
        print(classification_report(y_test, y_pred, target_names=["Normal", "Attacker"]))

    # Feature importance (Random Forest)
    rf_model = models["Random Forest"]
    feat_imp = pd.Series(rf_model.feature_importances_, index=feature_names)
    feat_imp = feat_imp.sort_values(ascending=False)
    print("\n[INFO] Top 10 Feature Importances (Random Forest):")
    for feat, imp in feat_imp.head(10).items():
        print(f"  {feat:25s}: {imp:.4f}")

    # ─── Generate Plots ──────────────────────────────────────────────────
    print("\n[STEP 4] Generating plots...")

    # 1. Model Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
    x = np.arange(len(model_names))
    w = 0.15
    for i, metric in enumerate(metrics):
        vals = [results[m][metric] for m in model_names]
        ax.bar(x + i * w, vals, w, label=metric.replace("_", " ").title())
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Ottawa CAV Misbehavior Detection - Model Comparison")
    ax.set_xticks(x + w * 2)
    ax.set_xticklabels(model_names, fontsize=9)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"), dpi=150)
    plt.close()
    print("  Saved: model_comparison.png")

    # 2. Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                    xticklabels=["Normal", "Attacker"],
                    yticklabels=["Normal", "Attacker"])
        axes[i].set_title(name)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
    plt.suptitle("Confusion Matrices - Ottawa CAV Dataset")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrices.png"), dpi=150)
    plt.close()
    print("  Saved: confusion_matrices.png")

    # 3. Feature Importance
    fig, ax = plt.subplots(figsize=(10, 6))
    feat_imp.head(15).plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title("Top 15 Features for Misbehavior Detection (Random Forest)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    print("  Saved: feature_importance.png")

    # 4. Performance Summary Table
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    table_data = []
    for name in model_names:
        r = results[name]
        table_data.append([name, f"{r['accuracy']:.3f}", f"{r['precision']:.3f}",
                          f"{r['recall']:.3f}", f"{r['f1_score']:.3f}", f"{r['auc_roc']:.3f}"])
    table = ax.table(cellText=table_data,
                     colLabels=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"],
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.title("Detection Performance Summary - Ottawa CAV Dataset", fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "performance_summary.png"), dpi=150)
    plt.close()
    print("  Saved: performance_summary.png")

    # ─── Save Results ────────────────────────────────────────────────────
    save_data = {}
    for name, res in results.items():
        save_data[name] = {k: v for k, v in res.items()
                          if k not in ["y_pred", "y_prob"]}
    with open(os.path.join(RESULTS_DIR, "ml_detection_results.json"), "w") as f:
        json.dump(save_data, f, indent=2)

    # ─── Final Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 60)
    best = max(results.items(), key=lambda x: x[1]["f1_score"])
    print(f"\n  Best Model: {best[0]}")
    print(f"  F1-Score:   {best[1]['f1_score']:.4f}")
    print(f"  Precision:  {best[1]['precision']:.4f}")
    print(f"  Recall:     {best[1]['recall']:.4f}")
    print(f"  AUC-ROC:    {best[1]['auc_roc']:.4f}")
    print(f"\n  Output files:")
    print(f"    Results:  {RESULTS_DIR}/ml_detection_results.json")
    print(f"    Models:   {MODELS_DIR}/")
    print(f"    Plots:    {PLOTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
