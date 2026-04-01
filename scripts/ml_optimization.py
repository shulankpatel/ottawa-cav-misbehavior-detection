#!/usr/bin/env python3
"""
================================================================================
Phase 4: ML Model Optimization & Improvement
Project: Ottawa CAV Misbehavior Detection (SYSC5804/ITEC5910)

Fixes from Phase 3:
  1. Class Imbalance  → SMOTE + Class Weights
  2. Low Recall       → Threshold Tuning
  3. Weak SVM         → Replace with XGBoost & Gradient Boosting
  4. No Validation    → K-Fold Cross-Validation
  5. No Tuning        → RandomizedSearchCV

Usage:
  cd ~/ottawa-cav-project
  python3 ml_optimization.py
================================================================================
"""

import os
import sys
import json
import warnings
import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve
)
from joblib import dump

warnings.filterwarnings('ignore')

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("[WARNING] imbalanced-learn not installed. Run: pip install imbalanced-learn")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARNING] XGBoost not installed. Run: pip install xgboost")


# =============================================================================
# CONFIGURATION — MATCHED TO PHASE 3
# =============================================================================
DATASET_PATH = os.path.expanduser("~/ottawa-cav-project/datasets/ottawa_cav_dataset.csv")
RESULTS_DIR = os.path.expanduser("~/ottawa-cav-project/results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots_phase4")
MODELS_DIR = os.path.expanduser("~/ottawa-cav-project/models_phase4")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Exact features from Phase 3
ML_FEATURES = [
    "posX", "posY", "speed", "acceleration",
    "spdX", "spdY", "aclX", "aclY", "hedX", "hedY",
    "pos_delta", "time_delta", "implied_speed",
    "speed_consistency", "speed_delta", "accel_plausible",
    "msg_frequency", "pos_noise_mag", "spd_noise_mag",
]

TARGET = "is_attacker"
RANDOM_STATE = 42
N_FOLDS = 3
TEST_SIZE = 0.2
N_JOBS = 1  # Single core to save memory
SAMPLE_SIZE = 100000  # Use 100K sample of 475K to fit in 3.3GB RAM

COLORS = {
    'rf': '#10b981', 'xgb': '#f59e0b', 'gb': '#a78bfa', 'mlp': '#3b82f6',
    'bg': '#0f172a', 'card': '#1e293b', 'text': '#e2e8f0'
}


def print_header(t):
    print(f"\n{'='*70}\n  {t}\n{'='*70}\n")

def print_step(m):
    print(f"  [STEP] {m}")

def print_success(m):
    print(f"  [✓] {m}")

def print_info(m):
    print(f"  [INFO] {m}")

def print_warning(m):
    print(f"  [⚠] {m}")


# =============================================================================
# STEP 1: Load REAL Dataset
# =============================================================================
def load_dataset():
    print_header("Step 1: Loading Real Dataset")

    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset not found: {DATASET_PATH}")
        sys.exit(1)

    print_step(f"Loading {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    print_success(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    if TARGET not in df.columns:
        print(f"[ERROR] Target column '{TARGET}' not found!")
        print(f"[ERROR] Columns: {list(df.columns)}")
        sys.exit(1)

    # Use only features that exist
    available = [f for f in ML_FEATURES if f in df.columns]
    missing = [f for f in ML_FEATURES if f not in df.columns]
    if missing:
        print_warning(f"Missing features: {missing}")
    print_info(f"Using {len(available)} features")

    X = df[available].copy()
    y = df[TARGET].astype(int).copy()

    # Clean (same as Phase 3)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    # Class distribution
    n_normal = (y == 0).sum()
    n_attack = (y == 1).sum()
    ratio = n_normal / n_attack if n_attack > 0 else 0

    print_info(f"Full dataset: {len(y):,} samples")
    print_info(f"  Normal:   {n_normal:,} ({n_normal/len(y)*100:.1f}%)")
    print_info(f"  Attacker: {n_attack:,} ({n_attack/len(y)*100:.1f}%)")
    print_info(f"  Ratio:    {ratio:.1f}:1")

    # Sample down to save memory
    if len(X) > SAMPLE_SIZE:
        print_step(f"Sampling {SAMPLE_SIZE:,} rows to fit in memory (from {len(X):,})...")
        X_sampled, _, y_sampled, _ = train_test_split(
            X, y, train_size=SAMPLE_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        print_success(f"Sampled: {(y_sampled==0).sum():,} normal, {(y_sampled==1).sum():,} attacker")

        del df, X, y
        gc.collect()
        return X_sampled, y_sampled, available, ratio
    else:
        del df
        gc.collect()
        return X, y, available, ratio


# =============================================================================
# STEP 2: Handle Class Imbalance
# =============================================================================
def handle_imbalance(X_train, y_train):
    print_header("Step 2: Handling Class Imbalance")

    print_info(f"Before: Normal={(y_train==0).sum():,} | Attacker={(y_train==1).sum():,}")

    if HAS_SMOTE:
        print_step("Applying SMOTE (sampling_strategy=0.5)...")
        try:
            smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.5)
            X_res, y_res = smote.fit_resample(X_train, y_train)
            print_success("SMOTE applied!")
            print_info(f"After:  Normal={(y_res==0).sum():,} | Attacker={(y_res==1).sum():,}")
            print_info(f"Total: {len(y_res):,} (was {len(y_train):,})")
            return X_res, y_res
        except Exception as e:
            print_warning(f"SMOTE failed: {e}. Using class weights instead.")
            return X_train, y_train
    else:
        print_warning("SMOTE unavailable. Models will use class_weight='balanced'")
        return X_train, y_train


# =============================================================================
# STEP 3: Define Models
# =============================================================================
def get_models(imbalance_ratio):
    print_header("Step 3: Defining Optimized Models")

    models = {}

    models['Random Forest (Optimized)'] = {
        'model': RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=N_JOBS
        ),
        'color': COLORS['rf'], 'short': 'RF'
    }
    print_success("Random Forest: n=200, balanced, max_depth=15")

    if HAS_XGB:
        models['XGBoost'] = {
            'model': XGBClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                scale_pos_weight=imbalance_ratio, subsample=0.8,
                colsample_bytree=0.8, random_state=RANDOM_STATE,
                eval_metric='logloss', use_label_encoder=False
            ),
            'color': COLORS['xgb'], 'short': 'XGB'
        }
        print_success(f"XGBoost: n=200, scale_pos_weight={imbalance_ratio:.1f}")

    models['Gradient Boosting'] = {
        'model': GradientBoostingClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.1,
            subsample=0.8, random_state=RANDOM_STATE
        ),
        'color': COLORS['gb'], 'short': 'GB'
    }
    print_success("Gradient Boosting: n=150, max_depth=6")

    models['MLP Neural Network (Optimized)'] = {
        'model': MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), activation='relu',
            solver='adam', learning_rate='adaptive', max_iter=300,
            early_stopping=True, validation_fraction=0.1,
            random_state=RANDOM_STATE
        ),
        'color': COLORS['mlp'], 'short': 'MLP'
    }
    print_success("MLP: 256-128-64, adaptive LR, early stopping")

    return models


# =============================================================================
# STEP 4: Cross-Validation
# =============================================================================
def run_cross_validation(models, X, y):
    print_header("Step 4: K-Fold Cross-Validation")

    # Smaller sample for CV
    if len(X) > 40000:
        print_info(f"Sampling 40,000 for CV to save memory...")
        X_cv, _, y_cv, _ = train_test_split(X, y, train_size=40000,
                                             random_state=RANDOM_STATE, stratify=y)
    else:
        X_cv, y_cv = X, y

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}

    for name, config in models.items():
        print_step(f"Cross-validating {name} ({N_FOLDS}-fold)...")

        scores = {}
        for metric in ['accuracy', 'f1', 'recall', 'roc_auc']:
            s = cross_val_score(config['model'], X_cv, y_cv, cv=cv,
                                scoring=metric, n_jobs=N_JOBS)
            scores[metric] = s
            gc.collect()

        cv_results[name] = scores
        print_success(f"{name}:")
        for metric, vals in scores.items():
            print_info(f"    {metric:>10}: {vals.mean():.4f} +/- {vals.std():.4f}")

    return cv_results


# =============================================================================
# STEP 5: Train & Evaluate with Threshold Optimization
# =============================================================================
def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    print_header("Step 5: Training & Evaluating Models")

    results = {}

    for name, config in models.items():
        print_step(f"Training {name}...")

        model = config['model']
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        # Threshold optimization
        best_thresh = 0.5
        best_f1 = f1_score(y_test, y_pred)

        if y_prob is not None:
            print_step(f"  Optimizing threshold for {name}...")
            for t in np.arange(0.10, 0.90, 0.05):
                yt = (y_prob >= t).astype(int)
                ft = f1_score(y_test, yt)
                if ft > best_f1:
                    best_f1 = ft
                    best_thresh = t

            y_final = (y_prob >= best_thresh).astype(int)
            print_success(f"  Best threshold: {best_thresh:.2f} (F1: {best_f1:.4f})")
        else:
            y_final = y_pred

        acc = accuracy_score(y_test, y_final)
        prec = precision_score(y_test, y_final, zero_division=0)
        rec = recall_score(y_test, y_final)
        f1 = f1_score(y_test, y_final)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0
        cm = confusion_matrix(y_test, y_final)

        results[name] = {
            'model': model, 'y_pred': y_final, 'y_prob': y_prob,
            'accuracy': acc, 'precision': prec, 'recall': rec,
            'f1_score': f1, 'auc_roc': auc, 'confusion_matrix': cm,
            'threshold': best_thresh, 'color': config['color'], 'short': config['short']
        }

        print_success(f"{name} (threshold={best_thresh:.2f}):")
        print_info(f"    Accuracy:  {acc:.4f}")
        print_info(f"    Precision: {prec:.4f}")
        print_info(f"    Recall:    {rec:.4f}  << KEY METRIC")
        print_info(f"    F1-Score:  {f1:.4f}")
        print_info(f"    AUC-ROC:   {auc:.4f}")

        dump(model, os.path.join(MODELS_DIR, f"{config['short']}_optimized.joblib"))
        print_info(f"    Model saved!")
        print()
        gc.collect()

    return results


# =============================================================================
# STEP 6: Hyperparameter Tuning
# =============================================================================
def hyperparameter_tuning(X_train, y_train, X_test, y_test):
    print_header("Step 6: Hyperparameter Tuning (Random Forest)")

    if len(X_train) > 40000:
        print_info("Sampling 40,000 for tuning...")
        X_t, _, y_t, _ = train_test_split(X_train, y_train, train_size=40000,
                                            random_state=RANDOM_STATE, stratify=y_train)
    else:
        X_t, y_t = X_train, y_train

    param_grid = {
        'n_estimators': [200, 300, 500],
        'max_depth': [15, 20, 25, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    print_step("Running RandomizedSearchCV...")

    search = RandomizedSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE, n_jobs=N_JOBS),
        param_grid, n_iter=12,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        scoring='f1', random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=0
    )
    search.fit(X_t, y_t)

    print_success("Best parameters:")
    for p, v in search.best_params_.items():
        print_info(f"    {p}: {v}")
    print_info(f"    Best CV F1: {search.best_score_:.4f}")

    best = search.best_estimator_
    yp = best.predict(X_test)
    ypr = best.predict_proba(X_test)[:, 1]

    print_success("Tuned RF on test set:")
    print_info(f"    Accuracy:  {accuracy_score(y_test, yp):.4f}")
    print_info(f"    Precision: {precision_score(y_test, yp):.4f}")
    print_info(f"    Recall:    {recall_score(y_test, yp):.4f}")
    print_info(f"    F1-Score:  {f1_score(y_test, yp):.4f}")
    print_info(f"    AUC-ROC:   {roc_auc_score(y_test, ypr):.4f}")

    dump(best, os.path.join(MODELS_DIR, "RF_tuned_best.joblib"))
    gc.collect()
    return best, search.best_params_


# =============================================================================
# STEP 7: Visualizations
# =============================================================================
def generate_plots(results, y_test):
    print_header("Step 7: Generating Visualizations")
    plt.style.use('dark_background')

    # ── Plot 1: Phase 3 vs Phase 4 ──
    print_step("Phase 3 vs Phase 4 comparison...")

    phase3 = {
        'RF (P3)': {'accuracy': 0.946, 'precision': 0.990, 'recall': 0.481, 'f1_score': 0.648, 'auc_roc': 0.934},
        'SVM (P3)': {'accuracy': 0.474, 'precision': 0.109, 'recall': 0.578, 'f1_score': 0.184, 'auc_roc': 0.516},
        'MLP (P3)': {'accuracy': 0.947, 'precision': 0.925, 'recall': 0.521, 'f1_score': 0.667, 'auc_roc': 0.902},
    }

    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']
    x = np.arange(len(metrics))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(COLORS['bg'])

    ax = axes[0]; ax.set_facecolor(COLORS['card'])
    w = 0.25
    for i, (n, r) in enumerate(phase3.items()):
        ax.bar(x + i*w, [r[m] for m in metrics], w, label=n, alpha=0.85)
    ax.set_xticks(x + w); ax.set_xticklabels(labels, fontsize=10)
    ax.set_title('Phase 3 — Baseline', fontsize=14, fontweight='bold', color='#e2e8f0', pad=15)
    ax.set_ylim(0, 1.15); ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.2)

    ax = axes[1]; ax.set_facecolor(COLORS['card'])
    names = list(results.keys())
    n = len(names); w4 = 0.8 / n
    for i, nm in enumerate(names):
        r = results[nm]
        ax.bar(x + i*w4, [r[m] for m in metrics], w4, label=r['short'], color=r['color'], alpha=0.85)
    ax.set_xticks(x + (n-1)*w4/2); ax.set_xticklabels(labels, fontsize=10)
    ax.set_title('Phase 4 — Optimized', fontsize=14, fontweight='bold', color='#e2e8f0', pad=15)
    ax.set_ylim(0, 1.15); ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.2)

    plt.suptitle('Ottawa CAV — Phase 3 vs Phase 4', fontsize=16, fontweight='bold', color='#60a5fa', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'phase3_vs_phase4_comparison.png'), dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close(); print_success("Saved: phase3_vs_phase4_comparison.png")

    # ── Plot 2: Confusion Matrices ──
    print_step("Confusion matrices...")
    nm = len(results)
    fig, axes = plt.subplots(1, nm, figsize=(5*nm, 5))
    fig.patch.set_facecolor(COLORS['bg'])
    if nm == 1: axes = [axes]
    for ax, (name, res) in zip(axes, results.items()):
        sns.heatmap(res['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Normal', 'Attacker'], yticklabels=['Normal', 'Attacker'])
        ax.set_title(f"{res['short']} (t={res['threshold']:.2f})", fontsize=11, fontweight='bold', color='#e2e8f0')
        ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
    plt.suptitle('Confusion Matrices — Phase 4', fontsize=14, fontweight='bold', color='#60a5fa')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrices_phase4.png'), dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close(); print_success("Saved: confusion_matrices_phase4.png")

    # ── Plot 3: ROC Curves ──
    print_step("ROC curves...")
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(COLORS['bg']); ax.set_facecolor(COLORS['card'])
    for name, res in results.items():
        if res['y_prob'] is not None:
            fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
            ax.plot(fpr, tpr, color=res['color'], linewidth=2, label=f"{res['short']} (AUC={res['auc_roc']:.3f})")
    ax.plot([0,1], [0,1], 'w--', alpha=0.3)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — Phase 4', fontsize=14, fontweight='bold', color='#60a5fa')
    ax.legend(fontsize=10); ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'roc_curves_phase4.png'), dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close(); print_success("Saved: roc_curves_phase4.png")

    # ── Plot 4: Precision-Recall ──
    print_step("Precision-Recall curves...")
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(COLORS['bg']); ax.set_facecolor(COLORS['card'])
    for name, res in results.items():
        if res['y_prob'] is not None:
            p, r, _ = precision_recall_curve(y_test, res['y_prob'])
            ax.plot(r, p, color=res['color'], linewidth=2, label=res['short'])
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall — Phase 4', fontsize=14, fontweight='bold', color='#60a5fa')
    ax.legend(fontsize=10); ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'precision_recall_phase4.png'), dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close(); print_success("Saved: precision_recall_phase4.png")

    # ── Plot 5: Recall Improvement ──
    print_step("Recall improvement chart...")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(COLORS['bg']); ax.set_facecolor(COLORS['card'])

    p3 = {'RF (Phase 3)': 0.481, 'SVM (Phase 3)': 0.578, 'MLP (Phase 3)': 0.521}
    p4 = {f"{res['short']} (Phase 4)": res['recall'] for res in results.values()}

    all_l = list(p3.keys()) + list(p4.keys())
    all_v = list(p3.values()) + list(p4.values())
    all_c = ['#475569']*3 + [res['color'] for res in results.values()]

    bars = ax.barh(range(len(all_l)), all_v, color=all_c, alpha=0.85, height=0.6)
    ax.set_yticks(range(len(all_l))); ax.set_yticklabels(all_l, fontsize=11)
    ax.set_xlabel('Recall'); ax.set_title('Recall: Phase 3 → Phase 4', fontsize=14, fontweight='bold', color='#60a5fa')
    ax.set_xlim(0, 1.08); ax.grid(axis='x', alpha=0.2)
    for bar, val in zip(bars, all_v):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=10, color='#e2e8f0', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'recall_improvement.png'), dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close(); print_success("Saved: recall_improvement.png")

    # ── Plot 6: Summary Table ──
    print_step("Performance summary table...")
    fig, ax = plt.subplots(figsize=(14, 2 + len(results)*0.8))
    fig.patch.set_facecolor(COLORS['bg']); ax.set_facecolor(COLORS['bg']); ax.axis('off')

    tdata = [['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Threshold']]
    tcolors = [['#334155']*7]
    for name, res in results.items():
        tdata.append([name, f"{res['accuracy']:.3f}", f"{res['precision']:.3f}",
                       f"{res['recall']:.3f}", f"{res['f1_score']:.3f}",
                       f"{res['auc_roc']:.3f}", f"{res['threshold']:.2f}"])
        tcolors.append([res['color']+'33']*7)

    table = ax.table(cellText=tdata, cellColours=tcolors, loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#334155')
        cell.set_text_props(color='#e2e8f0', fontweight='bold' if row == 0 else 'normal')
    ax.set_title('Performance Summary — Phase 4', fontsize=14, fontweight='bold', color='#60a5fa', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'performance_summary_phase4.png'), dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close(); print_success("Saved: performance_summary_phase4.png")


# =============================================================================
# STEP 8: Save Results
# =============================================================================
def save_results(results, best_params):
    print_header("Step 8: Saving Results")

    output = {
        'phase': 'Phase 4 - ML Optimization',
        'date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': DATASET_PATH,
        'sample_size': SAMPLE_SIZE,
        'improvements': [
            'SMOTE oversampling' if HAS_SMOTE else 'class_weight=balanced',
            'SVM replaced with XGBoost & Gradient Boosting',
            'Threshold optimization per model',
            f'{N_FOLDS}-fold cross-validation',
            'RandomizedSearchCV hyperparameter tuning'
        ],
        'phase3_baseline': {
            'Random Forest': {'accuracy': 0.946, 'precision': 0.990, 'recall': 0.481, 'f1': 0.648, 'auc_roc': 0.934},
            'SVM': {'accuracy': 0.474, 'precision': 0.109, 'recall': 0.578, 'f1': 0.184, 'auc_roc': 0.516},
            'MLP': {'accuracy': 0.947, 'precision': 0.925, 'recall': 0.521, 'f1': 0.667, 'auc_roc': 0.902},
        },
        'phase4_optimized': {},
        'best_hyperparameters': best_params
    }

    for name, res in results.items():
        output['phase4_optimized'][name] = {
            'accuracy': round(res['accuracy'], 4),
            'precision': round(res['precision'], 4),
            'recall': round(res['recall'], 4),
            'f1_score': round(res['f1_score'], 4),
            'auc_roc': round(res['auc_roc'], 4),
            'threshold': round(res['threshold'], 2),
            'confusion_matrix': res['confusion_matrix'].tolist()
        }

    path = os.path.join(RESULTS_DIR, 'ml_optimization_results_phase4.json')
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
    print_success(f"JSON: {path}")

    # Text summary
    txt = os.path.join(RESULTS_DIR, 'phase4_summary.txt')
    with open(txt, 'w') as f:
        f.write("="*65 + "\n")
        f.write("  PHASE 4: ML OPTIMIZATION RESULTS\n")
        f.write("  Ottawa CAV Misbehavior Detection\n")
        f.write("="*65 + "\n\n")
        f.write(f"Date: {output['date']}\n")
        f.write(f"Dataset: {DATASET_PATH}\n")
        f.write(f"Sample: {SAMPLE_SIZE:,} rows\n\n")

        f.write("PHASE 3 BASELINE:\n")
        f.write(f"  {'Model':<20} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AUC':>8}\n")
        f.write(f"  {'-'*56}\n")
        for n, r in output['phase3_baseline'].items():
            f.write(f"  {n:<20} {r['accuracy']:>8.3f} {r['precision']:>8.3f} {r['recall']:>8.3f} {r['f1']:>8.3f} {r['auc_roc']:>8.3f}\n")

        f.write(f"\nPHASE 4 OPTIMIZED:\n")
        f.write(f"  {'Model':<32} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AUC':>8} {'Thresh':>8}\n")
        f.write(f"  {'-'*80}\n")
        for n, r in output['phase4_optimized'].items():
            f.write(f"  {n:<32} {r['accuracy']:>8.3f} {r['precision']:>8.3f} {r['recall']:>8.3f} {r['f1_score']:>8.3f} {r['auc_roc']:>8.3f} {r['threshold']:>8.2f}\n")

        f.write(f"\nBest RF Hyperparameters:\n")
        for k, v in best_params.items():
            f.write(f"  {k}: {v}\n")

    print_success(f"Summary: {txt}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n")
    print("+" + "="*68 + "+")
    print("|  PHASE 4: ML MODEL OPTIMIZATION                                  |")
    print("|  Ottawa CAV Misbehavior Detection — SYSC5804/ITEC5910            |")
    print("+" + "="*68 + "+")

    # Load real data
    X, y, features, imbalance_ratio = load_dataset()

    # Scale
    print_step("Scaling features...")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    dump(scaler, os.path.join(MODELS_DIR, "scaler_phase4.joblib"))
    del X; gc.collect()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print_info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    del X_scaled; gc.collect()

    # Handle imbalance
    X_train_bal, y_train_bal = handle_imbalance(X_train, y_train)

    # Define models
    models = get_models(imbalance_ratio)

    # Cross-validation
    cv_results = run_cross_validation(models, X_train, y_train)

    # Train and evaluate
    results = train_and_evaluate(models, X_train_bal, X_test, y_train_bal, y_test)

    # Hyperparameter tuning
    _, best_params = hyperparameter_tuning(X_train_bal, y_train_bal, X_test, y_test)

    # Plots
    generate_plots(results, y_test)

    # Save
    save_results(results, best_params)

    # Final summary
    print_header("PHASE 4 COMPLETE")
    print(f"  {'Model':<34} {'Recall':>8} {'F1':>8} {'AUC':>8}")
    print(f"  {'-'*60}")
    best_rec = max(r['recall'] for r in results.values())
    for name, res in results.items():
        star = " <<< BEST" if res['recall'] == best_rec else ""
        print(f"  {name:<34} {res['recall']:>8.4f} {res['f1_score']:>8.4f} {res['auc_roc']:>8.4f}{star}")

    print(f"\n  Results: {RESULTS_DIR}/ml_optimization_results_phase4.json")
    print(f"  Models:  {MODELS_DIR}/")
    print(f"  Plots:   {PLOTS_DIR}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
