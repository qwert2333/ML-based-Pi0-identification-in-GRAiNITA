import argparse
import os
import sys
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve

# Clean modern project imports
from configs.paths import FilePaths


def evaluate_bdt(
    model,
    X_test,
    y_test,
    output_dir="./plots",
    prefix="bdt_eval",
    log_roc=False,
    working_point=0.5,
    top_n=None,
):
    """Evaluates a trained model on test data:
      1. Score Distributions (Signal pi0 vs. Background gamma)
      2. ROC Curve (Linear or Log Scale for FPR)
      3. Contamination / Confusion Matrix
      4. Feature Importances (Ranking Table & Bar Plot)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Predict signal probabilities (scores for pi0)
    y_scores = model.predict_proba(X_test)[:, 1]

    # Set overall aesthetic style
    sns.set_theme(style="whitegrid", font_scale=1.1)
    model_name = type(model).__name__

    # -------------------------------------------------------------------------
    # 1. BDT Score Distribution Plot
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    sig_scores = y_scores[y_test == 1]
    bkg_scores = y_scores[y_test == 0]

    # Signal = pi0
    ax.hist(
        sig_scores,
        bins=50,
        range=(0, 1),
        density=True,
        histtype="stepfilled",
        alpha=0.4,
        color="royalblue",
        label=r"Signal ($\pi^0$)",
    )
    ax.hist(
        sig_scores,
        bins=50,
        range=(0, 1),
        density=True,
        histtype="step",
        linewidth=1.5,
        color="royalblue",
    )

    # Background = gamma
    ax.hist(
        bkg_scores,
        bins=50,
        range=(0, 1),
        density=True,
        histtype="stepfilled",
        alpha=0.4,
        color="crimson",
        label=r"Background ($\gamma$)",
    )
    ax.hist(
        bkg_scores,
        bins=50,
        range=(0, 1),
        density=True,
        histtype="step",
        linewidth=1.5,
        color="crimson",
    )

    ax.set_xlabel(f"{model_name} Output Score", fontsize=12)
    ax.set_ylabel("Probability Density (Normalized)", fontsize=12)
    ax.set_title(f"{model_name} Classifier Response Distribution", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.legend(frameon=True, facecolor="white")

    dist_path = os.path.join(output_dir, f"{prefix}_score_distribution.pdf")
    plt.savefig(dist_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved score distribution plot to: {dist_path}")

    # -------------------------------------------------------------------------
    # 2. Receiver Operating Characteristic (ROC) Curve
    # -------------------------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))

    if log_roc:
        valid_idx = fpr > 0
        ax.plot(
            fpr[valid_idx],
            tpr[valid_idx],
            color="darkorange",
            lw=2,
            label=f"{model_name} (AUC = {roc_auc:.6f})",
        )
        ax.set_xscale("log")
        ax.set_xlabel("False Positive Rate / Background Efficiency (Log Scale)", fontsize=12)
        ax.set_xlim(1e-6, 1.0)
    else:
        ax.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"{model_name} (AUC = {roc_auc:.6f})",
        )
        ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Classifier")
        ax.set_xlabel("False Positive Rate (Background Efficiency)", fontsize=12)
        ax.set_xlim(0.0, 1.0)

    ax.set_ylabel("True Positive Rate (Signal Efficiency)", fontsize=12)
    ax.set_ylim(0.0, 1.05)
    scale_label = "Log Scale" if log_roc else "Linear Scale"
    ax.set_title(f"ROC Curve ({scale_label})", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, facecolor="white")

    scale_suffix = "log" if log_roc else "linear"
    roc_path = os.path.join(output_dir, f"{prefix}_roc_{scale_suffix}.pdf")
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC plot ({scale_label}) to: {roc_path}")

    # -------------------------------------------------------------------------
    # 3. Contamination / Confusion Matrix
    # -------------------------------------------------------------------------
    y_pred = (y_scores >= working_point).astype(int)

    cm_normalized = confusion_matrix(y_test, y_pred, normalize="true")
    cm_counts = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))

    annot_labels = np.array(
        [
            [f"{val:.2%}\n({count:,})" for val, count in zip(row_val, row_cnt)]
            for row_val, row_cnt in zip(cm_normalized, cm_counts)
        ]
    )

    sns.heatmap(
        cm_normalized,
        annot=annot_labels,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=[r"Pred $\gamma$ (0)", r"Pred $\pi^0$ (1)"],
        yticklabels=[r"True $\gamma$ (0)", r"True $\pi^0$ (1)"],
        ax=ax,
    )

    ax.set_title(
        f"Contamination Matrix (Working Point = {working_point})",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)

    cm_path = os.path.join(output_dir, f"{prefix}_contamination_matrix.pdf")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved contamination matrix plot to: {cm_path}")

    # -------------------------------------------------------------------------
    # 4. Feature Importance Plot
    # -------------------------------------------------------------------------
    df_imp = None
    if hasattr(model, "feature_importances_") or (
        hasattr(model, "get_booster") and hasattr(model.get_booster(), "get_score")
    ):
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            booster = model.get_booster()
            score_dict = booster.get_score(importance_type="weight")
            importances = list(score_dict.values())

        if hasattr(model, "feature_names_in_"):
            feature_names = model.feature_names_in_
        elif hasattr(model, "get_booster") and model.get_booster().feature_names:
            feature_names = model.get_booster().feature_names
        elif isinstance(X_test, pd.DataFrame):
            feature_names = X_test.columns
        else:
            feature_names = [f"Feature_{i}" for i in range(len(importances))]

        df_imp = pd.DataFrame(
            {"Feature": feature_names, "Importance": importances}
        ).sort_values("Importance", ascending=False)

        print("\n--- Variable Importance Ranking ---")
        print(df_imp.to_string(index=False))

        df_imp_plot = df_imp.head(top_n) if top_n and top_n < len(df_imp) else df_imp
        df_imp_plot = df_imp_plot.sort_values("Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(10, max(5, len(df_imp_plot) * 0.4)))
        ax.barh(
            df_imp_plot["Feature"],
            df_imp_plot["Importance"],
            color="skyblue",
            edgecolor="black",
        )
        ax.set_xlabel("Importance", fontsize=12)
        ax.set_ylabel("Feature", fontsize=12)
        ax.set_title(f"{model_name} Feature Importance", fontsize=14, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.7)

        imp_path = os.path.join(output_dir, f"{prefix}_feature_importance.pdf")
        plt.savefig(imp_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved feature importance plot to: {imp_path}")

    return {
        "auc": roc_auc,
        "fpr": fpr,
        "tpr": tpr,
        "confusion_matrix": cm_normalized,
        "feature_importances": df_imp,
    }


def main():
    from src.models.bdt_models import BDTClassifier
    parser = argparse.ArgumentParser(
        description="Evaluate a trained BDT model on test dataset."
    )
    parser.add_argument(
        "-m",
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model file (.joblib)",
    )
    parser.add_argument(
        "-v",
        "--version-tag",
        type=str,
        default="30-80GeV_v2",
        help="Dataset version tag identifier",
    )
    parser.add_argument(
        "-t",
        "--test-dir",
        type=str,
        default=None,
        help="Directory containing test ROOT chunk files",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./plots",
        help="Directory where evaluation plots will be saved",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="bdt_eval",
        help="Prefix for saved figure filenames",
    )
    parser.add_argument(
        "-w",
        "--working-point",
        type=float,
        default=0.5,
        help="BDT score threshold for the contamination matrix (default: 0.5)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Limit feature importance plot to top N features",
    )
    parser.add_argument(
        "--log-roc",
        action="store_true",
        help="Plot ROC curve with logarithmic False Positive Rate axis",
    )
    args = parser.parse_args()

    paths = FilePaths()

    # 1. Resolve test dataset directory
    test_dir = args.test_dir or paths.get_split_dir("bdt", args.version_tag, "test")

    # 2. Load the trained model
    print(f"--> Loading trained model from: {args.model_path}")
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        sys.exit(1)

    model = joblib.load(args.model_path)

    # 3. Load test dataset from ROOT chunks via BDTClassifier wrapper
    print(f"--> Loading test ROOT chunks from: {test_dir}")
    if hasattr(model, "load_dataset_from_chunks"):
        X_test, y_test, _ = model.load_dataset_from_chunks(test_dir)
    else:
        bdt_wrapper = BDTClassifier(config_dict={})
        X_test, y_test, _ = bdt_wrapper.load_dataset_from_chunks(test_dir)

    print(f"--> Loaded {len(X_test)} test samples with {X_test.shape[1]} features.")

    # 4. Run Evaluation
    print("--> Running model evaluation...")
    metrics = evaluate_bdt(
        model=model,
        X_test=X_test,
        y_test=y_test,
        output_dir=args.output_dir,
        prefix=args.prefix,
        log_roc=args.log_roc,
        working_point=args.working_point,
        top_n=args.top_n,
    )

    print("\n--- Summary Performance ---")
    print(f"  * Test ROC AUC : {metrics['auc']:.4f}")
    print(
        f"  * Working Point: {args.working_point} (Signal Eff: {metrics['confusion_matrix'][1,1]:.2%})"
    )


if __name__ == "__main__":
    main()