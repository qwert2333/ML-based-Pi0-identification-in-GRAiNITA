"""BDT Training and Evaluation Module."""

import logging
import os
import matplotlib.pyplot as plt
import pandas as pd
from configs.paths import FilePaths
from models.bdt_models import BDTClassifier
from src.evaluation import evaluate_bdt

DEFAULT_BDT_CONFIG = {
    "n_estimators": 3000,
    "learning_rate": 0.03,
    "early_stopping_rounds": 50,
    "max_depth": 4,
    "min_child_weight": 10,
    "gamma": 0.2,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "eval_metric": "logloss",
    "tree_method": "hist",
}

def evaluate_and_plot_feature_importance(
    model, feature_names: list, run_dir: str, logger=None, working_point: float = 0.5
):
    """Extracts BDT feature importances, saves them to CSV, and generates a horizontal bar chart plot."""
    # 1. Extract importances cleanly across XGBoost, LightGBM, or Scikit-Learn APIs
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "get_score"):  # XGBoost Booster API fallback
        score_dict = model.get_score(importance_type="gain")
        importances = [
            score_dict.get(
                f"f{i}", score_dict.get(name, 0.0)
            )
            for i, name in enumerate(feature_names)
        ]
    else:
        if logger:
            logger.warning(
                "Model does not expose standard feature_importances_ attribute."
            )
        return None

    # 2. Build sorted DataFrame
    df_imp = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values(by="importance", ascending=True)

    # Save to CSV (descending order)
    csv_path = os.path.join(run_dir, "feature_importance.csv")
    df_imp.sort_values(by="importance", ascending=False).to_csv(
        csv_path, index=False
    )

    if logger:
        logger.info(f"Saved feature importances table to: {csv_path}")
        logger.info("\nTop Feature Importances:\n" + df_imp.sort_values(by="importance", ascending=False).head(10).to_string(index=False))

    # 3. Plot Horizontal Bar Chart
    plt.figure(figsize=(9, max(4, len(feature_names) * 0.4)))
    bars = plt.barh(
        df_imp["feature"],
        df_imp["importance"],
        color="#2b5c8f",
        edgecolor="black",
        alpha=0.85,
    )

    plt.xlabel("Importance (Gain / Weight)", fontsize=11, fontweight="bold")
    plt.ylabel("Features", fontsize=11, fontweight="bold")
    plt.title("BDT Feature Importance", fontsize=13, fontweight="bold", pad=12)
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()

    # Save Plot
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, "feature_importance.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    if logger:
        logger.info(f"Saved feature importance plot to: {plot_path}")

    return df_imp

def run_bdt_training(
    version_tag: str = "30-80GeV_var_angle_bdt_extended",
    train_dir: str = None,
    val_dir: str = None,
    test_dir: str = None,
    run_dir: str = None,
    config_dict: dict = None,
    prefix: str = "clue_gamma_pi0",
    working_point: float = 0.5,
    early_stopping_rounds: int = 50,
    verbose: int = 50,
    logger: logging.Logger = None,
    feature_cols: list = None,
) -> dict:
    """Executes BDT dataset loading, training, checkpointing, and evaluation."""
    def log(msg: str):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    # Resolve paths automatically using FilePaths class
    paths = FilePaths()
    paths.ensure_dirs()

    train_dir = train_dir or paths.get_split_dir("bdt", version_tag, "train")
    val_dir = val_dir or paths.get_split_dir("bdt", version_tag, "val")
    test_dir = test_dir or paths.get_split_dir("bdt", version_tag, "test")
    run_dir = run_dir or paths.get_run_dir("bdt", version_tag)

    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(
            f"Training/Validation directory missing:\n Train: {train_dir}\n Val: {val_dir}"
        )

    config = config_dict if config_dict is not None else DEFAULT_BDT_CONFIG

    log(f"--> Initializing BDT Classifier with early stopping rounds = {early_stopping_rounds}")
    bdt = BDTClassifier(
        config_dict=config,
        early_stopping_rounds=early_stopping_rounds,
        feature_cols=feature_cols,
    )
    if feature_cols is not None:
        log(f"--> Restricting BDT inputs to {len(feature_cols)} features: {list(feature_cols)}")

    log(f"--> Starting BDT training using:\n  * Train Dir: {train_dir}\n  * Val Dir:   {val_dir}")
    bdt.train(train_dir=train_dir, val_dir=val_dir, verbose=verbose)

    model_save_path = os.path.join(run_dir, f"{prefix}_bdt_model.joblib")
    bdt.save(model_save_path)
    log(f"--> Saved fitted BDT model checkpoint to: {model_save_path}")

    # Evaluation
    metrics = {}
    if test_dir and os.path.exists(test_dir):
        log(f"\n--> Loading test set from: {test_dir}")
        X_test, y_test, _ = bdt.load_dataset_from_chunks(test_dir)

        if hasattr(bdt.model, "feature_names_in_"):
            expected_features = bdt.model.feature_names_in_
            missing = [f for f in expected_features if f not in X_test.columns]
            if missing:
                raise ValueError(f"Test dataset is missing features: {missing}")
            X_test = X_test[expected_features]
            log(f"--> Aligned test dataset to {len(expected_features)} model features.")


        evaluate_and_plot_feature_importance(
        model=bdt.model,
        feature_names=expected_features,  # e.g., X_train.columns or list of feature strings
        run_dir=run_dir,
        logger=logger,
        working_point=working_point,)

        log("--> Running linear-scale evaluation...")
        metrics["linear"] = evaluate_bdt(
            model=bdt.model,
            X_test=X_test,
            y_test=y_test,
            output_dir=plots_dir,
            prefix=prefix,
            log_roc=False,
            working_point=working_point,
        )

        log("--> Running log-scale evaluation...")
        metrics["log"] = evaluate_bdt(
            model=bdt.model,
            X_test=X_test,
            y_test=y_test,
            output_dir=plots_dir,
            prefix=prefix,
            log_roc=True,
            working_point=working_point,
        )
    else:
        log(f"Warning: Test directory '{test_dir}' not found. Skipping evaluation.")

    return metrics


if __name__ == "__main__":
    run_bdt_training(version_tag="30-80GeV_var_angle_bdt_extended")