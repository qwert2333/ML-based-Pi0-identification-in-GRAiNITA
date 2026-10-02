import glob
import os
import joblib
import pandas as pd
import uproot
import xgboost as xgb


class BDTClassifier:
    """Wrapper class for training and evaluating XGBoost BDT on CLUE shower data."""

    def __init__(self, config_dict=None, early_stopping_rounds=30, feature_cols=None):
        if config_dict is None:
            config_dict = {
                "n_estimators": 1000,
                "max_depth": 5,
                "learning_rate": 0.03,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "eval_metric": "logloss",
                "tree_method": "hist",
            }

        self.early_stopping_rounds = early_stopping_rounds
        self.feature_names_ = None
        # Optional explicit feature list; None keeps every non-target column.
        self.feature_cols = list(feature_cols) if feature_cols is not None else None

        self.model = xgb.XGBClassifier(
            n_estimators=config_dict.get("n_estimators", 1000),
            max_depth=config_dict.get("max_depth", 5),
            learning_rate=config_dict.get("learning_rate", 0.03),
            subsample=config_dict.get("subsample", 0.8),
            colsample_bytree=config_dict.get("colsample_bytree", 0.8),
            eval_metric=config_dict.get("eval_metric", "logloss"),
            tree_method=config_dict.get("tree_method", "hist"),
            early_stopping_rounds=early_stopping_rounds,
            random_state=42,
        )

    @staticmethod
    def load_dataset_from_chunks(
        data_dir: str,
        tree_name: str = "events",
        target_cols: list = ["target_pdg", "target_energy", "label"],
    ):
        """Loads all chunked ROOT files from a directory, skipping any corrupted individual branches."""
        root_files = sorted(glob.glob(os.path.join(data_dir, "*.root")))
        if not root_files:
            raise FileNotFoundError(
                f"No ROOT files found in directory: {data_dir}"
            )

        dfs = []
        for fpath in root_files:
            try:
                with uproot.open(fpath) as f:
                    if tree_name in f:
                        tree = f[tree_name]
                        if tree.num_entries == 0:
                            print(
                                f"Warning: Skipping empty tree in {os.path.basename(fpath)}"
                            )
                            continue

                        # Read branches individually to bypass corrupted baskets
                        valid_branch_data = {}
                        for key in tree.keys():
                            try:
                                valid_branch_data[key] = tree[key].array(
                                    library="np"
                                )
                            except Exception as b_err:
                                print(
                                    f"Warning: Skipping corrupted branch '{key}' in {os.path.basename(fpath)}. Error: {b_err}"
                                )

                        if valid_branch_data:
                            df = pd.DataFrame(valid_branch_data)
                            dfs.append(df)
            except Exception as e:
                print(
                    f"Warning: Could not open file {os.path.basename(fpath)}. Error: {e}"
                )

        if not dfs:
            raise ValueError(
                f"No valid trees named '{tree_name}' could be loaded from {data_dir}"
            )

        full_df = pd.concat(dfs, ignore_index=True)

        # Separate feature matrix X and target label y
        feature_cols = [c for c in full_df.columns if c not in target_cols]
        X = full_df[feature_cols]
        y = full_df["label"].values if "label" in full_df.columns else None
        weights = (
            full_df["weight"].values if "weight" in full_df.columns else None
        )

        return X, y, weights

    def train(
            self,
            train_dir: str,
            val_dir: str,
            tree_name: str = "events",
            verbose: bool = True,
        ):
            """Loads datasets from directories and fits the model."""
            print(f"--> Loading Training Data from: {train_dir}")
            X_train, y_train, w_train = self.load_dataset_from_chunks(
                train_dir, tree_name=tree_name
            )

            print(f"--> Loading Validation Data from: {val_dir}")
            X_val, y_val, w_val = self.load_dataset_from_chunks(
                val_dir, tree_name=tree_name
            )

            # Align features in case corrupted branches were dropped
            common_cols = [c for c in X_train.columns if c in X_val.columns]
            if self.feature_cols is not None:
                missing = [c for c in self.feature_cols if c not in common_cols]
                if missing:
                    raise ValueError(f"Requested BDT features missing from data: {missing}")
                common_cols = list(self.feature_cols)
            X_train = X_train[common_cols]
            X_val = X_val[common_cols]
            self.feature_names_ = list(common_cols)

            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(
                X_train,
                y_train,
                sample_weight=w_train,
                eval_set=eval_set,
                verbose=verbose,
            )

            print("--> BDT Training complete!")

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        """Fits model directly on pandas DataFrames/NumPy arrays."""
        self.feature_names_ = (
            list(X_train.columns) if hasattr(X_train, "columns") else None
        )
        eval_set = (
            [(X_train, y_train), (X_val, y_val)]
            if X_val is not None and y_val is not None
            else None
        )

        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=verbose)

    def predict(self, X):
        """Predicts class labels (0 or 1)."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predicts class probabilities."""
        return self.model.predict_proba(X)

    def save(self, output_path: str):
        """Saves the entire BDTClassifier instance to disk."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(self, output_path)
        print(f"Saved BDT model to: {output_path}")

    @classmethod
    def load(cls, input_path: str) -> "BDTClassifier":
        """Loads a pre-trained BDTClassifier from disk."""
        instance = joblib.load(input_path)
        print(f"Loaded BDT model from: {input_path}")
        return instance