import argparse
import json
import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


def ensure_dirs():
    os.makedirs("outputs/plots", exist_ok=True)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def load_and_clean_data(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates()

    label_encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    return df, label_encoders


def plot_class_distribution(df, target):
    plt.figure(figsize=(8, 5))
    sns.countplot(x=target, data=df)
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig("outputs/plots/class_distribution.png")
    plt.close()


def _is_binary(y):
    try:
        return len(pd.Series(y).dropna().unique()) == 2
    except Exception:
        return False


def compute_metrics(y_true, y_pred):
    """
    Returns a compact metrics dict you can use in JSON + reporting.
    - If binary: precision/recall/f1 are computed with average='binary'
    - Else: uses weighted average (safe default for imbalance)
    """
    binary = _is_binary(y_true)
    avg = "binary" if binary else "weighted"

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
        # Keep these for completeness / write-up (you can cite class-wise results)
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return metrics


def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,  # fixed random_state values are used throughout to ensure reproducibility
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = compute_metrics(y_test, preds)

    return model, metrics


def save_feature_importance(model, X, top_n=30):
    importances = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values(by="importance", ascending=False)

    importances.head(top_n).to_csv("outputs/feature_importance.csv", index=False)


def semi_supervised_learning(X, y):
    # Create partially-unlabelled scenario (by splitting and pseudo-labelling)
    X_labeled, X_unlabeled, y_labeled, _ = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_labeled, y_labeled)

    pseudo_labels = model.predict(X_unlabeled)

    X_combined = pd.concat([X_labeled, X_unlabeled], axis=0)
    y_combined = pd.concat([y_labeled, pd.Series(pseudo_labels, index=X_unlabeled.index)], axis=0)

    model.fit(X_combined, y_combined)
    return model


def run_xgboost(X_train, y_train, X_test, y_test, tune=False):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("XGBoost not installed — skipping.")
        return None, None, None

    best_params = None

    if tune:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
        }
        base = XGBClassifier(random_state=42, eval_metric="logloss")
        grid = GridSearchCV(base, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
        grid.fit(X_train, y_train)

        model = grid.best_estimator_
        best_params = grid.best_params_
        save_json(best_params, "outputs/xgboost_best_params.json")
    else:
        model = XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss")
        model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = compute_metrics(y_test, preds)

    return model, metrics, best_params


def main(args):
    ensure_dirs()

    df, _ = load_and_clean_data(args.data_path)

    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if args.save_plots:
        plot_class_distribution(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}

    # Random Forest (baseline)
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    results["random_forest"] = {
        "accuracy": rf_metrics["accuracy"],
        "precision": rf_metrics["precision"],
        "recall": rf_metrics["recall"],
        "f1_score": rf_metrics["f1_score"],
    }

    if args.save_feature_importance:
        save_feature_importance(rf_model, X)

    # Semi-supervised (self-training via pseudo-labelling)
    if args.run_semi_supervised:
        semi_model = semi_supervised_learning(X, y)
        semi_preds = semi_model.predict(X_test)
        semi_metrics = compute_metrics(y_test, semi_preds)
        results["semi_supervised_random_forest"] = {
            "accuracy": semi_metrics["accuracy"],
            "precision": semi_metrics["precision"],
            "recall": semi_metrics["recall"],
            "f1_score": semi_metrics["f1_score"],
        }

    # No-financial-features ablation
    if args.run_no_financials:
        # Your dataset uses columns like USD/BTC; also keep "amount" just in case
        blocked_tokens = ("usd", "btc", "amount", "value", "price")
        non_financial_cols = [c for c in X.columns if not any(t in c.lower() for t in blocked_tokens)]
        X_nf = X[non_financial_cols]

        X_train_nf, X_test_nf, y_train_nf, y_test_nf = train_test_split(
            X_nf, y, test_size=0.2, random_state=42
        )

        _, nf_metrics = train_random_forest(X_train_nf, y_train_nf, X_test_nf, y_test_nf)
        results["no_financial_features_random_forest"] = {
            "accuracy": nf_metrics["accuracy"],
            "precision": nf_metrics["precision"],
            "recall": nf_metrics["recall"],
            "f1_score": nf_metrics["f1_score"],
        }

    # XGBoost
    if args.run_xgboost:
        _, xgb_metrics, _ = run_xgboost(
            X_train, y_train, X_test, y_test, tune=args.tune_xgboost
        )
        if xgb_metrics is not None:
            results["xgboost"] = {
                "accuracy": xgb_metrics["accuracy"],
                "precision": xgb_metrics["precision"],
                "recall": xgb_metrics["recall"],
                "f1_score": xgb_metrics["f1_score"],
            }

    save_json(results, "outputs/run_summary.json")

    print("Run complete. Results (key metrics):")
    for model_name, m in results.items():
        print(
            f"{model_name}: "
            f"acc={m['accuracy']:.4f}, "
            f"prec={m['precision']:.4f}, "
            f"rec={m['recall']:.4f}, "
            f"f1={m['f1_score']:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UGRansomware ML pipeline")

    parser.add_argument(
        "--data-path", default="data/data.csv", help="Path to dataset CSV"
    )
    parser.add_argument("--save-plots", action="store_true", help="Save EDA plots")
    parser.add_argument(
        "--save-feature-importance",
        action="store_true",
        help="Save feature importance CSV",
    )
    parser.add_argument(
        "--run-semi-supervised", action="store_true", help="Run semi-supervised experiment"
    )
    parser.add_argument(
        "--run-no-financials",
        action="store_true",
        help="Run no-financial-features ablation",
    )
    parser.add_argument("--run-xgboost", action="store_true", help="Run XGBoost model")
    parser.add_argument(
        "--tune-xgboost", action="store_true", help="Grid search XGBoost"
    )

    main(parser.parse_args())
