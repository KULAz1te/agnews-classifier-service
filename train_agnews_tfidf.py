"""
EAI 6010 Module 5 - Train a deployable AG News text classifier

This script retrains the traditional NLP model from the Module 3 AG News assignment:
TF-IDF + Logistic Regression.

Why this version:
- It does not need GPU or PyTorch.
- It is much easier to deploy as a microservice than the ULMFiT model.
- It saves a reusable model artifact for FastAPI / Cloud Run deployment.

Run in VS Code terminal:
    python train_agnews_tfidf.py

Optional quick run:
    python train_agnews_tfidf.py --train-per-class 500 --test-per-class 100
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline


SEED = 42
LABEL_MAP: Dict[int, str] = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci_Tech",
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def hf_split_to_df(dataset, split: str) -> pd.DataFrame:
    """Convert a Hugging Face AG News split into a clean pandas DataFrame."""
    df = pd.DataFrame(dataset[split])
    df["label_name"] = df["label"].map(LABEL_MAP)
    return df[["text", "label", "label_name"]]


def stratified_sample(df: pd.DataFrame, n_per_class: int, seed: int) -> pd.DataFrame:
    """Sample the same number of rows from each class."""
    parts = []
    for _, group in df.groupby("label", sort=True):
        n = min(n_per_class, len(group))
        parts.append(group.sample(n=n, random_state=seed))
    return (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def build_model(max_features: int, min_df: int, c_value: float) -> Pipeline:
    """Create the deployable traditional NLP pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=max_features,
            min_df=min_df,
        )),
        ("clf", LogisticRegression(
            C=c_value,
            max_iter=1000,
            n_jobs=-1,
            random_state=SEED,
        )),
    ])


def build_tuned_model() -> GridSearchCV:
    """Create a small hyperparameter search that is still practical on a laptop."""
    pipeline = build_model(max_features=50000, min_df=2, c_value=1.0)
    param_grid = {
        "tfidf__max_features": [30000, 50000],
        "tfidf__min_df": [1, 2],
        "clf__C": [1.0, 2.0, 4.0],
    }
    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-per-class", type=int, default=None)
    parser.add_argument("--test-per-class", type=int, default=None)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--c-value", type=float, default=2.0)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument(
        "--full-data",
        action="store_true",
        help="Use the full AG News train and test splits. This is the default unless sample sizes are provided.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run a small GridSearchCV hyperparameter search before saving the final model.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use the locally cached Hugging Face dataset without checking the Hub.",
    )
    args = parser.parse_args()

    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    set_seed(SEED)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading AG News dataset from Hugging Face...")
    ds = load_dataset("fancyzhx/ag_news")

    full_train_df = hf_split_to_df(ds, "train")
    full_test_df = hf_split_to_df(ds, "test")

    use_full_data = args.full_data or args.train_per_class is None or args.test_per_class is None

    if use_full_data:
        print("Using full AG News train and test splits...")
        train_subset = full_train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
        test_subset = full_test_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    else:
        print("Sampling balanced subsets...")
        train_subset = stratified_sample(full_train_df, args.train_per_class, SEED)
        test_subset = stratified_sample(full_test_df, args.test_per_class, SEED)

    train_df, valid_df = train_test_split(
        train_subset,
        test_size=0.20,
        random_state=SEED,
        stratify=train_subset["label_name"],
    )

    print(f"Training rows:   {len(train_df)}")
    print(f"Validation rows: {len(valid_df)}")
    print(f"Test rows:       {len(test_subset)}")
    print("Class distribution in training data:")
    print(train_df["label_name"].value_counts().sort_index())

    model = (
        build_tuned_model()
        if args.tune
        else build_model(max_features=args.max_features, min_df=args.min_df, c_value=args.c_value)
    )

    print("\nTraining TF-IDF + Logistic Regression model...")
    model.fit(train_df["text"], train_df["label_name"])

    best_params = None
    if args.tune:
        best_params = model.best_params_
        print(f"Best hyperparameters: {best_params}")
        model = model.best_estimator_

    print("Evaluating model...")
    valid_pred = model.predict(valid_df["text"])
    test_pred = model.predict(test_subset["text"])

    valid_acc = accuracy_score(valid_df["label_name"], valid_pred)
    test_acc = accuracy_score(test_subset["label_name"], test_pred)

    report = classification_report(
        test_subset["label_name"],
        test_pred,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(
        test_subset["label_name"],
        test_pred,
        labels=list(LABEL_MAP.values()),
    )

    metrics = {
        "model_name": "TF-IDF + Logistic Regression",
        "dataset": "AG News",
        "labels": LABEL_MAP,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "test_rows": int(len(test_subset)),
        "full_data": bool(use_full_data),
        "tuned": bool(args.tune),
        "best_params": best_params,
        "train_per_class_before_split": None if use_full_data else args.train_per_class,
        "test_per_class": None if use_full_data else args.test_per_class,
        "max_features": args.max_features,
        "min_df": args.min_df,
        "c_value": args.c_value,
        "validation_accuracy": float(valid_acc),
        "test_accuracy": float(test_acc),
        "classification_report": report,
        "confusion_matrix_labels": list(LABEL_MAP.values()),
        "confusion_matrix": cm.tolist(),
    }

    artifact = {
        "model": model,
        "label_map": LABEL_MAP,
        "metrics": metrics,
    }

    model_path = out_dir / "agnews_tfidf_logreg.joblib"
    metrics_path = out_dir / "agnews_metrics.json"
    sample_path = out_dir / "sample_inputs.json"

    joblib.dump(artifact, model_path)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    sample_inputs = [
        {"text": "Apple announced a new AI feature for its latest iPhone."},
        {"text": "The team won the final game after scoring in overtime."},
        {"text": "Global markets fell after the central bank changed interest rates."},
        {"text": "World leaders met to discuss a new international agreement."},
    ]
    with sample_path.open("w", encoding="utf-8") as f:
        json.dump(sample_inputs, f, indent=2)

    print("\nDone.")
    print(f"Validation accuracy: {valid_acc:.4f}")
    print(f"Test accuracy:       {test_acc:.4f}")
    print(f"Saved model to:      {model_path.resolve()}")
    print(f"Saved metrics to:    {metrics_path.resolve()}")
    print(f"Saved samples to:    {sample_path.resolve()}")

    print("\nQuick local prediction test:")
    example = "Apple announced a new AI feature for its latest iPhone."
    pred = model.predict([example])[0]
    prob = float(model.predict_proba([example]).max())
    print({"text": example, "predicted_category": pred, "confidence": round(prob, 4)})


if __name__ == "__main__":
    main()
