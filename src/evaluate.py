"""Evaluation utilities for classification models."""

from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def model_performance_classification(
    model: Any,
    predictors: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    """
    Compute classification metrics for a fitted model.

    Parameters
    ----------
    model : sklearn estimator
        Fitted classifier or pipeline ending in one.
    predictors : pd.DataFrame
        Feature matrix.
    target : pd.Series
        Ground-truth labels.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with columns Accuracy, Recall, Precision, F1.
    """
    pred = model.predict(predictors)

    return pd.DataFrame(
        {
            "Accuracy": accuracy_score(target, pred),
            "Recall": recall_score(target, pred),
            "Precision": precision_score(target, pred),
            "F1": f1_score(target, pred),
        },
        index=[0],
    )


def train_and_evaluate(
    name: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Fit a model and compute its train and test classification metrics.

    Designed to be called once per candidate model and collected into a list
    or dict for side-by-side comparison.

    Parameters
    ----------
    name : str
        Human-readable label for the model.
    model : sklearn estimator
        Unfitted classifier or pipeline ending in one.
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training labels.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        Test labels.

    Returns
    -------
    dict
        Mapping with keys:
        - "name": the label passed in
        - "model": the fitted model
        - "train_perf": training-set metrics (pd.DataFrame)
        - "test_perf": test-set metrics (pd.DataFrame)
    """
    model.fit(X_train, y_train)

    return {
        "name": name,
        "model": model,
        "train_perf": model_performance_classification(model, X_train, y_train),
        "test_perf": model_performance_classification(model, X_test, y_test),
    }
