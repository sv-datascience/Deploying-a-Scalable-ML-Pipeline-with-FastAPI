import os

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from ml.data import apply_label, process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    save_model,
    train_model,
)

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "census.csv")


@pytest.fixture(scope="module")
def data():
    """Load the census data once for all tests."""
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="module")
def split_data(data):
    """Split the data the same way train_model.py does."""
    return train_test_split(data, test_size=0.20, random_state=42, stratify=data["salary"])


@pytest.fixture(scope="module")
def processed(split_data):
    """Process a small training sample and the test set, then train a quick model."""
    train, test = split_data
    train_small = train.sample(2000, random_state=42)
    X_train, y_train, encoder, lb = process_data(
        train_small, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=CAT_FEATURES, label="salary",
        training=False, encoder=encoder, lb=lb,
    )
    model = train_model(X_train, y_train)
    return X_train, y_train, X_test, y_test, model


def test_train_model(processed):
    """
    train_model should return a fitted model that uses the expected algorithm
    (GradientBoostingClassifier).
    """
    _, _, _, _, model = processed
    assert isinstance(model, GradientBoostingClassifier)
    assert hasattr(model, "estimators_")  # only exists after fitting


def test_inference(processed):
    """
    inference should return a numpy array with one binary (0/1) prediction per row.
    """
    _, _, X_test, _, model = processed
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == X_test.shape[0]
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics():
    """
    compute_model_metrics should return the expected precision, recall, and F1
    for a small known example.
    """
    y = np.array([1, 1, 0, 0, 1])
    preds = np.array([1, 0, 0, 1, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(2 / 3)
    assert fbeta == pytest.approx(2 / 3)


def test_data_split_size_and_type(data, split_data):
    """
    The train and test datasets should be DataFrames with the expected 80/20 sizes.
    """
    train, test = split_data
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert len(train) + len(test) == len(data)
    assert len(test) == pytest.approx(0.20 * len(data), abs=1)


def test_process_data_shapes(processed):
    """
    process_data should return feature arrays with matching row counts for X and y,
    and the same number of columns for train and test.
    """
    X_train, y_train, X_test, y_test, _ = processed
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]


def test_apply_labels():
    """
    apply_label should convert binary predictions into the salary strings.
    """
    assert apply_label([1]) == ">50K"
    assert apply_label([0]) == "<=50K"


def test_save_and_load_model(processed, tmp_path):
    """
    A saved model should load back and give identical predictions.
    """
    _, _, X_test, _, model = processed
    path = os.path.join(tmp_path, "model.pkl")
    save_model(model, path)
    loaded = load_model(path)
    np.testing.assert_array_equal(inference(model, X_test), inference(loaded, X_test))
