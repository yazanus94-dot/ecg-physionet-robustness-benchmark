"""Spec #4: fold-specific imputation inside the sklearn pipeline."""

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ecg_robust.models import build_models


def test_pipeline_imputer_fit_on_train_only():
    X_tr = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0]])
    y_tr = np.array(["N", "N", "S", "S"])
    X_va = np.array([[np.nan, 7.0]])
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", DummyClassifier()),
    ])
    pipe.fit(X_tr, y_tr)
    # Imputer statistics come from the training fold only (median of [1,2,3,4]).
    assert pipe.named_steps["imputer"].statistics_[0] == pytest.approx(2.5)
    # The imputed value of the validation row is the train median, not a
    # validation-derived statistic.
    assert np.isclose(pipe.named_steps["imputer"].transform(X_va)[0, 0], 2.5)
    assert len(pipe.predict(X_va)) == 1


def test_all_model_pipelines_include_imputer():
    for name, model in build_models().items():
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ])
        assert "imputer" in pipe.named_steps
        assert "scaler" in pipe.named_steps
        assert "model" in pipe.named_steps
