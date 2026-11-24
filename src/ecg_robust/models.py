"""Machine-learning models for ECG arrhythmia classification.

Models (Phase 6 — stronger experimental comparison, no result manipulation):
- ``DummyClassifier`` (most_frequent baseline, patient-independent).
- ``RandomForestClassifier`` (n_estimators=100, class_weight="balanced").
- ``LogisticRegression`` (max_iter=2000, class_weight="balanced").
- ``HistGradientBoostingClassifier`` (default ``class_weight=None``; included
  as an unweighted comparator, not as an imbalance-handling model).
- ``SVC`` (RBF kernel, C=1.0, class_weight="balanced").

Class weighting (``class_weight="balanced"``) is used where applicable
for the class-imbalanced AAMI annotations. HistGB deliberately uses the
default ``class_weight=None`` so the zoo includes an unweighted comparator.
"""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def build_models(random_state: int = 42) -> dict[str, object]:
    """Build the classifier zoo (unfitted).

    Per the project spec, all primary evaluation is patient-independent
    (patient-independent CV), and no model is selected by testing data.
    """
    return {
        "dummy": DummyClassifier(strategy="most_frequent", random_state=random_state),
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=random_state, class_weight="balanced"
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=100, random_state=random_state
        ),
        "svm_rbf": SVC(
            kernel="rbf", C=1.0, random_state=random_state, class_weight="balanced"
        ),
    }
