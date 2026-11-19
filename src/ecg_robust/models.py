"""Machine-learning models for ECG arrhythmia classification.

Models:
- ``DummyClassifier`` (most_frequent baseline, patient-independent).
- ``RandomForestClassifier`` (100 trees, class weighting).
- ``SupportVectorMachine`` (RBF kernel, C=1.0, class weighting).

Class weighting (``class_weight="balanced"``) is used where appropriate
for the class-imbalanced AAMI annotations.
"""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def build_models(random_state: int = 42) -> dict[str, object]:
    """Build the three classifiers (unfitted)."""
    return {
        "dummy": DummyClassifier(strategy="most_frequent", random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=random_state, class_weight="balanced"
        ),
        "svm_rbf": SVC(
            kernel="rbf", C=1.0, random_state=random_state, class_weight="balanced"
        ),
    }
