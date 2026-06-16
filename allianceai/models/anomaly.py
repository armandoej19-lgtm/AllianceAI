"""
Anomaly detection for financial time-series.

IsolationForest (scikit-learn) is a tree-based, unsupervised algorithm that
isolates anomalies without assuming any particular data distribution — ideal
for financial data that is neither Gaussian nor stationary.

Detected anomalies are flagged with a contamination score (−1 = anomaly,
+1 = normal) and the specific quarter/period is logged for human review.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from allianceai.core.logging_config import get_logger

logger = get_logger(__name__)


def detect_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.1,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Run IsolationForest on *df* (or a subset of its columns).

    Returns the input DataFrame augmented with:
        anomaly_score  — raw IsolationForest score (lower = more anomalous)
        is_anomaly     — boolean flag

    *contamination* is the expected fraction of anomalies in the data.
    For most mature companies 5–10% is reasonable; set higher for startups.
    """
    cols = columns or df.select_dtypes(include=np.number).columns.tolist()
    subset = df[cols].copy()

    # Drop rows where all selected columns are NaN; impute remaining NaN with column median.
    subset.dropna(how="all", inplace=True)
    subset.fillna(subset.median(), inplace=True)

    if len(subset) < 4:
        logger.warning("Not enough rows (%d) for anomaly detection — need ≥4.", len(subset))
        df["anomaly_score"] = np.nan
        df["is_anomaly"] = False
        return df

    scaler = StandardScaler()
    X = scaler.fit_transform(subset.values)

    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )
    preds  = iso.fit_predict(X)       # −1 = anomaly, +1 = normal
    scores = iso.score_samples(X)     # lower = more anomalous

    result = df.copy()
    result.loc[subset.index, "anomaly_score"] = scores
    result.loc[subset.index, "is_anomaly"]    = preds == -1

    anomaly_dates = subset.index[preds == -1].tolist()
    if anomaly_dates:
        logger.info(
            "IsolationForest flagged %d anomalous period(s): %s",
            len(anomaly_dates),
            [str(d)[:10] for d in anomaly_dates],
        )
    else:
        logger.info("No anomalies detected by IsolationForest.")

    return result
