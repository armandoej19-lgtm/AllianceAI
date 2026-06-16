"""
DuckDB-backed persistence layer.

DuckDB is an in-process analytical database — no server required, queries are
SQL, and it handles DataFrames natively.  All fetched financial data is cached
here so we don't hammer rate-limited APIs on repeated runs.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from allianceai.core.config import settings
from allianceai.core.logging_config import get_logger

logger = get_logger(__name__)


class Storage:
    """Thin wrapper around a DuckDB connection with caching helpers."""

    def __init__(self, db_path: str | None = None) -> None:
        path = db_path or settings.db_path
        logger.info("Opening DuckDB at '%s'", path)
        self._conn = duckdb.connect(path)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS fetched_frames (
                cache_key   VARCHAR PRIMARY KEY,
                fetched_at  TIMESTAMP NOT NULL,
                payload     JSON NOT NULL
            )
        """)
        # Forecasts persisted as "seeds": one row per (metric, future period).
        # Re-running the pipeline upserts — we keep the latest prediction made
        # for each target period until it is evaluated against actuals.
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                ticker      VARCHAR NOT NULL,
                statement   VARCHAR NOT NULL,
                metric      VARCHAR NOT NULL,
                method      VARCHAR,
                made_at     TIMESTAMP NOT NULL,
                target_date DATE NOT NULL,
                yhat        DOUBLE,
                yhat_lower  DOUBLE,
                yhat_upper  DOUBLE,
                PRIMARY KEY (ticker, statement, metric, target_date)
            )
        """)
        # Once official data covers a target_date, the prediction is graded
        # and moved here. This history is what calibrates future forecasts.
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_outcomes (
                ticker       VARCHAR NOT NULL,
                statement    VARCHAR NOT NULL,
                metric       VARCHAR NOT NULL,
                method       VARCHAR,
                made_at      TIMESTAMP,
                target_date  DATE NOT NULL,
                yhat         DOUBLE,
                actual       DOUBLE,
                pct_error    DOUBLE,
                evaluated_at TIMESTAMP NOT NULL,
                PRIMARY KEY (ticker, statement, metric, target_date)
            )
        """)
        logger.debug("DuckDB schema initialised.")

    # ------------------------------------------------------------------
    # Prediction seed store
    # ------------------------------------------------------------------

    def save_predictions(self, rows: list[dict]) -> int:
        """Upsert forecast rows. Returns the number of rows written."""
        for r in rows:
            self._conn.execute(
                """
                INSERT INTO predictions
                    (ticker, statement, metric, method, made_at, target_date,
                     yhat, yhat_lower, yhat_upper)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (ticker, statement, metric, target_date)
                DO UPDATE SET method = excluded.method,
                              made_at = excluded.made_at,
                              yhat = excluded.yhat,
                              yhat_lower = excluded.yhat_lower,
                              yhat_upper = excluded.yhat_upper
                """,
                [r["ticker"], r["statement"], r["metric"], r.get("method"),
                 r["made_at"], r["target_date"],
                 r.get("yhat"), r.get("yhat_lower"), r.get("yhat_upper")],
            )
        logger.info("Stored %d prediction seed(s).", len(rows))
        return len(rows)

    def pending_predictions(self, ticker: str) -> pd.DataFrame:
        """Predictions for *ticker* not yet graded against actuals."""
        return self._conn.execute(
            "SELECT * FROM predictions WHERE ticker = ? ORDER BY target_date",
            [ticker],
        ).fetchdf()

    def record_outcomes(self, rows: list[dict]) -> int:
        """Grade predictions: move them into prediction_outcomes."""
        for r in rows:
            self._conn.execute(
                """
                INSERT INTO prediction_outcomes
                    (ticker, statement, metric, method, made_at, target_date,
                     yhat, actual, pct_error, evaluated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (ticker, statement, metric, target_date)
                DO UPDATE SET actual = excluded.actual,
                              pct_error = excluded.pct_error,
                              evaluated_at = excluded.evaluated_at
                """,
                [r["ticker"], r["statement"], r["metric"], r.get("method"),
                 r.get("made_at"), r["target_date"], r.get("yhat"),
                 r.get("actual"), r.get("pct_error"), r["evaluated_at"]],
            )
            self._conn.execute(
                """DELETE FROM predictions
                   WHERE ticker = ? AND statement = ? AND metric = ? AND target_date = ?""",
                [r["ticker"], r["statement"], r["metric"], r["target_date"]],
            )
        if rows:
            logger.info("Graded %d prediction(s) against official data.", len(rows))
        return len(rows)

    def outcome_history(self, ticker: str, metric: str | None = None) -> pd.DataFrame:
        """Graded prediction history — the calibration training set."""
        q = "SELECT * FROM prediction_outcomes WHERE ticker = ?"
        params: list = [ticker]
        if metric:
            q += " AND metric = ?"
            params.append(metric)
        return self._conn.execute(q + " ORDER BY target_date", params).fetchdf()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def cache_dataframe(self, key: str, df: pd.DataFrame) -> None:
        """Serialise a DataFrame to JSON and upsert it into the cache."""
        payload = df.to_json(orient="split", date_format="iso")
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO fetched_frames (cache_key, fetched_at, payload)
            VALUES (?, ?, ?)
            ON CONFLICT (cache_key) DO UPDATE SET fetched_at = excluded.fetched_at,
                                                   payload    = excluded.payload
            """,
            [key, now, payload],
        )
        logger.debug("Cached DataFrame under key='%s'.", key)

    def load_dataframe(self, key: str) -> pd.DataFrame | None:
        """
        Return the cached DataFrame for *key* if it exists and is still fresh.
        Returns None when the entry is missing or stale.
        """
        row = self._conn.execute(
            "SELECT fetched_at, payload FROM fetched_frames WHERE cache_key = ?",
            [key],
        ).fetchone()

        if row is None:
            logger.debug("Cache miss for key='%s'.", key)
            return None

        raw_ts = row[0]
        if isinstance(raw_ts, str):
            raw_ts = datetime.fromisoformat(raw_ts)
        fetched_at = raw_ts.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600

        if age_hours > settings.cache_ttl_hours:
            logger.info("Stale cache entry for key='%s' (%.1f h old). Will re-fetch.", key, age_hours)
            return None

        logger.debug("Cache hit for key='%s' (%.1f h old).", key, age_hours)
        return pd.read_json(io.StringIO(row[1]), orient="split")

    def close(self) -> None:
        self._conn.close()
        logger.debug("DuckDB connection closed.")


# Module-level singleton — import this wherever persistence is needed.
storage = Storage()
