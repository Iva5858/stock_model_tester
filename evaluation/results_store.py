from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Optional

import pandas as pd


@dataclass
class ExperimentRecord:
    exp_id: str
    ticker: str
    model: str
    horizon: int
    target: str
    metrics: dict
    path: Path


class ResultsStore(ABC):
    @abstractmethod
    def list_experiments(
        self,
        ticker: str | None = None,
        model: str | None = None,
        horizon: int | None = None,
        target: str | None = None,
    ) -> list[ExperimentRecord]: ...

    @abstractmethod
    def load_predictions(self, exp_id: str) -> pd.DataFrame: ...

    @abstractmethod
    def load_metrics(self, exp_id: str) -> dict: ...


class FileResultsStore(ResultsStore):
    """Reads from results/ directory.  Supports both v1 and v2 path layouts.

    v2: results/<TICKER>/<MODEL>/h<H>/<TARGET>/<exp_id>/
    v1: results/<TICKER>/<MODEL>/<exp_id>/  — treated as h=1, next_return, schema_version=1
    """

    def __init__(self, results_root: str | Path = "results"):
        self._root = Path(results_root)

    # ── internal ──────────────────────────────────────────────────────────────

    def _iter_all(self) -> list[tuple[Path, str, str, int, str]]:
        """Yield (exp_dir, ticker, model, horizon, target) for every experiment."""
        results: list[tuple[Path, str, str, int, str]] = []
        if not self._root.exists():
            return results

        for ticker_dir in self._root.iterdir():
            if not ticker_dir.is_dir():
                continue
            ticker = ticker_dir.name
            for model_dir in ticker_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                model = model_dir.name
                for child in model_dir.iterdir():
                    if not child.is_dir():
                        continue
                    if child.name.startswith("h") and child.name[1:].isdigit():
                        # v2: model_dir/h<H>/<TARGET>/<exp_id>/
                        horizon = int(child.name[1:])
                        for target_dir in child.iterdir():
                            if not target_dir.is_dir():
                                continue
                            target = target_dir.name
                            for exp_dir in target_dir.iterdir():
                                if exp_dir.is_dir() and (exp_dir / "metrics.json").exists():
                                    results.append((exp_dir, ticker, model, horizon, target))
                    else:
                        # v1: model_dir/<exp_id>/
                        exp_dir = child
                        if exp_dir.is_dir() and (exp_dir / "metrics.json").exists():
                            results.append((exp_dir, ticker, model, 1, "next_return"))
        return results

    def _find_exp(self, exp_id: str) -> Optional[Path]:
        for exp_dir, *_ in self._iter_all():
            if exp_dir.name == exp_id:
                return exp_dir
        return None

    # ── public ────────────────────────────────────────────────────────────────

    def list_experiments(
        self,
        ticker: str | None = None,
        model: str | None = None,
        horizon: int | None = None,
        target: str | None = None,
    ) -> list[ExperimentRecord]:
        records = []
        for exp_dir, t, m, h, tgt in self._iter_all():
            if ticker is not None and t.upper() != ticker.upper():
                continue
            if model is not None and m != model:
                continue
            if horizon is not None and h != horizon:
                continue
            if target is not None and tgt != target:
                continue
            try:
                raw = json.loads((exp_dir / "metrics.json").read_text())
            except Exception:
                continue
            records.append(ExperimentRecord(
                exp_id=exp_dir.name,
                ticker=t,
                model=m,
                horizon=h,
                target=tgt,
                metrics=raw,
                path=exp_dir,
            ))
        return sorted(records, key=lambda r: r.exp_id)

    def load_predictions(self, exp_id: str) -> pd.DataFrame:
        path = self._find_exp(exp_id)
        if path is None:
            raise FileNotFoundError(f"No experiment found with exp_id='{exp_id}'")
        preds_file = path / "predictions.csv"
        if not preds_file.exists():
            raise FileNotFoundError(f"predictions.csv missing in {path}")
        return pd.read_csv(preds_file, parse_dates=["date"])

    def load_metrics(self, exp_id: str) -> dict:
        path = self._find_exp(exp_id)
        if path is None:
            raise FileNotFoundError(f"No experiment found with exp_id='{exp_id}'")
        return json.loads((path / "metrics.json").read_text())
