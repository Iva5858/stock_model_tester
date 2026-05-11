"""Test that all registries are populated and importable."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_loader_registry():
    from data.base_loader import _LOADER_REGISTRY
    import data  # noqa: F401 — triggers registrations
    assert "yfinance" in _LOADER_REGISTRY
    assert "csv" in _LOADER_REGISTRY
    assert "fred" in _LOADER_REGISTRY


def test_loader_registry_instantiable():
    from data import get_loader
    loader = get_loader("csv")
    assert hasattr(loader, "load")


def test_get_loader_unknown_raises():
    from data.base_loader import get_loader_class
    import pytest
    with pytest.raises(ValueError, match="Unknown loader"):
        get_loader_class("nonexistent")


def test_model_registry_has_classifiers():
    from models import list_models
    models = list_models()
    assert "LogisticRegressionClassifier" in models
    assert "XGBoostClassifier" in models
    assert "RandomForestClassifier" in models
    assert "LightGBMClassifier" in models


def test_model_registry_no_conflict():
    from models import list_models
    models = list_models()
    # Regression and classifier must be separate entries
    assert "XGBoost" in models
    assert "XGBoostClassifier" in models
    assert models["XGBoost"] is not models["XGBoostClassifier"]


def test_transform_registry():
    import features  # noqa: F401 — triggers registrations
    from features.base_transform import _TRANSFORM_REGISTRY
    for name in ("lag_returns", "rolling_stats", "volume_delta", "technical", "macro_fred"):
        assert name in _TRANSFORM_REGISTRY, f"'{name}' not in transform registry"


def test_transform_instantiable():
    from features.base_transform import get_transform
    cls = get_transform("lag_returns")
    t = cls()
    assert hasattr(t, "fit_transform")


def test_get_transform_unknown_raises():
    from features.base_transform import get_transform
    import pytest
    with pytest.raises(ValueError, match="Unknown transform"):
        get_transform("does_not_exist")


def test_allocator_registry():
    from strategy.allocators import _ALLOCATOR_REGISTRY
    for name in ("equal", "signal_weighted", "min_variance", "max_sharpe"):
        assert name in _ALLOCATOR_REGISTRY, f"'{name}' not in allocator registry"


def test_allocator_instantiable():
    from strategy.allocators import get_allocator
    cls = get_allocator("equal")
    alloc = cls()
    assert hasattr(alloc, "allocate")


def test_cost_model_registry():
    from strategy.cost_models import _COST_REGISTRY
    assert "fixed_bps" in _COST_REGISTRY


def test_cost_model_instantiable():
    from strategy.cost_models import get_cost_model
    cls = get_cost_model("fixed_bps")
    cm = cls(bps=5.0)
    assert hasattr(cm, "cost_per_step")


# ── DAG checks ────────────────────────────────────────────────────────────────

def _module_imports_from(module_name: str, forbidden_prefix: str) -> list[str]:
    """Return list of submodule names that module imports from forbidden prefix."""
    import importlib
    import sys
    mod = importlib.import_module(module_name)
    violations = []
    for attr in dir(mod):
        val = getattr(mod, attr, None)
        if hasattr(val, "__module__") and val.__module__:
            if val.__module__.startswith(forbidden_prefix):
                violations.append(f"{module_name}.{attr} from {val.__module__}")
    return violations


def test_data_does_not_import_features():
    """data/ must not import from features/."""
    import data.base_loader
    import data.csv_loader
    import data.yfinance_loader
    # These will raise ImportError if features are imported at module level
    for mod_name in ("data.base_loader", "data.csv_loader", "data.yfinance_loader"):
        import importlib
        mod = importlib.import_module(mod_name)
        src = Path(mod.__file__).read_text()
        assert "from features" not in src and "import features" not in src, \
            f"{mod_name} imports from features/"


def test_models_do_not_import_experiments():
    """models/ must not import from experiments/."""
    for mod_name in ("models.base_model", "models.ml_models", "models.baseline"):
        import importlib
        mod = importlib.import_module(mod_name)
        src = Path(mod.__file__).read_text()
        assert "from experiments" not in src and "import experiments" not in src, \
            f"{mod_name} imports from experiments/"


def test_evaluation_does_not_import_models():
    """evaluation/ must not import from models/."""
    for mod_name in ("evaluation.metrics", "evaluation.reporter", "evaluation.results_store"):
        import importlib
        mod = importlib.import_module(mod_name)
        src = Path(mod.__file__).read_text()
        assert "from models" not in src and "import models" not in src, \
            f"{mod_name} imports from models/"


def _has_project_import(src: str, package: str) -> bool:
    """Return True if src has a real import from project package (not stdlib lookalikes)."""
    import re
    # Match lines like 'from data import ...' or 'from data.foo import ...'
    # but NOT 'from dataclasses', 'from datetime', etc.
    # The package name must be followed by whitespace, dot, or end-of-token.
    pattern_from = re.compile(rf"^\s*from {re.escape(package)}(?:[\s.])", re.MULTILINE)
    pattern_import = re.compile(rf"^\s*import {re.escape(package)}(?:[\s.]|$)", re.MULTILINE)
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if pattern_from.match(line) or pattern_import.match(line):
            if "__future__" in stripped:
                continue
            return True
    return False


def test_strategy_does_not_import_data_directly():
    """strategy/ must not import from data/ directly."""
    strategy_dir = Path(__file__).parent.parent / "strategy"
    for py_file in strategy_dir.rglob("*.py"):
        src = py_file.read_text()
        assert not _has_project_import(src, "data"), \
            f"{py_file} imports from data/ directly"


def test_strategy_does_not_import_features_directly():
    """strategy/ must not import from features/ directly."""
    strategy_dir = Path(__file__).parent.parent / "strategy"
    for py_file in strategy_dir.rglob("*.py"):
        src = py_file.read_text()
        assert not _has_project_import(src, "features"), \
            f"{py_file} imports from features/ directly"
