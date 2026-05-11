"""Generates a self-contained HTML dashboard from comparison data + predictions."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Model metadata ────────────────────────────────────────────────────────────

_CATEGORIES = {
    "NaiveLastValue":    "Baseline",
    "RollingMean":       "Baseline",
    "HistoricalMean":    "Baseline",
    "Ridge":             "Linear",
    "Lasso":             "Linear",
    "ElasticNet":        "Linear",
    "XGBoost":           "Tree ML",
    "RandomForest":      "Tree ML",
    "SVR":               "Tree ML",
    "LightGBM":          "Tree ML",
    "CatBoost":          "Tree ML",
    "ARIMA":             "Classical",
    "SARIMA":            "Classical",
    "ETS":               "Classical",
    "GARCH":             "Classical",
    "MonteCarlo":        "Stochastic",
    "OrnsteinUhlenbeck": "Stochastic",
    "HMM":               "Regime",
    "MarkovSwitching":   "Regime",
    "Ensemble":          "Ensemble",
}

_CATEGORY_COLORS = {
    "Baseline":   "#94a3b8",
    "Linear":     "#818cf8",
    "Tree ML":    "#34d399",
    "Classical":  "#f59e0b",
    "Stochastic": "#60a5fa",
    "Regime":     "#f472b6",
    "Ensemble":   "#c084fc",
}

_METRICS = {
    "rmse": {
        "label": "Avg Error",
        "lower_better": True,
        "format": ".4f",
        "plain": "How far off predictions are on average. Lower = more accurate.",
    },
    "mae": {
        "label": "Typical Error",
        "lower_better": True,
        "format": ".4f",
        "plain": "The typical prediction error, less sensitive to big mistakes than Avg Error.",
    },
    "r2": {
        "label": "Predictive Power",
        "lower_better": False,
        "format": ".3f",
        "plain": "How much of the return variation the model explains. 0% means no better than guessing the average; negative means worse. Near-zero is normal for daily stock returns.",
    },
    "directional_accuracy": {
        "label": "Direction Correct",
        "lower_better": False,
        "format": ".1%",
        "plain": "How often the model correctly predicted whether the stock went up or down. Above 50% beats a coin flip.",
    },
    "sharpe": {
        "label": "Trading Signal Quality",
        "lower_better": False,
        "format": ".3f",
        "plain": "If you bought when the model predicted up and sold when it predicted down, this is your risk-adjusted return. Above 1.0 is considered good.",
    },
    "oos_r2": {
        "label": "OOS R² vs Mean",
        "lower_better": False,
        "format": ".4f",
        "plain": "Campbell-Thompson (2008) out-of-sample R²: improvement in MSE over the historical mean baseline. Positive = beats the mean; negative = worse than just predicting the average return.",
    },
    "rank_ic": {
        "label": "Rank IC",
        "lower_better": False,
        "format": ".4f",
        "plain": "Spearman rank correlation between predicted and actual returns. Measures how well the model ranks returns for investment purposes. Higher is better.",
    },
    "max_drawdown": {
        "label": "Max Drawdown",
        "lower_better": True,
        "format": ".2%",
        "plain": "The worst peak-to-trough loss of the long/short strategy. Closer to 0% is better.",
    },
    "calmar_ratio": {
        "label": "Calmar Ratio",
        "lower_better": False,
        "format": ".3f",
        "plain": "Annualised strategy return divided by maximum drawdown. Higher means more return per unit of worst-case loss.",
    },
}


# ── ROI simulation ────────────────────────────────────────────────────────────

def compute_roi(predictions_path: Path, initial: float = 10_000.0) -> dict | None:
    try:
        df = pd.read_csv(predictions_path, parse_dates=["date"])
    except Exception:
        return None

    y_true = df["y_true"].values.astype(float)
    y_pred = df["y_pred"].values.astype(float)

    if len(y_true) == 0:
        return None

    # Replace NaN/inf to avoid explosion
    y_true = np.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

    n_days = len(y_true)
    n_years = n_days / 252

    def _stats(daily_returns: np.ndarray) -> dict:
        cum = float(np.prod(1 + daily_returns))
        final = initial * cum
        ann = (max(cum, 1e-9) ** (1 / max(n_years, 1e-9))) - 1
        return {
            "final": round(final, 2),
            "profit": round(final - initial, 2),
            "total_pct": round((cum - 1) * 100, 2),
            "annual_pct": round(ann * 100, 2),
        }

    # Long/short: go long when pred > 0, short when pred ≤ 0
    ls_signal = np.where(y_pred > 0, 1.0, -1.0)
    ls_returns = np.clip(ls_signal * y_true, -0.5, 0.5)

    # Long-only: hold when pred > 0, cash otherwise
    lo_signal = (y_pred > 0).astype(float)
    lo_returns = lo_signal * y_true

    # Buy and hold
    bah_returns = y_true

    start_date = str(df["date"].iloc[0].date()) if pd.api.types.is_datetime64_any_dtype(df["date"]) else str(df["date"].iloc[0])
    end_date   = str(df["date"].iloc[-1].date()) if pd.api.types.is_datetime64_any_dtype(df["date"]) else str(df["date"].iloc[-1])

    return {
        "long_short":    _stats(ls_returns),
        "long_only":     _stats(lo_returns),
        "buy_and_hold":  _stats(bah_returns),
        "n_days":        n_days,
        "start_date":    start_date,
        "end_date":      end_date,
        "initial":       initial,
    }


# ── Colour helpers ────────────────────────────────────────────────────────────

def _rank_color(rank: int, total: int) -> str:
    if total <= 1:
        return "hsl(120,40%,92%)"
    hue = 120 * (1 - rank / (total - 1))
    return f"hsl({hue:.0f},55%,88%)"


def _color_cells(df: pd.DataFrame) -> pd.DataFrame:
    colors = pd.DataFrame("", index=df.index, columns=df.columns)
    for col, cfg in _METRICS.items():
        if col not in df.columns:
            continue
        series = df[col].copy()
        valid = series.notna()
        ranked = series[valid].rank(ascending=cfg["lower_better"], method="min") - 1
        total = valid.sum()
        for idx in df.index:
            if not valid[idx]:
                colors.loc[idx, col] = "background-color:#f8fafc;"
            else:
                colors.loc[idx, col] = f"background-color:{_rank_color(int(ranked[idx]), int(total))};"
    return colors


def _chart_data(df: pd.DataFrame, metric: str, lower_better: bool) -> dict:
    col = df[metric].dropna()
    col = col.sort_values(ascending=lower_better)
    return {
        "labels": col.index.tolist(),
        "values": [round(v, 6) for v in col.values],
        "colors": [_rank_color(i, len(col)) for i in range(len(col))],
    }


# ── Top performer summaries ───────────────────────────────────────────────────

def _top_performers(df: pd.DataFrame, roi_data: dict) -> list[dict]:
    findings = []

    if "directional_accuracy" in df.columns:
        best = df["directional_accuracy"].idxmax()
        val = df.loc[best, "directional_accuracy"]
        if pd.notna(val):
            findings.append({
                "icon": "🎯", "title": "Best Direction Predictor",
                "model": best, "value": f"{val:.1%}",
                "note": "of days it called up/down correctly",
            })

    if roi_data:
        best_model = max(
            roi_data,
            key=lambda m: roi_data[m].get("long_short", {}).get("final", 0),
        )
        roi = roi_data[best_model]["long_short"]
        profit = roi["profit"]
        sign = "+" if profit >= 0 else ""
        findings.append({
            "icon": "💰", "title": "Best Money Return",
            "model": best_model,
            "value": f"${roi['final']:,.0f}",
            "note": f"({sign}${profit:,.0f}) from $10,000 · long/short strategy",
        })

    if "sharpe" in df.columns:
        best = df["sharpe"].idxmax()
        val = df.loc[best, "sharpe"]
        if pd.notna(val):
            findings.append({
                "icon": "📈", "title": "Best Trading Signal Quality",
                "model": best, "value": f"{val:.3f}",
                "note": "Sharpe ratio (risk-adjusted return)",
            })

    return findings


# ── ROI table builder ─────────────────────────────────────────────────────────

def _roi_table_html(roi_data: dict, bah_final: float) -> str:
    if not roi_data:
        return "<p>No predictions data available.</p>"

    sorted_models = sorted(
        roi_data.keys(),
        key=lambda m: roi_data[m]["long_short"]["final"],
        reverse=True,
    )

    rows = []
    for model in sorted_models:
        r = roi_data[model]
        ls = r["long_short"]
        lo = r["long_only"]
        initial = r["initial"]

        # Traffic light vs buy-and-hold (long/short strategy)
        if ls["final"] > bah_final:
            light = '<span class="light green" title="Beat buy-and-hold">●</span>'
        elif ls["final"] > initial:
            light = '<span class="light yellow" title="Made profit but below buy-and-hold">●</span>'
        else:
            light = '<span class="light red" title="Lost money">●</span>'

        def fmt_val(v):
            sign = "+" if v >= 0 else ""
            color = "#16a34a" if v >= 0 else "#dc2626"
            return f'<span style="color:{color};font-weight:600">{sign}${v:,.0f}</span>'

        cat = _CATEGORIES.get(model, "Other")
        cat_color = _CATEGORY_COLORS.get(cat, "#94a3b8")

        rows.append(f"""<tr>
          <td class="model-name">
            <span class="cat-badge" style="background:{cat_color}">{cat}</span>
            <strong>{model}</strong>
          </td>
          <td style="text-align:right">${ls['final']:,.0f}</td>
          <td style="text-align:right">{fmt_val(ls['profit'])}</td>
          <td style="text-align:right">{ls['total_pct']:+.1f}%</td>
          <td style="text-align:right">{ls['annual_pct']:+.1f}%</td>
          <td style="text-align:right">${lo['final']:,.0f}</td>
          <td style="text-align:center">{light}</td>
        </tr>""")

    return f"""
    <table>
      <thead><tr>
        <th>Model</th>
        <th style="text-align:right">Final Value</th>
        <th style="text-align:right">Profit / Loss</th>
        <th style="text-align:right">Total Return</th>
        <th style="text-align:right">Annual Return</th>
        <th style="text-align:right">Long-Only Value</th>
        <th style="text-align:center">vs Buy &amp; Hold</th>
      </tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_html_report(
    df: pd.DataFrame,
    ticker: str,
    output_path: Path,
    roi_data: dict | None = None,
    initial_investment: float = 10_000.0,
) -> Path:
    colors = _color_cells(df)
    categories = [_CATEGORIES.get(m, "Other") for m in df.index]
    metric_cols = [c for c in _METRICS if c in df.columns]

    # ── ROI section ──────────────────────────────────────────────────────────
    roi_data = roi_data or {}

    # Get buy-and-hold from any model (all share the same y_true)
    bah_final = initial_investment
    bah_total_pct = 0.0
    bah_annual_pct = 0.0
    test_period = ""
    if roi_data:
        sample = next(iter(roi_data.values()))
        bah = sample["buy_and_hold"]
        bah_final = bah["final"]
        bah_total_pct = bah["total_pct"]
        bah_annual_pct = bah["annual_pct"]
        test_period = f"{sample['start_date']} → {sample['end_date']} ({sample['n_days']} trading days)"

    roi_table_html = _roi_table_html(roi_data, bah_final)

    # ROI bar chart data (long/short final values, sorted descending)
    roi_chart_data = {}
    if roi_data:
        sorted_roi = sorted(roi_data.items(),
                            key=lambda x: x[1]["long_short"]["final"],
                            reverse=True)
        roi_chart_data = {
            "labels": [m for m, _ in sorted_roi],
            "values": [r["long_short"]["final"] for _, r in sorted_roi],
            "colors": [
                "#16a34a" if r["long_short"]["final"] > bah_final
                else "#f59e0b" if r["long_short"]["final"] > initial_investment
                else "#dc2626"
                for _, r in sorted_roi
            ],
            "bah": bah_final,
            "initial": initial_investment,
        }

    # ── Table rows ───────────────────────────────────────────────────────────
    table_rows = []
    for i, model in enumerate(df.index):
        cat = categories[i]
        cat_color = _CATEGORY_COLORS.get(cat, "#94a3b8")
        cells = [
            f'<td class="model-name">'
            f'<span class="cat-badge" style="background:{cat_color}">{cat}</span>'
            f'<strong>{model}</strong></td>'
        ]
        for col in metric_cols:
            val = df.loc[model, col]
            bg = colors.loc[model, col]
            if pd.isna(val):
                cells.append('<td style="background:#f8fafc;color:#94a3b8;">—</td>')
            else:
                formatted = format(val, _METRICS[col]["format"])
                cells.append(f'<td style="{bg}">{formatted}</td>')
        table_rows.append(f"<tr>{''.join(cells)}</tr>")

    table_headers = '<th>Model</th>' + "".join(
        f'<th title="{_METRICS[c]["plain"]}">{_METRICS[c]["label"]} '
        f'<span class="sort-icon">{"↓" if _METRICS[c]["lower_better"] else "↑"}</span></th>'
        for c in metric_cols
    )

    # ── Chart data ───────────────────────────────────────────────────────────
    dir_data   = _chart_data(df, "directional_accuracy", lower_better=False) if "directional_accuracy" in df.columns else None
    sharpe_data = _chart_data(df, "sharpe", lower_better=False) if "sharpe" in df.columns else None
    rmse_data  = _chart_data(df, "rmse", lower_better=True) if "rmse" in df.columns else None

    top = _top_performers(df, roi_data)
    top_cards_html = "".join(f"""
        <div class="top-card">
          <div class="top-icon">{t['icon']}</div>
          <div class="top-body">
            <div class="top-title">{t['title']}</div>
            <div class="top-model">{t['model']}</div>
            <div class="top-value">{t['value']} <span class="top-note">{t['note']}</span></div>
          </div>
        </div>""" for t in top)

    glossary_html = "".join(f"""
        <div class="glossary-item">
          <div class="glossary-label">{cfg['label']}
            <span class="direction">({"lower is better" if cfg["lower_better"] else "higher is better"})</span>
          </div>
          <div class="glossary-desc">{cfg['plain']}</div>
        </div>""" for col, cfg in _METRICS.items() if col in metric_cols)

    legend_html = "".join(
        f'<span class="legend-item"><span class="legend-dot" style="background:{color}"></span>{cat}</span>'
        for cat, color in _CATEGORY_COLORS.items() if cat in categories
    )

    now = datetime.now().strftime("%B %d, %Y at %H:%M")
    n_models = len(df)
    bah_sign = "+" if bah_total_pct >= 0 else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stock Prediction Dashboard — {ticker}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f0f4f8;color:#1e293b;line-height:1.5}}
  .page{{max-width:1280px;margin:0 auto;padding:32px 24px}}

  .header{{background:linear-gradient(135deg,#1e3a5f 0%,#0f766e 100%);color:white;border-radius:16px;padding:36px 40px;margin-bottom:28px}}
  .header h1{{font-size:2rem;font-weight:700;margin-bottom:8px}}
  .header .sub{{opacity:.8;font-size:1rem}}
  .ticker-badge{{display:inline-block;background:rgba(255,255,255,.2);padding:4px 14px;border-radius:20px;font-weight:700;font-size:1.1rem;margin-right:12px}}

  .section{{background:white;border-radius:12px;padding:28px 32px;margin-bottom:24px;box-shadow:0 1px 3px rgba(0,0,0,.07)}}
  .section h2{{font-size:1.2rem;font-weight:600;margin-bottom:18px;color:#0f172a;border-bottom:2px solid #f1f5f9;padding-bottom:12px}}

  .top-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}}
  .top-card{{background:#f8fafc;border-radius:10px;padding:20px;display:flex;align-items:flex-start;gap:16px;border:1px solid #e2e8f0}}
  .top-icon{{font-size:2rem;line-height:1}}
  .top-title{{font-size:.75rem;text-transform:uppercase;letter-spacing:.08em;color:#64748b;font-weight:600;margin-bottom:4px}}
  .top-model{{font-size:1.2rem;font-weight:700;color:#0f172a;margin-bottom:4px}}
  .top-value{{font-size:1.1rem;font-weight:600;color:#0f766e}}
  .top-note{{font-size:.8rem;color:#64748b;font-weight:400}}

  .explainer-box{{background:#eff6ff;border-left:4px solid #3b82f6;border-radius:0 8px 8px 0;padding:16px 20px;margin-bottom:16px}}
  .explainer-box p{{color:#1e40af;font-size:.95rem}}

  .bah-banner{{background:#fef9c3;border:1px solid #fde047;border-radius:8px;padding:16px 20px;margin-bottom:20px;font-size:.9rem;color:#713f12}}
  .bah-banner strong{{font-size:1rem}}

  .roi-strategy-note{{font-size:.82rem;color:#64748b;margin-bottom:16px;line-height:1.6}}
  .roi-strategy-note strong{{color:#374151}}

  .table-wrap{{overflow-x:auto}}
  table{{width:100%;border-collapse:collapse;font-size:.88rem}}
  th{{background:#1e293b;color:#e2e8f0;padding:11px 14px;text-align:left;font-weight:600;white-space:nowrap}}
  .sort-icon{{opacity:.6;font-size:.75rem}}
  td{{padding:10px 14px;border-bottom:1px solid #f1f5f9;white-space:nowrap}}
  tr:hover td{{filter:brightness(.97)}}
  td.model-name{{min-width:190px}}
  .cat-badge{{display:inline-block;font-size:.68rem;padding:2px 8px;border-radius:10px;color:white;font-weight:600;margin-right:8px;vertical-align:middle;opacity:.9}}

  .light{{font-size:1.1rem}}
  .light.green{{color:#16a34a}}
  .light.yellow{{color:#d97706}}
  .light.red{{color:#dc2626}}

  .charts-grid{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}
  @media(max-width:768px){{.charts-grid{{grid-template-columns:1fr}}}}
  .chart-card{{background:#f8fafc;border-radius:10px;padding:20px;border:1px solid #e2e8f0}}
  .chart-title{{font-size:.88rem;font-weight:600;color:#374151;margin-bottom:6px}}
  .chart-sub{{font-size:.75rem;color:#6b7280;margin-bottom:14px}}

  .glossary-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}}
  .glossary-item{{background:#f8fafc;border-radius:8px;padding:16px;border:1px solid #e2e8f0}}
  .glossary-label{{font-weight:600;color:#0f172a;margin-bottom:6px}}
  .glossary-desc{{font-size:.85rem;color:#475569;line-height:1.6}}
  .direction{{font-weight:400;color:#64748b;font-size:.8rem}}

  .legend{{display:flex;flex-wrap:wrap;gap:12px;margin-top:16px}}
  .legend-item{{display:flex;align-items:center;gap:6px;font-size:.8rem;color:#475569}}
  .legend-dot{{width:10px;height:10px;border-radius:50%;display:inline-block}}
  .color-note{{font-size:.78rem;color:#64748b;margin-top:14px;padding:8px 12px;background:#f8fafc;border-radius:6px;border:1px solid #e2e8f0;display:inline-block}}
  .disclaimer{{font-size:.75rem;color:#94a3b8;margin-top:16px;line-height:1.6}}
  .footer{{text-align:center;color:#94a3b8;font-size:.8rem;margin-top:32px;padding:16px}}
</style>
</head>
<body><div class="page">

<div class="header">
  <h1><span class="ticker-badge">{ticker}</span> Stock Prediction Dashboard</h1>
  <div class="sub">{n_models} models compared &nbsp;·&nbsp; Generated {now}</div>
</div>

<div class="section">
  <div class="explainer-box">
    <p><strong>What is this?</strong> Each row below is a different algorithm that tried to predict whether {ticker}'s stock would go up or down each day, trained on 20 years of historical data (2004–2024) and tested on the most recent 20% of that period. The goal is to find which model produces the most useful trading signals.</p>
  </div>
  <div class="legend">{legend_html}</div>
</div>

<div class="section">
  <h2>Top Performers at a Glance</h2>
  <div class="top-grid">{top_cards_html}</div>
</div>

<div class="section">
  <h2>💰 If You Had Invested ${initial_investment:,.0f}…</h2>
  <div class="bah-banner">
    <strong>Buy &amp; Hold Benchmark</strong> — simply buying {ticker} at the start of the test period and holding:<br>
    Final value: <strong>${bah_final:,.0f}</strong> &nbsp;|&nbsp;
    Total return: <strong>{bah_sign}{bah_total_pct:.1f}%</strong> &nbsp;|&nbsp;
    Annualised: <strong>{bah_sign}{bah_annual_pct:.1f}%/yr</strong>
    {"&nbsp;·&nbsp; Test period: " + test_period if test_period else ""}
  </div>
  <div class="roi-strategy-note">
    <strong>Long/Short strategy:</strong> each day, buy ${ticker} if the model predicts the price will rise; sell short if it predicts a fall.
    The <strong>Long-Only</strong> column is more conservative: only buy when the model is bullish, otherwise hold cash.
    All figures assume no transaction costs and are for illustration only.
  </div>
  <div class="table-wrap">{roi_table_html}</div>
  <div class="disclaimer">
    ⚠️ Past performance does not guarantee future results. This is a simplified backtest — real trading involves transaction costs,
    slippage, and execution risk. These figures are for educational comparison purposes only.
    🟢 Beat buy-and-hold &nbsp; 🟡 Made profit but below buy-and-hold &nbsp; 🔴 Lost money
  </div>
</div>

{"<div class='section'><h2>Portfolio Value by Model (Long/Short Strategy)</h2>" +
 "<div class='chart-card'><div class='chart-title'>Final portfolio value starting from $" + f"{initial_investment:,.0f}" + "</div>" +
 "<div class='chart-sub'>Green = beat buy-and-hold · Yellow = profitable but below B&H · Red = lost money. Dashed line = buy-and-hold benchmark.</div>" +
 "<canvas id='roiChart' height='340'></canvas></div></div>" if roi_chart_data else ""}

<div class="section">
  <h2>Statistical Comparison</h2>
  <p style="font-size:.85rem;color:#64748b;margin-bottom:16px">Sorted by average error (lowest first). Hover column headers for explanations.</p>
  <div class="color-note">
    Cell colour: <span style="background:hsl(120,55%,88%);padding:1px 8px;border-radius:3px">green</span> = best in column &nbsp;
    <span style="background:hsl(60,55%,88%);padding:1px 8px;border-radius:3px">yellow</span> = middle &nbsp;
    <span style="background:hsl(0,55%,88%);padding:1px 8px;border-radius:3px">red</span> = worst
  </div><br>
  <div class="table-wrap">
    <table>
      <thead><tr>{table_headers}</tr></thead>
      <tbody>{''.join(table_rows)}</tbody>
    </table>
  </div>
</div>

<div class="section">
  <h2>Visual Comparison</h2>
  <div class="charts-grid">
    <div class="chart-card">
      <div class="chart-title">Direction Accuracy — Did it predict up/down correctly?</div>
      <div class="chart-sub">50% = coin flip. Any model above 50% has a useful directional signal.</div>
      <canvas id="dirChart" height="340"></canvas>
    </div>
    <div class="chart-card">
      <div class="chart-title">Trading Signal Quality (Sharpe Ratio)</div>
      <div class="chart-sub">Risk-adjusted return of following this model's buy/sell signals. Higher is better; above 1.0 is considered strong.</div>
      <canvas id="sharpeChart" height="340"></canvas>
    </div>
    <div class="chart-card">
      <div class="chart-title">Average Prediction Error</div>
      <div class="chart-sub">How far off were the return predictions on average. Lower means more accurate.</div>
      <canvas id="rmseChart" height="340"></canvas>
    </div>
  </div>
</div>

<div class="section">
  <h2>Metric Glossary</h2>
  <div class="glossary-grid">{glossary_html}</div>
</div>

<div class="footer">Generated by Stock Prediction Framework · {now}</div>
</div>

<script>
const roiData  = {json.dumps(roi_chart_data)};
const dirData  = {json.dumps(dir_data)};
const shData   = {json.dumps(sharpe_data)};
const rmseData = {json.dumps(rmse_data)};

function hBar(id, data, fmtFn) {{
  if (!data || !document.getElementById(id)) return;
  new Chart(document.getElementById(id), {{
    type: 'bar',
    data: {{
      labels: data.labels,
      datasets: [{{ data: data.values, backgroundColor: data.colors, borderRadius: 4, borderSkipped: false }}]
    }},
    options: {{
      indexAxis: 'y', responsive: true,
      plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{ label: ctx => ' ' + fmtFn(ctx.raw) }} }} }},
      scales: {{
        x: {{ grid: {{ color: '#f1f5f9' }}, ticks: {{ font: {{ size: 11 }} }} }},
        y: {{ ticks: {{ font: {{ size: 11 }} }}, grid: {{ display: false }} }}
      }}
    }}
  }});
}}

// ROI chart with B&H reference line
if (roiData && roiData.labels && document.getElementById('roiChart')) {{
  new Chart(document.getElementById('roiChart'), {{
    type: 'bar',
    data: {{
      labels: roiData.labels,
      datasets: [{{
        label: 'Final Value ($)',
        data: roiData.values,
        backgroundColor: roiData.colors,
        borderRadius: 4, borderSkipped: false,
      }}]
    }},
    options: {{
      indexAxis: 'y', responsive: true,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{ callbacks: {{ label: ctx => ' $' + ctx.raw.toLocaleString('en-US', {{maximumFractionDigits: 0}}) }} }},
        annotation: {{}}
      }},
      scales: {{
        x: {{
          grid: {{ color: '#f1f5f9' }},
          ticks: {{ font: {{ size: 11 }}, callback: v => '$' + v.toLocaleString() }}
        }},
        y: {{ ticks: {{ font: {{ size: 11 }} }}, grid: {{ display: false }} }}
      }}
    }},
    plugins: [{{
      id: 'bahLine',
      afterDraw(chart) {{
        const ctx2 = chart.ctx;
        const xScale = chart.scales.x;
        const x = xScale.getPixelForValue(roiData.bah);
        const x0 = xScale.getPixelForValue(roiData.initial);
        ctx2.save();
        // Buy and hold line
        ctx2.beginPath();
        ctx2.setLineDash([6, 4]);
        ctx2.strokeStyle = '#0f766e';
        ctx2.lineWidth = 2;
        ctx2.moveTo(x, chart.chartArea.top);
        ctx2.lineTo(x, chart.chartArea.bottom);
        ctx2.stroke();
        // Label
        ctx2.fillStyle = '#0f766e';
        ctx2.font = 'bold 11px sans-serif';
        ctx2.fillText('B&H', x + 4, chart.chartArea.top + 14);
        // Initial line
        ctx2.beginPath();
        ctx2.setLineDash([4, 4]);
        ctx2.strokeStyle = '#94a3b8';
        ctx2.lineWidth = 1;
        ctx2.moveTo(x0, chart.chartArea.top);
        ctx2.lineTo(x0, chart.chartArea.bottom);
        ctx2.stroke();
        ctx2.fillStyle = '#94a3b8';
        ctx2.font = '10px sans-serif';
        ctx2.fillText('$10k', x0 + 2, chart.chartArea.top + 14);
        ctx2.restore();
      }}
    }}]
  }});
}}

hBar('dirChart',  dirData,  v => (v*100).toFixed(1)+'%');
hBar('sharpeChart', shData, v => v.toFixed(3));
hBar('rmseChart', rmseData, v => v.toFixed(5));
</script>
</body></html>"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
