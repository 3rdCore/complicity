"""
Unified plotting function for experiment summary plots.

Called from both ``main.ipynb`` (single experiment) and ``plot.ipynb``
(batch regeneration over many experiment folders).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from source.utils.eval_helper import (
    NATS_TO_BITS,
    add_asymptotic_model,
    build_interpolated_threshold_models,
    build_threshold_lines,
    compute_envelope_indices,
    find_envelope_intersections,
    find_type_transitions,
    get_mdl_predicted_quantities,
)
from source.utils.notebook_helpers import _compute_kp_with_convergence_cutoff

# ---------------------------------------------------------------------------
# Color map shared across both notebooks
# ---------------------------------------------------------------------------

COLOR_MAP = {
    "Overfit": "k",
    "Color-based": "#ad5752",  # red – spurious (color)
    "Digit-based": "#8ab169",  # green – robust (digit)
    "Bayes-Optimal predictor": "tab:green",
    "Only Watermark": "#748fbe",  # blue – Bayes optimal (watermark)
    "Watermark-only": "#748fbe",  # blue – Bayes optimal (watermark)
    "color": "#ad5752",  # red – spurious (color)
    "digit": "#8ab169",  # green – robust (digit)
    "watermark": "#748fbe",  # blue – Bayes optimal (watermark)
}
COLOR_TRAIN_ACC = "#3B3B3B"
COLOR_VAL_ACC = "#d5792e"


def apply_publication_style() -> None:
    """Set publication-ready Matplotlib style (call once per session)."""
    plt.style.use("seaborn-v0_8-colorblind")
    mpl.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "axes.linewidth": 1.2,
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auto_detect_features(
    model_types: np.ndarray,
) -> tuple[dict, dict, str, str]:
    """Return (FEATURE_A, FEATURE_B, SECONDARY_ACC_COL, SECONDARY_ACC_LABEL)."""
    has_color = any("color" in m.lower() for m in model_types)
    has_watermark = any("watermark" in m.lower() for m in model_types)

    if has_color and not has_watermark:
        fa = {"name": "Color", "model_key": "color", "threshold_type": "color_threshold"}
        fb = {"name": "Digit", "model_key": "digit", "threshold_type": "digit_threshold"}
        sec_col, sec_lbl = "grayscale_acc_mean", "Grayscale acc"
    elif has_watermark and not has_color:
        fa = {"name": "Watermark", "model_key": "watermark", "threshold_type": "watermark_threshold"}
        fb = {"name": "Digit", "model_key": "digit", "threshold_type": "digit_threshold"}
        sec_col, sec_lbl = "grayscale_acc_mean", "No-watermark acc"
    else:
        fa = {"name": "Feature A", "model_key": "color", "threshold_type": "color_threshold"}
        fb = {"name": "Feature B", "model_key": "digit", "threshold_type": "digit_threshold"}
        sec_col, sec_lbl = "grayscale_acc_mean", "Grayscale acc"

    return fa, fb, sec_col, sec_lbl


def _compute_k_p_results(
    df_nats: pd.DataFrame,
    skip_bayes: bool,
    kp_abs_delta_threshold: float = 5e-5,
) -> Dict[str, dict]:
    """Compute K(p) for every model type in *df_nats*.

    Returns ``k_p_results_env`` dict identical in layout to what ``main.ipynb``
    cell 18 produces.
    """
    k_p_results_env: Dict[str, dict] = {}

    for model_type, model_df in df_nats.groupby("model_type"):
        if skip_bayes and "bayes" in model_type.lower():
            continue
        model_df = model_df.sort_values("dataset_size")
        x_m = model_df["dataset_size"].values
        if len(x_m) < 2:
            continue

        asym_row = model_df.loc[model_df["dataset_size"].idxmax()]
        asym_orig_loss = float(asym_row["mean_original_log_loss"])
        asym_orig_acc = float(asym_row.get("mean_original_acc", np.nan))
        asym_test_acc = float(asym_row.get("mean_test_acc", np.nan))
        asym_test_loss = float(asym_row["mean_test_log_loss"])
        asym_gs_acc = float(asym_row.get("mean_grayscale_acc", np.nan))
        asym_gs_ll = float(asym_row.get("mean_grayscale_log_loss", np.nan))
        asym_wm_acc = (
            float(asym_row.get("mean_watermark_only_acc", np.nan))
            if "mean_watermark_only_acc" in model_df.columns
            else np.nan
        )
        asym_wm_ll = (
            float(asym_row.get("mean_watermark_only_log_loss", np.nan))
            if "mean_watermark_only_log_loss" in model_df.columns
            else np.nan
        )
        asym_co_acc = (
            float(asym_row.get("mean_color_only_acc", np.nan))
            if "mean_color_only_acc" in model_df.columns
            else np.nan
        )
        asym_do_acc = (
            float(asym_row.get("mean_digit_only_acc", np.nan))
            if "mean_digit_only_acc" in model_df.columns
            else np.nan
        )

        # Bayes models may have a pre-computed K(p) column
        if (
            model_type == "Bayes-Optimal predictor"
            and "k_p_nats" in model_df.columns
            and model_df["k_p_nats"].notna().any()
        ):
            k_p = float(model_df["k_p_nats"].iloc[0])
        else:
            y_m = model_df["mean_test_log_loss"].values
            k_p, *_ = _compute_kp_with_convergence_cutoff(
                x_m, y_m, abs_delta_threshold=kp_abs_delta_threshold
            )

        k_p_results_env[model_type] = {
            "Kp_nats": k_p,
            "asymptotic_original_loss_nats": asym_orig_loss,
            "asymptotic_original_acc": asym_orig_acc,
            "asymptotic_test_acc": asym_test_acc,
            "asymptotic_test_loss_nats": asym_test_loss,
            "asymptotic_grayscale_acc": asym_gs_acc,
            "asymptotic_grayscale_loss_nats": asym_gs_ll,
            "asymptotic_watermark_only_acc": asym_wm_acc,
            "asymptotic_watermark_only_loss_nats": asym_wm_ll,
            "asymptotic_color_only_acc": asym_co_acc,
            "asymptotic_digit_only_acc": asym_do_acc,
            "min_size": x_m[0],
            "max_size": x_m[-1],
        }

    return k_p_results_env


def _aggregate_exp2(
    results_df: pd.DataFrame,
    experiment_setting: int,
    permutation_attributes: List[str],
) -> pd.DataFrame:
    """Aggregate per-run Experiment 2 results into mean/std per dataset size."""
    agg_dict: dict = {
        "val_acc": ["mean", "std"],
        "train_acc": ["mean", "std"],
        "grayscale_acc": ["mean", "std"],
    }
    if experiment_setting == 1 and "majority_acc" in results_df.columns:
        agg_dict["majority_acc"] = ["mean", "std"]
    if experiment_setting == 1 and "color_only_acc" in results_df.columns:
        agg_dict["color_only_acc"] = ["mean", "std"]
    if experiment_setting == 1 and "digit_only_acc" in results_df.columns:
        agg_dict["digit_only_acc"] = ["mean", "std"]
    if experiment_setting == 2 and "watermark_only_acc" in results_df.columns:
        agg_dict["watermark_only_acc"] = ["mean", "std"]
    for attr in permutation_attributes:
        for prefix in ("pvalue_", "acc_drop_"):
            col = f"{prefix}{attr}"
            if col in results_df.columns:
                agg_dict[col] = ["mean", "std"]

    summary_df = results_df.groupby("dataset_size").agg(agg_dict).reset_index()
    summary_df.columns = ["_".join(col).strip() if col[1] else col[0] for col in summary_df.columns.values]
    summary_df = summary_df.rename(columns={"dataset_size_": "dataset_size"})
    return summary_df


# ---------------------------------------------------------------------------
# Main plotting entry point
# ---------------------------------------------------------------------------


def plot_experiment_summary(
    pcl_results_df: pd.DataFrame,
    results_df: pd.DataFrame,
    *,
    k_p_results: Optional[Dict[str, dict]] = None,
    experiment_setting: int = 1,
    spur_prob: float = 0.0,
    flip_prob: float = 0.0,
    env_noisiness: float = 0.0,
    watermark_bank_size: int = 2,
    uninformative_majority: bool = False,
    permutation_attributes: Optional[List[str]] = None,
    skip_bayes: bool = False,
    include_bayes_intermediates: bool = False,
    show_transition_lines: bool = True,
    show_non_envelope_lines: bool = False,
    nb_threshold_models: int = 0,
    nb_interpolated_models: int = 200,
    threshold_metric: str = "mean_original_acc",
    kp_abs_delta_threshold: float = 5e-5,
    x_plot_max: int = 249999,
    save_dir: Optional[Path] = None,
    experiment_metadata: Optional[dict] = None,
) -> plt.Figure:
    """
    Generate the 2×3 experiment summary figure.

    Parameters
    ----------
    pcl_results_df : DataFrame
        Experiment 1 PCL results (columns include ``model_type``,
        ``dataset_size``, ``mean_test_log_loss``, etc.)
    results_df : DataFrame
        Experiment 2 per-run accuracy results.
    k_p_results : dict, optional
        Pre-computed K(p) dict from the K(p) cell.  When *None* the values
        are re-derived from *pcl_results_df*.
    experiment_setting : int
        1 or 2.
    spur_prob, flip_prob, env_noisiness : float
        Dataset generation parameters shown in the suptitle.
    watermark_bank_size : int
        Watermark bank size (Setting 2).
    uninformative_majority : bool
        Setting 1 flag.
    permutation_attributes : list[str], optional
        Feature names used for permutation p-values and accuracy drops.
        Inferred from *results_df* columns when *None*.
    skip_bayes : bool
        Hide Bayes-Optimal curves entirely.
    include_bayes_intermediates : bool
        Include intermediate (threshold + interpolated) Bayes models on the
        compression envelope.
    show_transition_lines : bool
        Draw vertical lines at MDL-predicted feature transitions.
    show_non_envelope_lines : bool
        Show compression lines *not* on the lower envelope (faded).
    nb_threshold_models, nb_interpolated_models : int
        Number of threshold / interpolated models for envelope construction.
    threshold_metric : str
        Column used for threshold & interpolated model construction.
    kp_abs_delta_threshold : float
        Absolute delta threshold passed to ``_compute_kp_with_convergence_cutoff``
        for K(p) estimation convergence detection.  Lower values keep more of
        the tail; higher values cut earlier.  Default ``5e-5``.
    x_plot_max : int
        Right limit of the x axis (dataset size).
    save_dir : Path, optional
        When given the figure is saved to *save_dir* as both PNG and PDF.
    experiment_metadata : dict, optional
        Metadata dict to attach as columns to ``summary_df`` (only relevant
        when saving CSVs from ``main.ipynb``).

    Returns
    -------
    matplotlib.figure.Figure
    """
    BAYES_THRESHOLD_TYPE = "bayes_threshold"

    if permutation_attributes is None:
        permutation_attributes = [
            c.replace("pvalue_", "") for c in results_df.columns if c.startswith("pvalue_")
        ]

    # Auto-skip Bayes when there is no spurious feature
    if experiment_setting == 1 and spur_prob == 0:
        skip_bayes = True
    elif experiment_setting == 2 and env_noisiness == 0:
        skip_bayes = True

    # ── Feature detection ─────────────────────────────────────────────
    model_types_in_pcl = pcl_results_df["model_type"].unique()
    FEATURE_A, FEATURE_B, SECONDARY_ACC_COL, SECONDARY_ACC_LABEL = _auto_detect_features(model_types_in_pcl)

    # ── Aggregate Experiment 2 ────────────────────────────────────────
    summary_df = _aggregate_exp2(results_df, experiment_setting, permutation_attributes)
    if experiment_metadata is not None:
        for key, val in experiment_metadata.items():
            summary_df[key] = val

    # ── K(p) computation (reuse or recompute) ─────────────────────────
    df_nats = pcl_results_df.copy()
    for col in df_nats.columns:
        try:
            df_nats[col] = pd.to_numeric(df_nats[col])
        except (ValueError, TypeError):
            pass

    if k_p_results is not None:
        # Reuse pre-computed K(p) from cell 18
        k_p_results_env: Dict[str, dict] = {}
        for model_type, res in k_p_results.items():
            if skip_bayes and "bayes" in model_type.lower():
                continue
            k_p_results_env[model_type] = {
                "Kp_nats": res["Kp_nats"],
                "asymptotic_original_loss_nats": res["asymptotic_original_loss_nats"],
                "asymptotic_original_acc": res["asymptotic_original_acc"],
                "asymptotic_test_acc": res["asymptotic_test_acc"],
                "asymptotic_test_loss_nats": res["asymptotic_test_loss_nats"],
                "asymptotic_grayscale_acc": res["asymptotic_grayscale_acc"],
                "asymptotic_grayscale_loss_nats": res["asymptotic_grayscale_loss_nats"],
                "asymptotic_watermark_only_acc": res.get("asymptotic_watermark_only_acc", np.nan),
                "asymptotic_watermark_only_loss_nats": res.get("asymptotic_watermark_only_loss_nats", np.nan),
                "min_size": res["min_size"],
                "max_size": res["max_size"],
            }
    else:
        k_p_results_env = _compute_k_p_results(
            df_nats, skip_bayes, kp_abs_delta_threshold=kp_abs_delta_threshold
        )

    if len(k_p_results_env) < 2:
        print("  WARNING: fewer than 2 models — skipping figure.")
        return None

    # ── Build threshold models & envelope (combined feature loop) ─────
    x_min = df_nats["dataset_size"].min()
    x_max = df_nats["dataset_size"].max()
    x_linear = np.logspace(np.log10(max(x_min, 1)), np.log10(x_max), 2000)

    has_wm_only = "mean_watermark_only_acc" in df_nats.columns

    threshold_models: list = []
    threshold_lines: list = []

    for feature in [FEATURE_B, FEATURE_A]:
        for cand in [m for m in model_types_in_pcl if feature["model_key"] in m.lower()]:
            mdf = df_nats[df_nats["model_type"] == cand].sort_values("dataset_size")
            if len(mdf) < 2:
                continue
            tms_interp = build_interpolated_threshold_models(
                mdf,
                feature["model_key"],
                feature["name"],
                feature["threshold_type"],
                nb_models=nb_interpolated_models,
                has_watermark_only=has_wm_only,
                interpolation_metric=threshold_metric,
            )
            threshold_models.extend(tms_interp)
            threshold_lines.extend(build_threshold_lines(tms_interp, x_linear))

    # Bayes intermediates
    if not skip_bayes and include_bayes_intermediates:
        for cand in [m for m in model_types_in_pcl if "bayes" in m.lower()]:
            mdf = df_nats[df_nats["model_type"] == cand].sort_values("dataset_size")
            if len(mdf) < 2:
                continue
            tms_interp = build_interpolated_threshold_models(
                mdf,
                "bayes",
                "Bayes",
                BAYES_THRESHOLD_TYPE,
                nb_models=nb_interpolated_models,
                has_watermark_only=has_wm_only,
                interpolation_metric=threshold_metric,
            )
            threshold_models.extend(tms_interp)
            threshold_lines.extend(build_threshold_lines(tms_interp, x_linear))

    # p-value / acc_drop columns from PCL df
    pvalue_cols_from_df = [
        c for c in df_nats.columns if c.startswith("mean_pvalue_") or c.startswith("mean_acc_drop_")
    ]

    # Add asymptotic models
    for model_name in k_p_results_env:
        if FEATURE_A["model_key"] in model_name.lower():
            ft_name, ft_type = FEATURE_A["name"], FEATURE_A["threshold_type"]
        elif FEATURE_B["model_key"] in model_name.lower():
            ft_name, ft_type = FEATURE_B["name"], FEATURE_B["threshold_type"]
        elif "bayes" in model_name.lower():
            if skip_bayes:
                continue
            ft_name, ft_type = "Bayes", BAYES_THRESHOLD_TYPE
        else:
            continue
        asym = add_asymptotic_model(
            k_p_results_env,
            model_name,
            ft_name,
            ft_type,
            pvalue_cols_from_df,
            df_nats,
        )
        threshold_models.append(asym)
        threshold_lines.append(asym["k_p"] + x_linear * asym["slope"])

    if not threshold_models:
        print("  WARNING: no threshold models built — skipping figure.")
        return None

    # ── Envelope & MDL predictions ────────────────────────────────────
    envelope_indices = compute_envelope_indices(threshold_models, x_min, x_max)
    _ = find_envelope_intersections(threshold_models, x_min, x_max)  # computed for side-effects / future use
    mdl_pred = get_mdl_predicted_quantities(threshold_models, x_linear)
    transitions = find_type_transitions(mdl_pred["model_type"], x_linear)

    # ── Exp2 data arrays ──────────────────────────────────────────────
    exp2_sizes = summary_df["dataset_size"].values
    exp2_val_acc = summary_df["val_acc_mean"].values
    exp2_train_acc = summary_df["train_acc_mean"].values if "train_acc_mean" in summary_df.columns else None
    exp2_secondary_acc = (
        summary_df[SECONDARY_ACC_COL].values if SECONDARY_ACC_COL in summary_df.columns else None
    )
    exp2_wm_acc = (
        summary_df["watermark_only_acc_mean"].values
        if "watermark_only_acc_mean" in summary_df.columns
        else None
    )
    exp2_color_only_acc = (
        summary_df["color_only_acc_mean"].values if "color_only_acc_mean" in summary_df.columns else None
    )
    exp2_digit_only_acc = (
        summary_df["digit_only_acc_mean"].values if "digit_only_acc_mean" in summary_df.columns else None
    )

    # ── Color-maps for gradient lines ─────────────────────────────────
    asymptotic_model_colors: dict = {}
    for model_name, res in k_p_results_env.items():
        if FEATURE_B["model_key"] in model_name.lower():
            asymptotic_model_colors[FEATURE_B["threshold_type"]] = COLOR_MAP.get(model_name, "tab:blue")
        elif FEATURE_A["model_key"] in model_name.lower():
            asymptotic_model_colors[FEATURE_A["threshold_type"]] = COLOR_MAP.get(model_name, "tab:orange")
        elif "bayes" in model_name.lower():
            asymptotic_model_colors[BAYES_THRESHOLD_TYPE] = COLOR_MAP.get(model_name, "tab:green")

    colormaps_by_type: dict = {}
    for mtype, target_color in asymptotic_model_colors.items():
        cmap = LinearSegmentedColormap.from_list(
            f"{mtype}_gradient",
            [(0.0, "#C2C2C2"), (1.0, target_color)],
            N=256,
        )
        colormaps_by_type[mtype] = cmap

    # Dynamic normalization based on threshold_metric and actual data range
    all_thresholds = [m["threshold"] for m in threshold_models if "threshold" in m]
    if all_thresholds:
        t_min, t_max = min(all_thresholds), max(all_thresholds)
        # Use LogNorm for K(p) thresholds (large range), Normalize for accuracy (0-1 range)
        if "k_p" in threshold_metric.lower() or "k(p)" in threshold_metric.lower():
            # K(p) values: use log scale with actual data range
            norm_all = mpl.colors.LogNorm(vmin=max(t_min, 1), vmax=t_max)
        else:
            # Accuracy or other metrics: use linear scale
            norm_all = mpl.colors.Normalize(vmin=t_min, vmax=t_max)
    else:
        # Fallback to default
        norm_all = mpl.colors.Normalize(vmin=0.5, vmax=1.0)

    color_secondary = asymptotic_model_colors.get(FEATURE_B["threshold_type"], "#2ca02c")
    color_feature_a = asymptotic_model_colors.get(FEATURE_A["threshold_type"], "#1f77b4")

    # =====================================================================
    # FIGURE: 2 rows × 3 cols
    # Row 1: PCL curves | Asymptotic compression | Envelope compression
    # Row 2: Accuracy   | P-values               | Accuracy drop
    # =====================================================================
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(30, 12),
        gridspec_kw={"width_ratios": [1, 1, 1]},
    )
    ax_pcl = axes[0, 0]
    ax_asym = axes[0, 1]
    ax_compress = axes[0, 2]
    ax_acc = axes[1, 0]
    ax_pval = axes[1, 1]
    ax_acc_drop = axes[1, 2]

    # -----------------------------------------------------------------
    # Row 1, Col 0: PCL curves
    # -----------------------------------------------------------------
    ax_pcl_acc = ax_pcl.twinx()
    for model_type in pcl_results_df["model_type"].unique():
        if skip_bayes and "bayes" in model_type.lower():
            continue
        mdf = pcl_results_df[pcl_results_df["model_type"] == model_type].sort_values("dataset_size")
        lc = COLOR_MAP.get(model_type, "tab:purple")
        ax_pcl.plot(
            mdf["dataset_size"],
            mdf["mean_test_log_loss"],
            label=model_type,
            color=lc,
            linewidth=2.5,
            alpha=0.85,
        )
        if "mean_original_acc" in mdf.columns:
            ax_pcl_acc.plot(
                mdf["dataset_size"],
                mdf["mean_original_acc"],
                color=lc,
                linewidth=2.0,
                alpha=0.55,
                linestyle="--",
            )
        if "std_test_log_loss" in mdf.columns:
            lo = np.maximum(mdf["mean_test_log_loss"] - mdf["std_test_log_loss"], 0)
            hi = mdf["mean_test_log_loss"] + mdf["std_test_log_loss"]
            ax_pcl.fill_between(mdf["dataset_size"], lo, hi, color=lc, alpha=0.15)

    ax_pcl.set_xscale("log")
    ax_pcl.set_xlabel("Dataset size (N)", fontweight="bold")
    ax_pcl.set_ylabel("Mean test log-loss (nats/sample)", fontweight="bold")
    ax_pcl_acc.set_ylabel("Original accuracy", fontweight="bold")
    ax_pcl_acc.set_ylim(0.0, 1.02)
    ax_pcl.set_title("PCL Curves (Experiment 1)", fontweight="bold", pad=15)
    ax_pcl.grid(True, ls="--", alpha=0.4)
    ax_pcl.set_xlim(x_min, x_plot_max)

    pcl_handles = []
    for model_type in pcl_results_df["model_type"].unique():
        if skip_bayes and "bayes" in model_type.lower():
            continue
        lc = COLOR_MAP.get(model_type, "tab:purple")
        res = k_p_results_env.get(model_type, {})
        asym_acc = res.get("asymptotic_original_acc", float("nan"))
        asym_loss_nats = res.get("asymptotic_original_loss_nats", float("nan"))
        asym_bits = asym_loss_nats * NATS_TO_BITS if not np.isnan(asym_loss_nats) else float("nan")
        lbl = f"{model_type}  (acc={asym_acc:.2f}, {asym_bits:.3f} bits/s)"
        pcl_handles.append(Patch(facecolor=lc, label=lbl))
    ax_pcl.legend(handles=pcl_handles, loc="best", frameon=True, shadow=True, fancybox=True)
    ax_pcl.spines["top"].set_visible(False)
    ax_pcl.spines["right"].set_visible(False)

    # -----------------------------------------------------------------
    # Row 1, Col 1: Asymptotic compression (final models only)
    # -----------------------------------------------------------------
    asym_handles = []
    for model_name, res in k_p_results_env.items():
        lc = COLOR_MAP.get(model_name, "tab:purple")
        y_line = (res["Kp_nats"] + x_linear * res["asymptotic_original_loss_nats"]) * NATS_TO_BITS
        ax_asym.plot(x_linear, y_line, "-", color=lc, lw=2.5, alpha=0.85, label=model_name)
        asym_handles.append(Patch(facecolor=lc, label=model_name))

    ax_asym.set_xscale("log")
    ax_asym.set_yscale("log")
    ax_asym.set_xlim(x_min, x_plot_max)
    ax_asym.set_xlabel("Dataset size (N)", fontweight="bold")
    ax_asym.set_ylabel("Total description length (bits)", fontweight="bold")
    ax_asym.set_title(r"Asymptotic compression: K(p) + N$\cdot$log-loss", fontweight="bold", pad=15)
    ax_asym.grid(True, ls="--", alpha=0.4)
    ax_asym.legend(handles=asym_handles, loc="best", frameon=True, shadow=True, fancybox=True)
    ax_asym.spines["top"].set_visible(False)
    ax_asym.spines["right"].set_visible(False)

    # -----------------------------------------------------------------
    # Row 1, Col 2: Envelope compression (intermediate models)
    # -----------------------------------------------------------------
    for i_env, (model, line_vals) in enumerate(zip(threshold_models, threshold_lines)):
        on_envelope = i_env in envelope_indices
        if not on_envelope and not show_non_envelope_lines:
            continue
        mtype = model["model_type"]
        if mtype not in colormaps_by_type:
            continue
        cmap_env = colormaps_by_type[mtype]
        color_env = cmap_env(norm_all(model["threshold"]))
        lw = 2.0 if on_envelope else 0.8
        alpha = 0.75 if on_envelope else 0.20
        ax_compress.plot(x_linear, line_vals * NATS_TO_BITS, "-", lw=lw, alpha=alpha, color=color_env)

    # Add asymptotic lines on envelope plot
    for model_name, res in k_p_results_env.items():
        lc = COLOR_MAP.get(model_name, "tab:purple")
        y_line = (res["Kp_nats"] + x_linear * res["asymptotic_original_loss_nats"]) * NATS_TO_BITS
        ax_compress.plot(x_linear, y_line, "-", color=lc, lw=2.5, alpha=0.85, label=model_name)

    ax_compress.set_xscale("log")
    ax_compress.set_yscale("log")
    ax_compress.set_xlim(x_min, x_plot_max)
    ax_compress.set_xlabel("Dataset size (N)", fontweight="bold")
    ax_compress.set_ylabel("Total description length (bits)", fontweight="bold")
    ax_compress.set_title(r"Compression envelope: K(p) + N$\cdot$log-loss", fontweight="bold", pad=15)
    ax_compress.grid(True, ls="--", alpha=0.4)
    ax_compress.legend(loc="best", fontsize=9, frameon=True, shadow=True, fancybox=True)
    ax_compress.spines["top"].set_visible(False)
    ax_compress.spines["right"].set_visible(False)

    # Sync y-axis between asymptotic and envelope
    y_lo = min(ax_asym.get_ylim()[0], ax_compress.get_ylim()[0])
    y_hi = max(ax_asym.get_ylim()[1], ax_compress.get_ylim()[1])
    ax_asym.set_ylim(y_lo, y_hi)
    ax_compress.set_ylim(y_lo, y_hi)

    # Transition lines on all Row-2 axes (and envelope / asym)
    if show_transition_lines:
        for tp in transitions:
            x_line = tp["N"]
            if x_line < 20:
                continue
            for ax in [ax_asym, ax_compress, ax_acc, ax_pval, ax_acc_drop]:
                ax.axvline(x=x_line, color="#8B4789", linestyle="-", alpha=0.65, linewidth=2.5)

    # -----------------------------------------------------------------
    # Row 2, Col 0: Accuracy (learned=solid, MDL=dashed)
    # -----------------------------------------------------------------
    ax_acc.plot(exp2_sizes, exp2_val_acc, "-", color=COLOR_VAL_ACC, lw=2.5, alpha=0.85)
    if exp2_train_acc is not None:
        ax_acc.plot(exp2_sizes, exp2_train_acc, "-", color=COLOR_TRAIN_ACC, lw=2.5, alpha=0.85)
    # In Setting 1, digit_only_acc already captures the same signal as secondary (grayscale) acc,
    # so skip the secondary line to avoid a duplicate curve of the same colour.
    show_secondary = exp2_secondary_acc is not None and not (
        experiment_setting == 1 and exp2_digit_only_acc is not None
    )
    if show_secondary:
        ax_acc.plot(exp2_sizes, exp2_secondary_acc, "-", color=color_secondary, lw=2.5, alpha=0.85)
    if exp2_wm_acc is not None and experiment_setting == 2:
        ax_acc.plot(exp2_sizes, exp2_wm_acc, "-", color=color_feature_a, lw=2.5, alpha=0.85)
    # Setting 1: color-only and digit-only test accuracies
    if exp2_color_only_acc is not None and experiment_setting == 1:
        ax_acc.plot(
            exp2_sizes, exp2_color_only_acc, "-", color=COLOR_MAP.get("color", "#ad5752"), lw=2.5, alpha=0.85
        )
    if exp2_digit_only_acc is not None and experiment_setting == 1:
        ax_acc.plot(
            exp2_sizes, exp2_digit_only_acc, "-", color=COLOR_MAP.get("digit", "#8ab169"), lw=2.5, alpha=0.85
        )

    mdl_pred_at_sizes = get_mdl_predicted_quantities(
        threshold_models,
        x_query=np.asarray(exp2_sizes, dtype=float),
    )
    ax_acc.plot(exp2_sizes, mdl_pred_at_sizes["accuracy"], "--", color=COLOR_VAL_ACC, lw=2.5, alpha=0.7)
    ax_acc.plot(
        exp2_sizes, mdl_pred_at_sizes["grayscale_accuracy"], "--", color=color_secondary, lw=2.5, alpha=0.7
    )
    if not np.all(np.isnan(mdl_pred_at_sizes["watermark_only_accuracy"])) and experiment_setting == 2:
        ax_acc.plot(
            exp2_sizes,
            mdl_pred_at_sizes["watermark_only_accuracy"],
            "--",
            color=color_feature_a,
            lw=2.5,
            alpha=0.7,
        )

    ax_acc.set_xscale("log")
    ax_acc.set_xlim(x_min, x_plot_max)
    ax_acc.set_ylim(0.45, 1.02)
    ax_acc.set_xlabel("Dataset size (N)", fontweight="bold")
    ax_acc.set_ylabel("Accuracy", fontweight="bold")
    ax_acc.set_title("Accuracy", fontweight="bold", pad=15)
    ax_acc.grid(True, which="both", ls="--", alpha=0.3)
    ax_acc.spines["top"].set_visible(False)
    ax_acc.spines["right"].set_visible(False)

    # Split legend: color = eval dataset, line-style = model source
    dataset_handles = [
        Line2D([0], [0], color=COLOR_TRAIN_ACC, lw=2.5, label="Train"),
        Line2D([0], [0], color=COLOR_VAL_ACC, lw=2.5, label="Val"),
    ]
    if show_secondary:
        dataset_handles.append(Line2D([0], [0], color=color_secondary, lw=2.5, label=SECONDARY_ACC_LABEL))
    if exp2_wm_acc is not None and experiment_setting == 2:
        dataset_handles.append(Line2D([0], [0], color=color_feature_a, lw=2.5, label="Watermark-only"))
    if exp2_color_only_acc is not None and experiment_setting == 1:
        dataset_handles.append(
            Line2D([0], [0], color=COLOR_MAP.get("color", "#ad5752"), lw=2.5, label="Color-only")
        )
    if exp2_digit_only_acc is not None and experiment_setting == 1:
        dataset_handles.append(
            Line2D([0], [0], color=COLOR_MAP.get("digit", "#8ab169"), lw=2.5, label="Digit-only")
        )
    style_handles = [
        Line2D([0], [0], color="gray", lw=2.5, ls="-", label="Learned"),
        Line2D([0], [0], color="gray", lw=2.5, ls="--", label="MDL predicted"),
    ]
    leg_dataset = ax_acc.legend(
        handles=dataset_handles,
        title="Eval. dataset",
        loc="center left",
        bbox_to_anchor=(0.0, 0.60),
        frameon=True,
        shadow=True,
        fancybox=True,
        fontsize=9,
        title_fontsize=9,
    )
    ax_acc.add_artist(leg_dataset)
    ax_acc.legend(
        handles=style_handles,
        title="Model",
        loc="upper left",
        frameon=True,
        shadow=True,
        fancybox=True,
        fontsize=9,
        title_fontsize=9,
    )

    # -----------------------------------------------------------------
    # Row 2, Col 1: P-value curves
    # -----------------------------------------------------------------
    for attr in permutation_attributes:
        pval_col = f"pvalue_{attr}_mean"
        pval_std_col = f"pvalue_{attr}_std"
        if pval_col in summary_df.columns:
            pvals = summary_df[pval_col].values
            if not np.all(np.isnan(pvals)):
                lc = COLOR_MAP.get(attr, "tab:gray")
                ax_pval.plot(exp2_sizes, pvals, "-", color=lc, lw=2.5, alpha=0.85)
                if pval_std_col in summary_df.columns:
                    stds = summary_df[pval_std_col].values
                    ax_pval.fill_between(
                        exp2_sizes,
                        np.maximum(pvals - stds, 0),
                        np.minimum(pvals + stds, 1),
                        color=lc,
                        alpha=0.15,
                    )
        mdl_pval_key = f"mean_pvalue_{attr}"
        if mdl_pval_key in mdl_pred_at_sizes:
            mdl_pvals = mdl_pred_at_sizes[mdl_pval_key]
            if not np.all(np.isnan(mdl_pvals)):
                lc = COLOR_MAP.get(attr, "tab:gray")
                ax_pval.plot(exp2_sizes, mdl_pvals, "--", color=lc, lw=2.5, alpha=0.7)

    ax_pval.axhline(0.01, color="red", ls="--", lw=1.5, alpha=0.7, label="α=0.01")
    ax_pval.set_xscale("log")
    ax_pval.set_xlim(x_min, x_plot_max)
    ax_pval.set_ylim(-0.05, 1.05)
    ax_pval.set_xlabel("Dataset size (N)", fontweight="bold")
    ax_pval.set_ylabel("P-value", fontweight="bold")
    ax_pval.set_title("Feature Reliance: P-values (Permutation Test)", fontweight="bold", pad=15)
    ax_pval.grid(True, which="both", ls="--", alpha=0.3)
    ax_pval.spines["top"].set_visible(False)
    ax_pval.spines["right"].set_visible(False)

    # -----------------------------------------------------------------
    # Row 2, Col 2: Accuracy drop curves
    # -----------------------------------------------------------------
    for attr in permutation_attributes:
        acc_drop_col_name = f"acc_drop_{attr}_mean"
        acc_drop_std_name = f"acc_drop_{attr}_std"
        if acc_drop_col_name in summary_df.columns:
            drops = summary_df[acc_drop_col_name].values
            if not np.all(np.isnan(drops)):
                lc = COLOR_MAP.get(attr, "tab:gray")
                ax_acc_drop.plot(exp2_sizes, drops, "-", color=lc, lw=2.5, alpha=0.85)
                if acc_drop_std_name in summary_df.columns:
                    drops_std = summary_df[acc_drop_std_name].values
                    ax_acc_drop.fill_between(
                        exp2_sizes,
                        drops - drops_std,
                        drops + drops_std,
                        color=lc,
                        alpha=0.15,
                    )
        mdl_drop_key = f"mean_acc_drop_{attr}"
        if mdl_drop_key in mdl_pred_at_sizes:
            mdl_drops = mdl_pred_at_sizes[mdl_drop_key]
            if not np.all(np.isnan(mdl_drops)):
                lc = COLOR_MAP.get(attr, "tab:gray")
                ax_acc_drop.plot(exp2_sizes, mdl_drops, "--", color=lc, lw=2.5, alpha=0.7)

    ax_acc_drop.axhline(0, color="gray", ls="-", lw=1, alpha=0.5)
    ax_acc_drop.set_xscale("log")
    ax_acc_drop.set_xlim(x_min, x_plot_max)
    ax_acc_drop.set_xlabel("Dataset size (N)", fontweight="bold")
    ax_acc_drop.set_ylabel("Accuracy drop (original − shuffled)", fontweight="bold")
    ax_acc_drop.set_title("Feature Reliance: Accuracy Drop", fontweight="bold", pad=15)
    ax_acc_drop.grid(True, which="both", ls="--", alpha=0.3)
    ax_acc_drop.spines["top"].set_visible(False)
    ax_acc_drop.spines["right"].set_visible(False)

    # -----------------------------------------------------------------
    # Suptitle & save
    # -----------------------------------------------------------------
    fig.subplots_adjust(hspace=0.40, wspace=0.30, right=0.82)
    setting_str = f"Scenario {"A" if experiment_setting == 1 else "B"}"
    if experiment_setting == 1:
        sp_str = (
            f"spur={spur_prob:.2f} · flip={flip_prob:.2f} "
            f"· env_noise={env_noisiness:.2f} · uninf={uninformative_majority}"
        )
    elif experiment_setting == 2:
        sp_str = (
            f"spur={spur_prob:.2f} · flip={flip_prob:.2f} "
            f"· env_noise={env_noisiness:.2f} · bank_size={watermark_bank_size}"
        )
    else:
        sp_str = ""
    fig.suptitle(f"{setting_str}: {sp_str}")

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            save_dir / "experiment_summary_plots.png", dpi=300, bbox_inches="tight", facecolor="white"
        )
        fig.savefig(save_dir / "experiment_summary_plots.pdf", bbox_inches="tight", facecolor="white")
        print(f"  Saved plots to {save_dir}")

    return fig
