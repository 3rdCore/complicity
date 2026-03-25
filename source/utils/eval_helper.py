import netcal.metrics
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    recall_score,
    roc_auc_score,
)


def predict_on_set(algorithm, loader, device):
    num_labels = loader.dataset.num_labels

    ys, atts, gs, ps = [], [], [], []

    algorithm.eval()
    with torch.no_grad():
        for batch in loader:
            # Handle old format (i, x, y, a), 5-tuple (i, x, y, a, digit),
            # and 6-tuple CMNIST format (i, x, y, a, env, digit)
            if len(batch) == 6:
                _, x, y, a, env, digit = batch
            elif len(batch) == 5:
                _, x, y, a, digit = batch
            else:
                _, x, y, a = batch

            p = algorithm.predict(x.to(device))
            if p.squeeze().ndim == 1:
                p = torch.sigmoid(p).detach().cpu().numpy()
            else:
                p = torch.softmax(p, dim=-1).detach().cpu().numpy()
                if num_labels == 2:
                    p = p[:, 1]

            ps.append(p)
            ys.append(y)
            atts.append(a)
            gs.append([f"y={yi},a={gi}" for c, (yi, gi) in enumerate(zip(y, a))])

    return (
        np.concatenate(ys, axis=0),
        np.concatenate(atts, axis=0),
        np.concatenate(ps, axis=0),
        np.concatenate(gs),
    )


def eval_metrics(algorithm, loader, device, thres=0.5):
    targets, attributes, preds, gs = predict_on_set(algorithm, loader, device)
    preds_rounded = preds >= thres if preds.squeeze().ndim == 1 else preds.argmax(1)
    label_set = np.unique(targets)

    res = {}
    res["overall"] = {
        **binary_metrics(targets, preds_rounded, label_set),
        **prob_metrics(targets, preds, label_set),
    }
    res["per_attribute"] = {}
    res["per_class"] = {}
    res["per_group"] = {}

    for a in np.unique(attributes):
        mask = attributes == a
        res["per_attribute"][int(a)] = {
            **binary_metrics(targets[mask], preds_rounded[mask], label_set),
            **prob_metrics(targets[mask], preds[mask], label_set),
        }

    classes_report = classification_report(targets, preds_rounded, output_dict=True, zero_division=0.0)
    res["overall"]["macro_avg"] = classes_report["macro avg"]
    res["overall"]["weighted_avg"] = classes_report["weighted avg"]
    for y in np.unique(targets):
        res["per_class"][int(y)] = classes_report[str(y)]

    for g in np.unique(gs):
        mask = gs == g
        res["per_group"][g] = {**binary_metrics(targets[mask], preds_rounded[mask], label_set)}

    # Adjusted accuracy averages accuracy across groups equally (not weighted by group size)
    res["adjusted_accuracy"] = sum([res["per_group"][g]["accuracy"] for g in np.unique(gs)]) / len(
        np.unique(gs)
    )

    # --- Aggregate per-attribute stats (min/max per metric across attributes) ---
    attr_df = pd.DataFrame(res["per_attribute"]) if len(res["per_attribute"]) else pd.DataFrame()
    if not attr_df.empty:
        res["min_attr"] = attr_df.min(axis=1).to_dict()
        res["max_attr"] = attr_df.max(axis=1).to_dict()
    else:
        res["min_attr"], res["max_attr"] = {}, {}

    # --- Aggregate per-group stats ---
    group_df = pd.DataFrame(res["per_group"]) if len(res["per_group"]) else pd.DataFrame()
    if not group_df.empty:
        # Preserve original API: per-metric min/max across groups
        res["min_group"] = group_df.min(axis=1).to_dict()
        res["max_group"] = group_df.max(axis=1).to_dict()

        # Also expose richer summaries: identify the single worst group by accuracy
        if "accuracy" in group_df.index:
            worst_group_id = group_df.loc["accuracy"].idxmin()
            best_group_id = group_df.loc["accuracy"].idxmax()
            res["worst_group_id"] = worst_group_id
            res["best_group_id"] = best_group_id
            # Copy full metric dicts for those groups
            res["worst_group_full"] = res["per_group"][worst_group_id]
            res["best_group_full"] = res["per_group"][best_group_id]
    else:
        res["min_group"], res["max_group"] = {}, {}

    # NOTE: If you observed all zeros in min_group, that likely means at least one group
    # had zero values for each metric (e.g., tiny or degenerate groups). Inspect
    # res["worst_group_full"] and res["per_group"] for detailed diagnostics.

    return res


def binary_metrics(targets, preds, label_set=[0, 1], return_arrays=False):
    if len(targets) == 0:
        return {}

    res = {"accuracy": accuracy_score(targets, preds), "n_samples": len(targets)}

    if len(label_set) == 2:
        CM = confusion_matrix(targets, preds, labels=label_set)

        res["TN"] = CM[0][0].item()
        res["FN"] = CM[1][0].item()
        res["TP"] = CM[1][1].item()
        res["FP"] = CM[0][1].item()
        res["error"] = res["FN"] + res["FP"]

        if res["TP"] + res["FN"] == 0:
            res["TPR"] = 0
            res["FNR"] = 1
        else:
            res["TPR"] = res["TP"] / (res["TP"] + res["FN"])
            res["FNR"] = res["FN"] / (res["TP"] + res["FN"])

        if res["FP"] + res["TN"] == 0:
            res["FPR"] = 1
            res["TNR"] = 0
        else:
            res["FPR"] = res["FP"] / (res["FP"] + res["TN"])
            res["TNR"] = res["TN"] / (res["FP"] + res["TN"])

        res["pred_prevalence"] = (res["TP"] + res["FP"]) / res["n_samples"]
        res["prevalence"] = (res["TP"] + res["FN"]) / res["n_samples"]
    else:
        CM = confusion_matrix(targets, preds, labels=label_set)
        res["TPR"] = recall_score(targets, preds, labels=label_set, average="macro", zero_division=0.0)

    if len(np.unique(targets)) > 1:
        res["balanced_acc"] = balanced_accuracy_score(targets, preds)

    if return_arrays:
        res["targets"] = targets
        res["preds"] = preds

    return res


def prob_metrics(targets, preds, label_set, return_arrays=False):
    if len(targets) == 0:
        return {}

    res = {
        "AUROC_ovo": roc_auc_score(targets, preds, multi_class="ovo", labels=label_set),
        "BCE": log_loss(targets, preds, labels=label_set),
        "ECE": netcal.metrics.ECE().measure(preds, targets),
    }

    # happens when you predict a class, but there are no samples with that class in the dataset
    try:
        res["AUROC"] = roc_auc_score(targets, preds, multi_class="ovr", labels=label_set)
    except Exception:
        res["AUROC"] = roc_auc_score(targets, preds, multi_class="ovo", labels=label_set)

    if len(set(targets)) == 2:
        res["AUPRC"] = average_precision_score(targets, preds, average="macro")
        res["brier"] = brier_score_loss(targets, preds)

    if return_arrays:
        res["targets"] = targets
        res["preds"] = preds

    return res


# =============================================================================
# Enveloppe computation utilities
# =============================================================================

NATS_TO_BITS = 1.0 / np.log(2)


def area_under_curve_up_to(x_vals, y_vals, N):
    """Compute area under curve from x_vals[0] to N.

    Returns (area, y_at_N).
    """
    if N <= x_vals[0]:
        return 0.0, float(y_vals[0])
    if N >= x_vals[-1]:
        return np.trapz(y_vals, x_vals), float(y_vals[-1])
    yN = float(np.interp(N, x_vals, y_vals))
    idx = np.searchsorted(x_vals, N)
    x_part = np.concatenate([x_vals[:idx], [N]])
    y_part = np.concatenate([y_vals[:idx], [yN]])
    area = np.trapz(y_part, x_part)
    return area, yN


def build_interpolated_threshold_models(
    df,
    feature_key,
    feature_name,
    threshold_type,
    nb_models=100,
    has_watermark_only=False,
    interpolation_metric="mean_original_acc",
):
    """Build intermediate threshold models by sampling an interpolation metric.

    The function samples ``nb_models`` linearly-spaced values of the chosen
    ``interpolation_metric`` between its value at the smallest and largest
    training sizes, then inverts the metric-vs-N curve to find the
    corresponding training size N for each sampled value.  All other metrics
    are interpolated at that N.

    Parameters
    ----------
    df : DataFrame
        Filtered PCL results for a single model type, sorted by dataset_size.
    feature_key, feature_name, threshold_type : str
        Identifiers for the feature
    nb_models : int
        Number of intermediate values to sample (linearly spaced).
    has_watermark_only : bool
        Whether watermark_only columns exist.
    interpolation_metric : str
        Column used to drive the interpolation.  Supported values:
        ``"mean_original_log_loss"`` (decreasing with N),
        ``"mean_original_acc"`` (increasing with N), or
        ``"k(p)"`` which linearly interpolates K(p) from 0 to its
        asymptotic (last dataset-size) value.

    Returns
    -------
    list of dict
        Interpolated threshold model dicts.
    """

    x = df["dataset_size"].values.astype(float)
    y_test = df["mean_test_log_loss"].values.astype(float)
    y_orig = df["mean_original_log_loss"].values.astype(float)
    y_acc = df["mean_test_acc"].values.copy().astype(float)
    y_orig_acc = df["mean_original_acc"].values.astype(float)
    y_grayscale_acc = df["mean_grayscale_acc"].values.astype(float)
    y_grayscale_ll = df["mean_grayscale_log_loss"].values.astype(float)
    y_wm_acc = df["mean_watermark_only_acc"].values.astype(float) if has_watermark_only else None
    y_wm_ll = df["mean_watermark_only_log_loss"].values.astype(float) if has_watermark_only else None

    has_color_only = "mean_color_only_acc" in df.columns
    y_co_acc = df["mean_color_only_acc"].values.astype(float) if has_color_only else None
    has_digit_only = "mean_digit_only_acc" in df.columns
    y_do_acc = df["mean_digit_only_acc"].values.astype(float) if has_digit_only else None

    pvalue_cols = [c for c in df.columns if c.startswith("mean_pvalue_")]
    acc_drop_cols = [c for c in df.columns if c.startswith("mean_acc_drop_")]

    # Handle NaN in accuracy (Bayes fallback)
    nan_mask = np.isnan(y_acc)
    if nan_mask.any():
        y_acc[nan_mask] = y_orig_acc[nan_mask]

    if len(x) < 2:
        return []

    # --- Precompute K(p) at each dataset size ---
    y_kp = np.zeros_like(x)
    for i_size, N_i in enumerate(x):
        area_i, test_loss_i = area_under_curve_up_to(x, y_test, N_i)
        width_i = N_i - x[0]
        y_kp[i_size] = max(area_i - test_loss_i * width_i, 0.0)

    # --- Select the interpolation driver curve ---
    _metric_key = interpolation_metric.lower().strip()
    if _metric_key == "k(p)":
        # K(p) is increasing with N (from 0 to asymptotic)
        y_driver = y_kp
        driver_is_decreasing = False
    elif _metric_key == "mean_original_log_loss":
        y_driver = y_orig  # decreasing with N
        driver_is_decreasing = True
    else:  # mean_original_acc
        y_driver = y_orig_acc  # increasing with N
        driver_is_decreasing = False

    driver_first = float(y_driver[0])
    driver_last = float(y_driver[-1])

    # For k(p) mode, sample from 0 to K(p) asymptotic (log-spaced)
    if _metric_key == "k(p)":
        driver_first = 0.0
        driver_last = float(y_kp[-1])
        if driver_last <= 0.0:
            return []
        # Log-spaced sampling: use a small epsilon as lower bound to avoid log(0)
        eps = driver_last / (nb_models * 100)
        driver_samples = np.concatenate(
            [[0.0], np.logspace(np.log10(eps), np.log10(driver_last), nb_models - 1)]
        )
    else:
        # Ensure there is actual variation to interpolate over
        if driver_is_decreasing and driver_last >= driver_first:
            return []
        if not driver_is_decreasing and driver_last <= driver_first:
            return []
        driver_samples = np.linspace(driver_first, driver_last, nb_models)

    # np.interp requires the x-array to be *increasing*.  For a decreasing
    # driver we flip both arrays so the driver values become ascending.
    if driver_is_decreasing:
        y_driver_interp = y_driver[::-1]
        x_interp = x[::-1]
    else:
        y_driver_interp = y_driver
        x_interp = x

    threshold_models = []
    for driver_target in driver_samples:
        # Invert: find N corresponding to this driver metric value
        N = float(np.interp(driver_target, y_driver_interp, x_interp))

        # K(p) via area under test-loss curve up to N
        k_p_i, test_loss_at_N = area_under_curve_up_to(x, y_test, N)
        width_N = N - x[0]
        k_p_i = k_p_i - test_loss_at_N * width_N

        # Interpolate all other metrics at N
        acc_at_N = float(np.interp(N, x, y_acc))
        orig_loss_at_N = float(np.interp(N, x, y_orig))
        orig_acc_at_N = float(np.interp(N, x, y_orig_acc))
        gs_acc_at_N = float(np.interp(N, x, y_grayscale_acc))
        gs_ll_at_N = float(np.interp(N, x, y_grayscale_ll))
        wm_acc_at_N = float(np.interp(N, x, y_wm_acc)) if y_wm_acc is not None else np.nan
        wm_ll_at_N = float(np.interp(N, x, y_wm_ll)) if y_wm_ll is not None else np.nan
        co_acc_at_N = float(np.interp(N, x, y_co_acc)) if y_co_acc is not None else np.nan
        do_acc_at_N = float(np.interp(N, x, y_do_acc)) if y_do_acc is not None else np.nan

        if _metric_key == "k(p)":
            interp_label = f"K(p)={driver_target:.2f}"
        elif _metric_key == "mean_original_log_loss":
            interp_label = f"loss={driver_target:.4f}"
        else:
            interp_label = f"acc={driver_target:.4f}"

        model = {
            "name": f"{feature_name} {interp_label}",
            "threshold": acc_at_N,
            "train_size": N,
            "k_p": max(k_p_i, 0.0),
            "slope": orig_loss_at_N,
            "accuracy": acc_at_N,
            "original_accuracy": orig_acc_at_N,
            "grayscale_accuracy": gs_acc_at_N,
            "grayscale_log_loss": gs_ll_at_N,
            "watermark_only_accuracy": wm_acc_at_N,
            "watermark_only_log_loss": wm_ll_at_N,
            "color_only_accuracy": co_acc_at_N,
            "digit_only_accuracy": do_acc_at_N,
            "test_log_loss": test_loss_at_N,
            "model_type": threshold_type,
        }
        for col in pvalue_cols + acc_drop_cols:
            col_vals = df[col].values.astype(float)
            model[col] = float(np.interp(N, x, col_vals))
        threshold_models.append(model)

    return threshold_models


def build_threshold_lines(threshold_models, x_linear):
    """Compute compression lines K(p) + x * slope for each threshold model.

    Kept for plotting — the sampled arrays are used by matplotlib.
    Envelope computation now uses the closed-form (k_p, slope) directly.
    """
    return [m["k_p"] + x_linear * m["slope"] for m in threshold_models]


def add_asymptotic_model(
    k_p_results, model_name, feature_name, threshold_type, pvalue_cols_from_df=None, df=None
):
    """Create a final (asymptotic) model entry from k_p_results.

    All input values in k_p_results are expected in NATS (with `_nats` suffix).
    The returned threshold_model dict keeps k_p and slope in NATS (same as
    build_threshold_models).  Convert to bits only at plotting time.

    Returns a threshold_model dict with k_p and slope in NATS.
    """
    res = k_p_results[model_name]
    model = {
        "name": f"{feature_name} final ({model_name})",
        "threshold": 1.0,
        "train_size": res["max_size"],
        "k_p": res["Kp_nats"],
        "slope": res["asymptotic_original_loss_nats"],
        "accuracy": res["asymptotic_test_acc"],
        "original_accuracy": res["asymptotic_original_acc"],
        "grayscale_accuracy": res["asymptotic_grayscale_acc"],
        "grayscale_log_loss": res["asymptotic_grayscale_loss_nats"],
        "watermark_only_accuracy": res.get("asymptotic_watermark_only_acc", np.nan),
        "watermark_only_log_loss": res.get("asymptotic_watermark_only_loss_nats", np.nan),
        "color_only_accuracy": res.get("asymptotic_color_only_acc", np.nan),
        "digit_only_accuracy": res.get("asymptotic_digit_only_acc", np.nan),
        "test_log_loss": res["asymptotic_test_loss_nats"],
        "model_type": threshold_type,
    }
    # Copy pvalue columns from the last row of the df
    if pvalue_cols_from_df is not None and df is not None:
        model_df = df[df["model_type"] == model_name].sort_values("dataset_size")
        if not model_df.empty:
            for col in pvalue_cols_from_df:
                model[col] = float(model_df[col].iloc[-1]) if col in model_df.columns else np.nan
    return model


# =============================================================================
# Analytical lower-envelope utilities
# =============================================================================
# Each threshold model defines an affine line  y = k_p + slope * x.
# All envelope queries reduce to argmin_i(k_p_i + slope_i * x) and
# closed-form pairwise intersections.
# =============================================================================


def _line_value(model, x):
    """Evaluate a model's compression line at *x*."""
    return model["k_p"] + model["slope"] * x


def _line_intersection_x(m1, m2):
    """Return the x-coordinate where two affine compression lines cross.

    Returns *None* when slopes are equal (parallel lines).
    """
    ds = m1["slope"] - m2["slope"]
    if ds == 0.0:
        return None
    return (m2["k_p"] - m1["k_p"]) / ds


def get_best_line_at_N(N_test, threshold_models):
    """Find which model index has minimum value at *N_test*.

    Works directly from the (k_p, slope) stored in each model dict —
    no sampled arrays needed.

    Parameters
    ----------
    N_test : float
    threshold_models : list[dict]
        Each dict must contain ``k_p`` and ``slope``.

    Returns
    -------
    (best_idx, best_value)
    """
    values = np.array([m["k_p"] + m["slope"] * N_test for m in threshold_models])
    best_idx = int(np.argmin(values))
    return best_idx, float(values[best_idx])


def compute_envelope_indices(threshold_models, x_min, x_max):
    """Find which model indices lie on the lower envelope in [x_min, x_max].

    Uses analytical pairwise intersections instead of numerical sampling.
    """
    n = len(threshold_models)
    if n == 0:
        return set()

    # Collect all candidate x-values: endpoints + pairwise intersections
    x_candidates = [float(x_min), float(x_max)]
    for i in range(n):
        for j in range(i + 1, n):
            x_cross = _line_intersection_x(threshold_models[i], threshold_models[j])
            if x_cross is not None and x_min <= x_cross <= x_max:
                x_candidates.append(x_cross)

    envelope_indices = set()
    for x_test in x_candidates:
        best_idx, _ = get_best_line_at_N(x_test, threshold_models)
        envelope_indices.add(best_idx)
        # Also probe just left/right of intersections to catch both sides
        eps = max(1.0, abs(x_test) * 1e-8)
        if x_test - eps >= x_min:
            idx_l, _ = get_best_line_at_N(x_test - eps, threshold_models)
            envelope_indices.add(idx_l)
        if x_test + eps <= x_max:
            idx_r, _ = get_best_line_at_N(x_test + eps, threshold_models)
            envelope_indices.add(idx_r)

    return envelope_indices


def find_envelope_intersections(threshold_models, x_min, x_max):
    """Find all pairwise intersections that lie on the lower envelope.

    Uses the closed-form intersection  x = (k_p_j - k_p_i) / (slope_i - slope_j).
    Determines which model is below the other before/after intersection using slope comparison.
    """
    n = len(threshold_models)
    envelope_intersections = []
    for i in range(n):
        for j in range(i + 1, n):
            x_cross = _line_intersection_x(threshold_models[i], threshold_models[j])
            if x_cross is None or not (x_min <= x_cross <= x_max):
                continue
            y_cross = _line_value(threshold_models[i], x_cross)

            # Use slope comparison to determine which model is below before/after intersection
            # If slope_i < slope_j: i is above j before x_cross, below after
            # If slope_i > slope_j: i is below j before x_cross, above after
            slope_i = threshold_models[i]["slope"]
            slope_j = threshold_models[j]["slope"]

            if slope_i < slope_j:
                # i shallower than j: j→i transition (j before, i after)
                model_before_idx, model_after_idx = j, i
            else:
                # i steeper than j: i→j transition (i before, j after)
                model_before_idx, model_after_idx = i, j

            # Verify both models are actually on the envelope at their respective sides
            # We still need to check against ALL models, not just i and j
            # Use a small offset to avoid numerical issues exactly at the intersection
            eps = max(1.0, abs(x_cross) * 1e-8)
            best_before, _ = get_best_line_at_N(x_cross - eps, threshold_models)
            best_after, _ = get_best_line_at_N(x_cross + eps, threshold_models)

            if best_before == model_before_idx and best_after == model_after_idx:
                envelope_intersections.append(
                    {
                        "N": x_cross,
                        "y": y_cross,
                        "model_from": threshold_models[model_before_idx]["name"],
                        "model_to": threshold_models[model_after_idx]["name"],
                    }
                )
    envelope_intersections.sort(key=lambda p: p["N"])
    return envelope_intersections


def get_mdl_predicted_quantities(threshold_models, x_query):
    """For each x in *x_query*, find the best-compressor model and extract its metrics.

    Parameters
    ----------
    threshold_models : list[dict]
        Each dict must contain ``k_p``, ``slope``, and metric fields.
    x_query : array-like
        The x-coordinates at which to evaluate the envelope.

    Returns
    -------
    dict with arrays: accuracy, grayscale_accuracy, watermark_only_accuracy,
    model_name, model_type, plus p-value / acc-drop columns.
    """
    x_query = np.asarray(x_query, dtype=float)
    n = len(x_query)

    # Vectorised argmin: build (n_models, n_query) matrix
    kps = np.array([m["k_p"] for m in threshold_models])
    slopes = np.array([m["slope"] for m in threshold_models])
    # lines_at_x[i, j] = kps[i] + slopes[i] * x_query[j]
    lines_at_x = kps[:, None] + slopes[:, None] * x_query[None, :]
    best_indices = np.argmin(lines_at_x, axis=0)

    best_acc = np.empty(n)
    best_grayscale_acc = np.empty(n)
    best_watermark_only_acc = np.full(n, np.nan)
    best_color_only_acc = np.full(n, np.nan)
    best_digit_only_acc = np.full(n, np.nan)
    best_model_name = []
    best_model_type = []

    pvalue_keys = [k for k in threshold_models[0] if k.startswith("mean_pvalue_")]
    acc_drop_keys = [k for k in threshold_models[0] if k.startswith("mean_acc_drop_")]
    pvalue_arrays = {k: np.full(n, np.nan) for k in pvalue_keys}
    acc_drop_arrays = {k: np.full(n, np.nan) for k in acc_drop_keys}

    for i in range(n):
        m = threshold_models[best_indices[i]]
        best_acc[i] = m["accuracy"]
        best_grayscale_acc[i] = m["grayscale_accuracy"]
        if "watermark_only_accuracy" in m:
            best_watermark_only_acc[i] = m["watermark_only_accuracy"]
        if "color_only_accuracy" in m:
            best_color_only_acc[i] = m["color_only_accuracy"]
        if "digit_only_accuracy" in m:
            best_digit_only_acc[i] = m["digit_only_accuracy"]
        best_model_name.append(m["name"])
        best_model_type.append(m["model_type"])
        for k in pvalue_keys:
            pvalue_arrays[k][i] = m.get(k, np.nan)
        for k in acc_drop_keys:
            acc_drop_arrays[k][i] = m.get(k, np.nan)

    return {
        "accuracy": best_acc,
        "grayscale_accuracy": best_grayscale_acc,
        "watermark_only_accuracy": best_watermark_only_acc,
        "color_only_accuracy": best_color_only_acc,
        "digit_only_accuracy": best_digit_only_acc,
        "model_name": best_model_name,
        "model_type": np.array(best_model_type),
        **pvalue_arrays,
        **acc_drop_arrays,
    }


def find_type_transitions(model_types, x_linear):
    """Find x positions where the dominant model type changes."""
    transitions = []
    for i in range(1, len(model_types)):
        if model_types[i - 1] != model_types[i]:
            transitions.append(
                {
                    "N": x_linear[i],
                    "from_type": model_types[i - 1],
                    "to_type": model_types[i],
                }
            )
    return transitions
