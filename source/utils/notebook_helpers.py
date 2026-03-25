from __future__ import annotations

import gc
import math
import random
import sys
import time
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
import itertools
from itertools import product

from source import algorithms
from source.algorithms import get_optimizers
from source import networks
from source.utils.misc import (
    InfiniteDataLoader,
    attach_dataset_attributes,
    denormalize_cmnist,
    get_base_dataset,
)


def job_log(message: str) -> None:
    """Write progress logs to the job stdout (bypass notebook capture)."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    sys.__stdout__.write(f"[{timestamp}] {message}\n")
    sys.__stdout__.flush()


def plot_dataset_samples(dataset, title: str, num_samples: int = 10, seed: int = 42):
    """Plot sample images from a dataset."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    n = min(num_samples, len(dataset))
    sample_indices = rng.choice(len(dataset), size=n, replace=False)

    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3))
    if n == 1:
        axes = [axes]

    for col, i in enumerate(sample_indices):
        idx, img, label, attr, env, digit = dataset[i]

        # Denormalize and convert to numpy for plotting
        img_denorm = denormalize_cmnist(img).permute(1, 2, 0).numpy()
        img_denorm = np.clip(img_denorm, 0, 1)

        axes[col].imshow(img_denorm)
        axes[col].set_title(
            f"y={label.item()}, a={attr.item()}\nenv={env.item()}, d={digit.item()}",
            fontsize=8,
        )
        axes[col].axis("off")

    plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    return fig


def _resolve_base_index(ds, idx: int) -> int:
    return ds.indices[idx] if isinstance(ds, Subset) else idx


def plot_samples(ds, ds_name: str, num_samples: int = 10, seed_offset: int = 0, *, seed: int):
    """Plot a few samples from a dataset."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, num_samples, figsize=(12, 2.5))
    rng = np.random.default_rng(seed + 999 + seed_offset)
    base_ds = get_base_dataset(ds)
    total_available = len(ds)

    if total_available == 0:
        for ax in axes:
            ax.axis("off")
        return fig

    sample_count = min(num_samples, total_available)
    sample_indices = rng.choice(total_available, size=sample_count, replace=False)

    for col_idx, subset_idx in enumerate(sample_indices):
        ax = axes[col_idx]
        base_idx = _resolve_base_index(ds, subset_idx)
        index, img, label, attr, env, digit = base_ds[base_idx]

        img_denorm = denormalize_cmnist(img).permute(1, 2, 0).numpy()
        img_denorm = np.clip(img_denorm, 0, 1)

        ax.imshow(img_denorm)
        title_bits = []
        if label is not None:
            title_bits.append(f"y={label}")
        if attr is not None:
            title_bits.append(f"a={attr}")
        if env is not None:
            title_bits.append(f"e={env}")
        if digit is not None:
            title_bits.append(f"d={digit}")
        if title_bits:
            ax.set_title(", ".join(title_bits), fontsize=8)
        ax.axis("off")

    for col_idx in range(sample_count, num_samples):
        axes[col_idx].axis("off")

    fig.suptitle(f"{ds_name}: {num_samples} samples", fontsize=10)
    plt.tight_layout()
    return fig


def _get_cfg(cfg: Optional[Dict], key: str, default=None):
    if cfg is None:
        return default
    return cfg.get(key, default)


def get_mean_log_loss_and_accuracy(algo_instance, eval_loader, device):
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    algo_instance.eval()
    with torch.no_grad():
        for i, x, y, a, e, d in eval_loader:
            x, y = x.to(device), y.to(device)
            logits, loss = algo_instance.predict(x, y, return_loss=True)
            scalar_loss = loss.item() if isinstance(loss, torch.Tensor) else loss
            total_loss += scalar_loss * x.size(0)

            y_hat = logits.argmax(dim=-1) if logits.ndim > 1 else (logits > 0.5).long()
            total_correct += (y_hat == y).sum().item()
            total_samples += x.size(0)

    mean_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return mean_loss, accuracy


def compute_pcl_curve(
    train_dataset,
    val_dataset,
    test_dataset,
    original_dataset,
    hparams,
    device,
    dataset_sizes: Iterable[int],
    *,
    base_seed=None,
    config: Optional[Dict] = None,
    job_logger: Optional[Callable[[str], None]] = None,
    extra_eval_datasets: Optional[Dict[str, Any]] = None,
    permutation_test_config: Optional[Dict] = None,
    model_name: Optional[str] = None,
):
    """Compute PCL curve, optionally evaluating on extra held-out datasets.

    Parameters
    ----------
    extra_eval_datasets : dict, optional
        Mapping of ``{name: dataset}`` for additional evaluation.
        For each entry the trained model is evaluated at every dataset size
        and columns ``mean_{name}_acc``, ``std_{name}_acc``,
        ``mean_{name}_log_loss``, ``std_{name}_log_loss`` are added to the
        returned DataFrame.
    permutation_test_config : dict, optional
        If provided, run permutation tests at each dataset size.
        Keys: ``'dataset'`` (the dataset to permute), ``'attributes'`` (list of attr names),
        ``'n_permutations'`` (int), ``'batch_size'`` (int), ``'num_workers'`` (int).
    """
    if extra_eval_datasets is None:
        extra_eval_datasets = {}
    dataset_size = len(train_dataset)
    candidate_sizes = sorted(
        {max(1, min(dataset_size, int(size))) for size in dataset_sizes if int(size) > 0}
    )
    if not candidate_sizes:
        raise ValueError("No valid dataset sizes provided for PCL computation.")

    debug_mode = _get_cfg(config, "debug_mode", False)

    shuffle_seed = base_seed if base_seed is not None else _get_cfg(config, "seed", 0)
    batch_size = _get_cfg(config, "batch_size", 64)
    num_workers = _get_cfg(config, "num_workers", 0)
    small_data_threshold = _get_cfg(config, "small_data_threshold", 1000)
    num_runs_small = _get_cfg(config, "num_runs_small", 5)
    num_runs_large = _get_cfg(config, "num_runs_large", 5)
    num_runs_max_size = _get_cfg(config, "num_runs_max_size", 5)
    es_min_delta = _get_cfg(config, "es_min_delta", 1e-4)
    es_patience = _get_cfg(config, "es_patience", 5)
    debug_max_steps = _get_cfg(config, "debug_max_steps", None)
    learner = _get_cfg(config, "learner", None)

    if job_logger is None:
        job_logger = print

    print(f"Dataset size: {dataset_size}, Evaluating sizes: {candidate_sizes}")
    pcl_results = []

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )
    original_loader = DataLoader(
        original_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )

    # Build loaders for extra held-out evaluation datasets
    extra_loaders: Dict[str, DataLoader] = {}
    for extra_name, extra_ds in extra_eval_datasets.items():
        extra_loaders[extra_name] = DataLoader(
            extra_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
        )

    # At the largest dataset size, use num_runs_max_size runs (if set) to
    # improve the estimate of the asymptotic loss used for K(p) computation.
    _max_size = candidate_sizes[-1]
    _effective_runs_max_size = num_runs_max_size if num_runs_max_size is not None else num_runs_large
    max_run_count = max(num_runs_small, num_runs_large, _effective_runs_max_size)

    # Store results as: results_by_size[size] = [(run_0_metrics), (run_1_metrics), ...]
    results_by_size = {size: [] for size in candidate_sizes}

    for run_idx in range(max_run_count):
        run_seed_base = shuffle_seed + run_idx * 1000
        torch.manual_seed(run_seed_base)
        np.random.seed(run_seed_base)
        random.seed(run_seed_base)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(run_seed_base)

        permuted_indices = np.random.permutation(dataset_size)

        # Inner loop: iterate through dataset sizes using SAME permutation
        for num_samples in candidate_sizes:
            if num_samples == _max_size and num_runs_max_size is not None:
                run_count = _effective_runs_max_size
            elif num_samples <= small_data_threshold:
                run_count = num_runs_small
            else:
                run_count = num_runs_large

            # Skip this run if it exceeds the required count for this size
            if run_idx >= run_count:
                continue
            _label = model_name if model_name else "EXP1"
            job_logger(f"[{_label}] Run {run_idx + 1}/{run_count}, Dataset size: {num_samples}")

            # Use the SAME permutation, just take first 'num_samples' elements
            subset_indices = permuted_indices[:num_samples]
            train_subset = Subset(train_dataset, subset_indices.tolist())
            attach_dataset_attributes(train_subset)

            AlgorithmClass = algorithms.get_algorithm_class(learner)
            algo = AlgorithmClass(
                train_dataset.data_type,
                train_dataset.INPUT_SHAPE,
                train_dataset.num_labels,
                train_dataset.num_attributes,
                len(train_subset),
                hparams,
            )
            algo = algo.to(device)

            train_loader = InfiniteDataLoader(
                dataset=train_subset,
                weights=None,
                batch_size=min(batch_size, num_samples),
                num_workers=num_workers,
                seed=run_seed_base,
            )

            train_iter = iter(train_loader)

            algo.train()
            best_val_loss = float("inf")
            bad_checks = 0

            checkpoint_freq = max(20, min(100, math.ceil(len(train_subset) / batch_size)))
            if debug_mode:
                checkpoint_freq = 1
            max_steps = debug_max_steps if debug_mode else None

            # Use manual while loop with manual tqdm updates for proper nested display
            step_pbar = tqdm(
                itertools.count(),
                desc=f"  Training (run={run_idx+1}, size={num_samples})",
                unit="step",
                leave=True,
                disable=not debug_mode,
            )

            step = 0
            while True:
                try:
                    i, x, y, a, e, d = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    i, x, y, a, e, d = next(train_iter)

                x, y, a = x.to(device), y.to(device), a.to(device)
                algo.update((i, x, y, a), step)
                step_pbar.update(1)

                if (step + 1) % checkpoint_freq == 0:
                    algo.eval()
                    val_losses = []
                    val_correct = 0
                    val_total = 0
                    with torch.no_grad():
                        for i_val, x_val, y_val, a_val, e_val, d_val in val_loader:
                            x_val, y_val = x_val.to(device), y_val.to(device)
                            logits, vloss = algo.predict(x_val, y_val, return_loss=True)
                            val_losses.append(vloss.item() if isinstance(vloss, torch.Tensor) else vloss)
                            y_hat = logits.argmax(dim=-1) if logits.ndim > 1 else (logits > 0.5).long()
                            val_correct += (y_hat == y_val).sum().item()
                            val_total += x_val.size(0)

                    mean_val_loss = np.mean(val_losses)
                    mean_val_acc = (val_correct / val_total) if val_total > 0 else 0.0

                    if (best_val_loss - mean_val_loss) > es_min_delta:
                        best_val_loss = mean_val_loss
                        bad_checks = 0
                    else:
                        bad_checks += 1

                    step_pbar.set_postfix(
                        {
                            "val": f"{mean_val_loss:.4f}",
                            "val_acc": f"{mean_val_acc:.4f}",
                            "best": f"{best_val_loss:.4f}",
                            "patience": f"{bad_checks}/{es_patience}",
                        },
                        refresh=True,
                    )
                    if bad_checks >= es_patience and not debug_mode:
                        step_pbar.close()
                        break

                    algo.train()

                if debug_mode and max_steps is not None and (step + 1) >= max_steps:
                    step_pbar.close()
                    break

                step += 1

            # Ensure progress bar is closed after training loop completes
            if not step_pbar.disable:
                step_pbar.close()

            algo.eval()
            val_loss, val_acc = get_mean_log_loss_and_accuracy(algo, val_loader, device)
            test_loss, test_acc = get_mean_log_loss_and_accuracy(algo, test_loader, device)
            job_logger(f"  val log-loss: {val_loss:.4f} nats, acc={val_acc:.4f}")
            job_logger(f"  test log-loss: {test_loss:.4f} nats, acc={test_acc:.4f}")

            # Compute train loss and original loss for every dataset size
            train_eval_loader = DataLoader(
                train_subset,
                batch_size=min(batch_size, num_samples),
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
            train_loss, train_acc = get_mean_log_loss_and_accuracy(algo, train_eval_loader, device)
            original_loss, original_acc = get_mean_log_loss_and_accuracy(algo, original_loader, device)
            job_logger(f"  train log-loss: {train_loss:.4f} nats, acc={train_acc:.4f}")
            job_logger(f"  original log-loss: {original_loss:.4f} nats, acc={original_acc:.4f}")

            # Evaluate on extra held-out datasets
            extra_metrics: Dict[str, tuple] = {}
            for extra_name, extra_loader in extra_loaders.items():
                e_loss, e_acc = get_mean_log_loss_and_accuracy(algo, extra_loader, device)
                extra_metrics[extra_name] = (e_loss, e_acc)
                job_logger(f"  {extra_name}: loss={e_loss:.4f}, acc={e_acc:.4f}")

            # Store results for this (run_idx, size) pair
            run_result = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "original_loss": original_loss,
                "original_acc": original_acc,
            }
            for extra_name, (e_loss, e_acc) in extra_metrics.items():
                run_result[f"{extra_name}_loss"] = e_loss
                run_result[f"{extra_name}_acc"] = e_acc

            # Permutation tests (if configured)
            if permutation_test_config is not None:
                perm_ds = permutation_test_config["dataset"]
                perm_attrs = permutation_test_config["attributes"]
                perm_n = permutation_test_config.get("n_permutations", 100)
                perm_bs = permutation_test_config.get("batch_size", batch_size)
                perm_nw = permutation_test_config.get("num_workers", 0)
                for attr in perm_attrs:
                    p_val, orig_acc_perm, mean_shuf_acc = compute_permutation_pvalue(
                        algo,
                        perm_ds,
                        attr,
                        device,
                        n_permutations=perm_n,
                        batch_size=perm_bs,
                        num_workers=perm_nw,
                        base_seed=run_seed_base + 10000 + num_samples,
                    )
                    run_result[f"pvalue_{attr}"] = p_val
                    run_result[f"acc_drop_{attr}"] = orig_acc_perm - mean_shuf_acc
                    job_logger(f"  perm {attr}: p={p_val:.3f}, drop={orig_acc_perm - mean_shuf_acc:+.4f}")

            results_by_size[num_samples].append(run_result)

            del algo
            del train_loader
            del train_iter
            del train_subset

    # Aggregate results across runs for each dataset size
    for num_samples in candidate_sizes:
        run_metrics_list = results_by_size[num_samples]
        if not run_metrics_list:
            continue
        run_train_losses = [r["train_loss"] for r in run_metrics_list]
        run_train_accs = [r["train_acc"] for r in run_metrics_list]
        run_val_losses = [r["val_loss"] for r in run_metrics_list]
        run_val_accs = [r["val_acc"] for r in run_metrics_list]
        run_test_losses = [r["test_loss"] for r in run_metrics_list]
        run_test_accs = [r["test_acc"] for r in run_metrics_list]
        run_original_losses = [r["original_loss"] for r in run_metrics_list]
        run_original_accs = [r["original_acc"] for r in run_metrics_list]

        run_count = len(run_val_losses)

        mean_train_log_loss = float(np.nanmean(run_train_losses))
        std_train_log_loss = float(np.nanstd(run_train_losses, ddof=1)) if run_count > 1 else 0.0
        mean_train_log_acc = float(np.nanmean(run_train_accs))
        std_train_log_acc = float(np.nanstd(run_train_accs, ddof=1)) if run_count > 1 else 0.0

        mean_val_log_loss = float(np.mean(run_val_losses))
        std_val_log_loss = float(np.std(run_val_losses, ddof=1)) if run_count > 1 else 0.0
        mean_val_acc = float(np.mean(run_val_accs))
        std_val_acc = float(np.std(run_val_accs, ddof=1)) if run_count > 1 else 0.0

        mean_test_log_loss = float(np.mean(run_test_losses))
        std_test_log_loss = float(np.std(run_test_losses, ddof=1)) if run_count > 1 else 0.0
        mean_test_acc = float(np.mean(run_test_accs))
        std_test_acc = float(np.std(run_test_accs, ddof=1)) if run_count > 1 else 0.0

        mean_original_log_loss = float(np.mean(run_original_losses))
        std_original_log_loss = float(np.std(run_original_losses, ddof=1)) if run_count > 1 else 0.0
        mean_original_acc = float(np.mean(run_original_accs))
        std_original_acc = float(np.std(run_original_accs, ddof=1)) if run_count > 1 else 0.0

        pcl_results.append(
            {
                "dataset_size": num_samples,
                "bits_per_sample": mean_test_log_loss / math.log(2) if mean_test_log_loss > 0 else 0,
                "bits_per_sample_std": (
                    std_test_log_loss / math.log(2) if (run_count > 1 and std_test_log_loss > 0) else 0
                ),
                "mean_train_log_loss": mean_train_log_loss,
                "std_train_log_loss": std_train_log_loss,
                "mean_train_log_acc": mean_train_log_acc,
                "std_train_log_acc": std_train_log_acc,
                "mean_val_log_loss": mean_val_log_loss,
                "std_val_log_loss": std_val_log_loss,
                "mean_val_acc": mean_val_acc,
                "std_val_acc": std_val_acc,
                "mean_test_log_loss": mean_test_log_loss,
                "std_test_log_loss": std_test_log_loss,
                "mean_test_acc": mean_test_acc,
                "std_test_acc": std_test_acc,
                "mean_original_log_loss": mean_original_log_loss,
                "std_original_log_loss": std_original_log_loss,
                "mean_original_acc": mean_original_acc,
                "std_original_acc": std_original_acc,
                "num_runs": run_count,
            }
        )

        # Aggregate extra held-out evaluation metrics
        for extra_name in extra_eval_datasets:
            loss_key = f"{extra_name}_loss"
            acc_key = f"{extra_name}_acc"
            run_extra_losses = [r[loss_key] for r in run_metrics_list if loss_key in r]
            run_extra_accs = [r[acc_key] for r in run_metrics_list if acc_key in r]
            if run_extra_losses:
                ec = len(run_extra_losses)
                pcl_results[-1][f"mean_{extra_name}_log_loss"] = float(np.mean(run_extra_losses))
                pcl_results[-1][f"std_{extra_name}_log_loss"] = (
                    float(np.std(run_extra_losses, ddof=1)) if ec > 1 else 0.0
                )
                pcl_results[-1][f"mean_{extra_name}_acc"] = float(np.mean(run_extra_accs))
                pcl_results[-1][f"std_{extra_name}_acc"] = (
                    float(np.std(run_extra_accs, ddof=1)) if ec > 1 else 0.0
                )

        # Aggregate permutatioan test p-values
        if permutation_test_config is not None:
            for attr in permutation_test_config["attributes"]:
                pval_key = f"pvalue_{attr}"
                drop_key = f"acc_drop_{attr}"
                run_pvals = [
                    r[pval_key]
                    for r in run_metrics_list
                    if pval_key in r and not np.isnan(r.get(pval_key, np.nan))
                ]
                run_drops = [
                    r[drop_key]
                    for r in run_metrics_list
                    if drop_key in r and not np.isnan(r.get(drop_key, np.nan))
                ]
                if run_pvals:
                    pc = len(run_pvals)
                    pcl_results[-1][f"mean_pvalue_{attr}"] = float(np.mean(run_pvals))
                    pcl_results[-1][f"std_pvalue_{attr}"] = (
                        float(np.std(run_pvals, ddof=1)) if pc > 1 else 0.0
                    )
                    pcl_results[-1][f"mean_acc_drop_{attr}"] = float(np.mean(run_drops))
                    pcl_results[-1][f"std_acc_drop_{attr}"] = (
                        float(np.std(run_drops, ddof=1)) if pc > 1 else 0.0
                    )

    # ---- compute K(p) from PCL curve and log summary ----------------------
    _label = model_name if model_name else "Model"
    if pcl_results:
        sizes_arr = [r["dataset_size"] for r in pcl_results]
        x = np.array(sizes_arr, dtype=float)
        y = np.array([r["mean_test_log_loss"] for r in pcl_results])

        kp, conv_idx, auc, asym_test_loss, width = _compute_kp_with_convergence_cutoff(x, y)
        asym_original_loss = float(pcl_results[-1]["mean_original_log_loss"])
        asym_original_acc = float(pcl_results[-1]["mean_original_acc"])

        job_logger(f"[{_label}] K(p): {kp:.4f} nats ({kp / np.log(2):.4f} bits)")
        job_logger(f"[{_label}] Asymptotic test loss: {asym_test_loss:.4f} nats")
        job_logger(f"[{_label}] Asymptotic original loss (slope): {asym_original_loss:.4f} nats")
        job_logger(f"[{_label}] Asymptotic original acc: {asym_original_acc:.4f}")

    gc.collect()  # Single cleanup at the end
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return pd.DataFrame(pcl_results)


def _train_single_model_for_pcl(
    dataset,
    val_dataset,
    hparams,
    device,
    num_samples,
    *,
    base_seed=None,
    config: Optional[Dict] = None,
):
    dataset_size = len(dataset)
    num_samples = max(1, min(dataset_size, int(num_samples)))
    shuffle_seed = base_seed if base_seed is not None else _get_cfg(config, "seed", 0)

    run_seed_base = hash((shuffle_seed, num_samples)) % (2**31)
    torch.manual_seed(run_seed_base)
    np.random.seed(run_seed_base)
    random.seed(run_seed_base)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(run_seed_base)

    permuted_indices = np.random.permutation(dataset_size)
    subset_indices = permuted_indices[:num_samples]
    train_subset = Subset(dataset, subset_indices.tolist())
    attach_dataset_attributes(train_subset)

    learner = _get_cfg(config, "learner", None)
    if learner is None:
        raise ValueError("config['learner'] is required for _train_single_model_for_pcl.")

    AlgorithmClass = algorithms.get_algorithm_class(learner)
    algo = AlgorithmClass(
        dataset.data_type,
        dataset.INPUT_SHAPE,
        dataset.num_labels,
        dataset.num_attributes,
        len(train_subset),
        hparams,
    )
    algo = algo.to(device)

    batch_size = _get_cfg(config, "batch_size", 64)
    num_workers = _get_cfg(config, "num_workers", 0)
    debug_mode = _get_cfg(config, "debug_mode", False)
    es_min_delta = _get_cfg(config, "es_min_delta", 1e-4)
    es_patience = _get_cfg(config, "es_patience", 5)
    debug_max_steps = _get_cfg(config, "debug_max_steps", None)

    train_loader = InfiniteDataLoader(
        dataset=train_subset,
        weights=None,
        batch_size=min(batch_size, num_samples),
        num_workers=num_workers,
        seed=run_seed_base,
    )
    train_iter = iter(train_loader)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    algo.train()
    best_val_loss = float("inf")
    bad_checks = 0
    checkpoint_freq = max(20, min(100, math.ceil(len(train_subset) / batch_size)))
    if debug_mode:
        checkpoint_freq = 1
    max_steps = debug_max_steps if debug_mode else None

    step_pbar = tqdm(
        itertools.count(),
        desc=f"  Training (n={num_samples})",
        leave=True,
        unit="step",
        position=0,
        dynamic_ncols=True,
        disable=not debug_mode,
    )
    for step in step_pbar:
        try:
            i, x, y, a, e, d = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            i, x, y, a, e, d = next(train_iter)

        x, y, a = x.to(device), y.to(device), a.to(device)
        algo.update((i, x, y, a), step)

        if (step + 1) % checkpoint_freq == 0:
            algo.eval()
            val_losses = []
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for i_val, x_val, y_val, a_val, e_val, d_val in val_loader:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    logits, vloss = algo.predict(x_val, y_val, return_loss=True)
                    val_losses.append(vloss.item() if isinstance(vloss, torch.Tensor) else vloss)
                    y_hat = logits.argmax(dim=-1) if logits.ndim > 1 else (logits > 0.5).long()
                    val_correct += (y_hat == y_val).sum().item()
                    val_total += x_val.size(0)

            mean_val_loss = np.mean(val_losses)
            mean_val_acc = (val_correct / val_total) if val_total > 0 else 0.0
            if (best_val_loss - mean_val_loss) > es_min_delta:
                best_val_loss = mean_val_loss
                bad_checks = 0
            else:
                bad_checks += 1

            step_pbar.set_postfix(
                {
                    "val": f"{mean_val_loss:.4f}",
                    "val_acc": f"{mean_val_acc:.4f}",
                    "best": f"{best_val_loss:.4f}",
                    "es": f"{bad_checks}/{es_patience}",
                }
            )

            if bad_checks >= es_patience and not debug_mode:
                step_pbar.close()
                break

            algo.train()

        if debug_mode and max_steps is not None and (step + 1) >= max_steps:
            step_pbar.close()
            break

    return algo


def _train_separate_feature_extractors_for_pcl(
    dataset,
    val_dataset,
    hparams,
    device,
    num_samples,
    *,
    attr_spec: Sequence[str],
    base_seed=None,
    config: Optional[Dict] = None,
):
    """Train separate models for each feature in attr_spec.

    Returns a dict {attr_name: trained_model} where each model predicts p(attr|x).
    """
    attr_spec = [a.lower() for a in attr_spec]
    if not attr_spec:
        raise ValueError("attr_spec must be non-empty.")

    attr_models = {}

    for attr_idx, attr_name in enumerate(attr_spec):
        print(f"  Training feature extractor for '{attr_name}'...")

        # Create hparams for this specific attribute
        attr_hparams = hparams.copy()
        attr_hparams.update(
            {
                "task_input": "image",
                "task_output": attr_name,
            }
        )

        # Use a different seed for each attribute model
        attr_seed = (base_seed or 0) + attr_idx * 1000

        # Train using the single model training function with attribute-specific dataset
        # We need to create a wrapper that sets up the correct target
        model = _train_single_attr_model_for_pcl(
            dataset,
            val_dataset,
            hparams,
            device,
            num_samples,
            attr_name=attr_name,
            base_seed=attr_seed,
            config=config,
        )
        attr_models[attr_name] = model

    return attr_models


def _train_single_attr_model_for_pcl(
    dataset,
    val_dataset,
    hparams,
    device,
    num_samples,
    *,
    attr_name: str,
    base_seed=None,
    config: Optional[Dict] = None,
):
    """Train a single model to predict one attribute from images."""
    attr_name = attr_name.lower()

    dataset_size = len(dataset)
    num_samples = max(1, min(dataset_size, int(num_samples)))
    shuffle_seed = base_seed if base_seed is not None else _get_cfg(config, "seed", 0)

    run_seed_base = hash((shuffle_seed, num_samples)) % (2**31)
    torch.manual_seed(run_seed_base)
    np.random.seed(run_seed_base)
    random.seed(run_seed_base)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(run_seed_base)

    permuted_indices = np.random.permutation(dataset_size)
    subset_indices = permuted_indices[:num_samples]
    train_subset = Subset(dataset, subset_indices.tolist())
    attach_dataset_attributes(train_subset)

    base_ds = get_base_dataset(dataset)
    num_classes = int(base_ds.task_cardinalities[attr_name])

    # Build model: featurizer + classifier for this attribute
    featurizer = networks.Featurizer(dataset.data_type, dataset.INPUT_SHAPE, hparams)
    classifier = networks.Classifier(
        featurizer.n_outputs, num_classes, hparams.get("nonlinear_classifier", False)
    )
    model = nn.Sequential(featurizer, classifier).to(device)

    optimizer = get_optimizers["sgd"](model, hparams["lr"], hparams["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()

    batch_size = _get_cfg(config, "batch_size", 64)
    num_workers = _get_cfg(config, "num_workers", 0)
    debug_mode = _get_cfg(config, "debug_mode", False)
    es_min_delta = _get_cfg(config, "es_min_delta", 1e-4)
    es_patience = _get_cfg(config, "es_patience", 5)
    debug_max_steps = _get_cfg(config, "debug_max_steps", None)

    train_loader = InfiniteDataLoader(
        dataset=train_subset,
        weights=None,
        batch_size=min(batch_size, num_samples),
        num_workers=num_workers,
        seed=run_seed_base,
    )
    train_iter = iter(train_loader)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_base_ds = get_base_dataset(val_dataset)

    model.train()
    best_val_loss = float("inf")
    bad_checks = 0
    checkpoint_freq = max(20, min(100, math.ceil(len(train_subset) / batch_size)))
    if debug_mode:
        checkpoint_freq = 1
    max_steps = debug_max_steps if debug_mode else None

    step_pbar = tqdm(
        itertools.count(),
        desc=f"    Training '{attr_name}' (n={num_samples})",
        leave=True,
        unit="step",
        disable=not debug_mode,
    )

    for step in step_pbar:
        try:
            i, x, y, a, e, d = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            i, x, y, a, e, d = next(train_iter)

        x = x.to(device)
        # Resolve target for this attribute
        idx_cpu = i.detach().cpu().long()
        if attr_name == "color":
            target = a.to(device)
        elif attr_name == "environment":
            target = e.to(device)
        elif attr_name == "digit":
            if hasattr(base_ds, "digit_id_tensor"):
                target = base_ds.digit_id_tensor[idx_cpu].to(device)
            else:
                target = d.to(device)
        else:
            raise ValueError(f"Unsupported CMNIST attribute: {attr_name}")

        logits = model(x)
        loss = loss_fn(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % checkpoint_freq == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for i_val, x_val, y_val, a_val, e_val, d_val in val_loader:
                    x_val = x_val.to(device)
                    # Resolve target for this attribute
                    idx_cpu_val = i_val.detach().cpu().long()
                    if attr_name == "color":
                        val_target = a_val.to(device)
                    elif attr_name == "environment":
                        val_target = e_val.to(device)
                    elif attr_name == "digit":
                        if hasattr(val_base_ds, "digit_id_tensor"):
                            val_target = val_base_ds.digit_id_tensor[idx_cpu_val].to(device)
                        else:
                            val_target = d_val.to(device)
                    else:
                        raise ValueError(f"Unsupported CMNIST attribute: {attr_name}")
                    val_logits = model(x_val)
                    val_losses.append(loss_fn(val_logits, val_target).item())

            mean_val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
            if (best_val_loss - mean_val_loss) > es_min_delta:
                best_val_loss = mean_val_loss
                bad_checks = 0
            else:
                bad_checks += 1

            step_pbar.set_postfix(
                {
                    "val": f"{mean_val_loss:.4f}",
                    "best": f"{best_val_loss:.4f}",
                    "es": f"{bad_checks}/{es_patience}",
                }
            )

            if bad_checks >= es_patience and not debug_mode:
                step_pbar.close()
                break
            model.train()

        if debug_mode and max_steps is not None and (step + 1) >= max_steps:
            step_pbar.close()
            break

    model.eval()

    # Wrap in a simple object with predict method for compatibility
    class AttrModelWrapper:
        def __init__(self, model):
            self._model = model

        def predict(self, x):
            return self._model(x)

        def eval(self):
            self._model.eval()

        def train(self, mode=True):
            self._model.train(mode)

    return AttrModelWrapper(model)


def _encode_attribute_combo(values, bases):
    encoded = 0
    mult = 1
    for val, base in zip(values, bases):
        encoded += int(val) * mult
        mult *= int(base)
    return encoded


def _decode_attribute_combo_indices(encoded: torch.Tensor, bases: Sequence[int]) -> torch.Tensor:
    """Decode encoded combo indices into per-field values.

    Encoding scheme matches `_encode_attribute_combo`:
        encoded = v0 + base0 * (v1 + base1 * (v2 + ...))

    Args:
        encoded: (N,) long tensor of encoded indices.
        bases: Per-field cardinalities.

    Returns:
        (N, K) long tensor of decoded values, where K=len(bases).
    """

    if encoded.ndim != 1:
        raise ValueError(f"encoded must be 1D, got shape={tuple(encoded.shape)}")
    if not bases:
        raise ValueError("bases must be non-empty")

    out = []
    rem = encoded
    for base in bases:
        base_t = int(base)
        out.append(rem % base_t)
        rem = rem // base_t
    return torch.stack(out, dim=1)


def sample_cmnist_tabular_inputs_from_attr_model(
    attr_model,
    dataset,
    *,
    attr_spec: Sequence[str],
    bases: Sequence[int],
    device,
    seed: int = 0,
    batch_size: int = 512,
    mode: str = "sample",
):
    """Sample CMNIST tabular inputs from p(attr_spec|x) predicted by `attr_model`.

    This is used to build a *model-implied* tabular input for subtask-2
    (predict label given features) without feeding ground-truth
    ['environment','color','digit'].

    Args:
        attr_model: Trained model for subtask-1 (image -> attribute-combo logits).
        dataset: CMNIST or a Subset wrapping a CMNIST.
        attr_spec: Ordered fields, e.g. ['environment','color','digit'].
        bases: Cardinalities in the same order as attr_spec.
        device: Torch device.
        seed: RNG seed for sampling.
        batch_size: Batch size for attribute inference.
        mode: 'sample' (categorical sample) or 'argmax'.

    Returns:
        Dict mapping each field name to a 1D long tensor of sampled values aligned
        with the *base dataset indices* provided by `indices` (caller decides how
        to apply them).
    """

    attr_spec = [a.lower() for a in attr_spec]
    if len(attr_spec) == 0:
        raise ValueError("attr_spec must be non-empty")
    if len(bases) != len(attr_spec):
        raise ValueError(f"bases length ({len(bases)}) must match attr_spec length ({len(attr_spec)})")

    base_ds = get_base_dataset(dataset)
    missing = [name for name in attr_spec if name not in getattr(base_ds, "task_cardinalities", {})]
    if missing:
        raise ValueError(f"attr_spec contains unknown fields for this dataset: {missing}")
    if not hasattr(base_ds, "precomputed_imgs"):
        raise ValueError("CMNIST base dataset must expose precomputed_imgs")

    # Resolve which base indices we are sampling for.
    if isinstance(dataset, Subset):
        base_indices = torch.as_tensor(dataset.indices, dtype=torch.long)
        parent = dataset.dataset
        while isinstance(parent, Subset):
            parent_indices = torch.as_tensor(parent.indices, dtype=torch.long)
            base_indices = parent_indices[base_indices]
            parent = parent.dataset
    else:
        base_indices = torch.arange(len(base_ds), dtype=torch.long)

    if base_indices.numel() == 0:
        empty = torch.empty((0,), dtype=torch.long)
        return {"indices": base_indices, **{name: empty for name in attr_spec}}

    # Deterministic generator for reproducibility (match sampling device).
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    total = base_indices.numel()
    x_all = base_ds.precomputed_imgs

    # Handle dict of separate models
    is_separate_models = isinstance(attr_model, dict)
    if is_separate_models:
        for m in attr_model.values():
            m.eval()
        sampled_by_attr = {name: [] for name in attr_spec}
        gens = {
            name: torch.Generator(device=device).manual_seed(int(seed) + idx * 1000)
            for idx, name in enumerate(attr_spec)
        }
        with torch.no_grad():
            for start in range(0, total, batch_size):
                idx = base_indices[start : start + batch_size]
                x = x_all[idx].to(device)
                for name in attr_spec:
                    logits = attr_model[name].predict(x)
                    probs = torch.softmax(logits, dim=-1)
                    if mode == "argmax":
                        enc = probs.argmax(dim=1)
                    elif mode == "sample":
                        enc = torch.multinomial(probs, num_samples=1, generator=gens[name]).squeeze(1)
                    else:
                        raise ValueError("mode must be 'sample' or 'argmax'")
                    sampled_by_attr[name].append(enc.detach().cpu())

        out = {"indices": base_indices}
        for name in attr_spec:
            out[name] = torch.cat(sampled_by_attr[name], dim=0).to(dtype=torch.long)
        return out

    attr_model.eval()

    sampled_encoded = []
    with torch.no_grad():
        for start in range(0, total, batch_size):
            idx = base_indices[start : start + batch_size]
            x = x_all[idx].to(device)
            logits = attr_model.predict(x)
            probs = torch.softmax(logits, dim=-1)

            if mode == "argmax":
                enc = probs.argmax(dim=1)
            elif mode == "sample":
                enc = torch.multinomial(probs, num_samples=1, generator=gen).squeeze(1)
            else:
                raise ValueError("mode must be 'sample' or 'argmax'")

            sampled_encoded.append(enc.detach().cpu())

    sampled_encoded = torch.cat(sampled_encoded, dim=0).to(dtype=torch.long)
    decoded = _decode_attribute_combo_indices(sampled_encoded, bases)
    out = {"indices": base_indices}
    for i, name in enumerate(attr_spec):
        out[name] = decoded[:, i]
    return out


def apply_sampled_cmnist_inputs_inplace_(
    dataset,
    *,
    sampled: Dict[str, torch.Tensor],
) -> None:
    """Apply sampled CMNIST inputs to a dataset in-place.

    Works on a CMNIST instance or a Subset wrapping CMNIST by applying the
    changes to the base dataset.
    """

    base_ds = get_base_dataset(dataset)
    if not hasattr(base_ds, "override_tabular_inputs_"):
        raise ValueError("Base dataset does not support override_tabular_inputs_().")

    base_ds.override_tabular_inputs_(
        environment=sampled.get("environment"),
        color=sampled.get("color"),
        digit=sampled.get("digit"),
        indices=sampled.get("indices"),
    )


def _compute_kp_with_convergence_cutoff(
    x: np.ndarray,
    y: np.ndarray,
    abs_delta_threshold: float = 5e-5,
    min_points: int = 5,
) -> tuple:
    """Compute K(p) from a PCL curve, truncating at the convergence point.

    When a sub-task converges quickly (within a few hundred samples), the tail
    of the PCL curve is dominated by noise, which can make the trapezoidal
    integral dip *below* the asymptotic rectangle and yield a **negative**
    K(p).  This helper detects the convergence point and only integrates up to
    there, avoiding the accumulation of tail noise.

    Convergence is detected as the first index *i* (with ``i >= min_points-1``)
    where the absolute change in test loss between consecutive dataset sizes
    ``|y[i] - y[i-1]|`` drops below ``abs_delta_threshold``.

    Parameters
    ----------
    x : array-like
        Dataset sizes (sorted ascending).
    y : array-like
        Test losses at each dataset size.
    abs_delta_threshold : float
        Absolute variation threshold.  When ``|y[i] - y[i-1]|`` is smaller
        than this value the curve is considered converged at index *i*.
    min_points : int
        Minimum number of points to include in the integration so that we
        never integrate over a degenerate interval.

    Returns
    -------
    kp : float
        Estimated K(p) in the same units as *y* (non-negative).
    conv_idx : int
        Index in *x* / *y* at which convergence was detected.
    area_under_curve : float
        Total area under the truncated curve.
    asymptotic_loss : float
        Loss at the convergence point (used as the rectangle height).
    width : float
        Width of the integration interval (x_trunc[-1] - x_trunc[0]).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2:
        return 0.0, 0, 0.0, float(y[-1]) if len(y) else np.nan, 0.0

    # Detect convergence: first index (after min_points) where the absolute
    # change between consecutive points drops below the threshold.
    conv_idx = len(y) - 1  # default: use all points
    for i in range(max(1, min_points - 1), len(y)):
        if abs(y[i] - y[i - 1]) < abs_delta_threshold:
            conv_idx = i
            break

    # Ensure at least min_points
    conv_idx = max(conv_idx, min(min_points - 1, len(y) - 1))

    # Integrate only up to convergence point
    x_trunc = x[: conv_idx + 1]
    y_trunc = y[: conv_idx + 1]

    if len(x_trunc) < 2:
        return 0.0, conv_idx, 0.0, float(y_trunc[-1]), 0.0

    area_under_curve = float(np.trapz(y_trunc, x_trunc))
    width = float(x_trunc[-1] - x_trunc[0])
    asymptotic_loss = float(y_trunc[-1])
    kp = area_under_curve - asymptotic_loss * width

    print("Convergence detected at index:", conv_idx, "dataset size:", x[conv_idx])
    return max(float(kp), 0.0), int(conv_idx), area_under_curve, asymptotic_loss, width


def _eval_model_cross_entropy(model, dataset, device, batch_size=64):
    """Evaluate a model's mean cross-entropy loss on a dataset.

    Expects ``model.predict(x)`` to return logits and the dataset
    to yield ``(i, x, y, a, e, d)`` tuples where ``y`` is the target.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_n = 0
    model.eval()
    with torch.no_grad():
        for _i, x, y, _a, _e, _d in loader:
            x, y = x.to(device), y.to(device)
            logits = model.predict(x)
            total_loss += loss_fn(logits, y).item()
            total_n += x.size(0)
    return total_loss / total_n if total_n > 0 else 0.0


def compute_bayes_optimal_pcl_curve(
    DatasetClass,
    data_path: str,
    hparams: dict,
    device,
    dataset_sizes: Iterable[int],
    attr_spec: Sequence[str],
    *,
    train_eval_dataset,
    test_eval_dataset,
    base_seed: int = 0,
    config: Optional[Dict] = None,
    job_logger: Optional[Callable[[str], None]] = None,
    extra_eval_datasets: Optional[Dict[str, Any]] = None,
    permutation_test_config: Optional[Dict] = None,
) -> pd.DataFrame:
    """Compute Bayes-Optimal PCL curve with **per-size** marginal log-likelihood.

    At each candidate dataset size *n* this function:

    1. Trains separate attribute extractors (one per attribute in *attr_spec*)
       on *n* training images.
    2. Evaluates each extractor on the held-out attribute test set to obtain the
       sub-task test loss at size *n*.
    3. Samples predicted features from the extractors, overrides the tabular
       label-predictor datasets, and trains a label predictor p(y|features) on
       *n* tabular samples.
    4. Evaluates the label predictor on the (true-feature) label test set to
       obtain the label sub-task test loss at size *n*.
    5. Computes the composite Bayes-optimal marginal
       ``p(y|x) = Σ_z p(y|z) Π_i p(z_i|x)``
       on *test_eval_dataset* and *train_eval_dataset* via
       :func:`compute_bayes_marginal_loglikelihood`.

    Finally, sub-task K(p) values are estimated from the per-size sub-task
    losses using the trapezoidal rule.

    Returns
    -------
    pd.DataFrame
        One row per candidate dataset size with columns for marginal
        loss / accuracy on test and train, sub-task K(p), etc.
    """
    attr_spec = [a.lower() for a in attr_spec]
    if extra_eval_datasets is None:
        extra_eval_datasets = {}
    if job_logger is None:
        job_logger = print

    # ---- create datasets ------------------------------------------------
    bayes_train = DatasetClass(data_path, "tr", hparams)
    bayes_val = DatasetClass(data_path, "va", hparams)
    base_ds = get_base_dataset(bayes_train)
    attr_bases = [base_ds.task_cardinalities[a] for a in attr_spec]

    # Attribute-specific test datasets (y = attribute target)
    attr_test_ds: Dict[str, object] = {}
    for attr_name in attr_spec:
        ah = hparams.copy()
        ah.update({"task_input": "image", "task_output": attr_name})
        attr_test_ds[attr_name] = DatasetClass(data_path, "te", ah)

    # Label predictor datasets (tabular input)
    label_hparams = hparams.copy()
    label_hparams.update({"task_input": attr_spec, "task_output": "label"})
    label_train = DatasetClass(data_path, "tr", label_hparams)
    label_val = DatasetClass(data_path, "va", label_hparams)
    label_test = DatasetClass(data_path, "te", label_hparams)

    # ---- resolve candidate sizes ----------------------------------------
    dataset_size = len(bayes_train)
    candidate_sizes = sorted({max(1, min(dataset_size, int(s))) for s in dataset_sizes if int(s) > 0})
    if not candidate_sizes:
        raise ValueError("No valid dataset sizes.")

    sampling_seed = base_seed + 777
    batch_size = _get_cfg(config, "batch_size", 64)
    small_data_threshold = _get_cfg(config, "small_data_threshold", 1000)
    num_runs_small = _get_cfg(config, "num_runs_small", 5)
    num_runs_large = _get_cfg(config, "num_runs_large", 5)
    num_runs_max_size = _get_cfg(config, "num_runs_max_size", 5)
    _max_size = candidate_sizes[-1]
    max_run_count = max(num_runs_small, num_runs_large, num_runs_max_size)

    # ---- train full-data attr_models for sampling ------------------------
    # Train attr_models on the largest dataset size so that feature sampling
    # is accurate even when the label_model is trained on small subsets.
    max_ns = candidate_sizes[-1]
    job_logger(f"[Bayes] Training full-data attr_models on {max_ns} samples for feature sampling...")
    full_attr_models = _train_separate_feature_extractors_for_pcl(
        bayes_train,
        bayes_val,
        hparams,
        device,
        max_ns,
        attr_spec=attr_spec,
        base_seed=base_seed + 999,
        config=config,
    )

    # ---- multi-run, per-size loop ----------------------------------------
    # Collect per-(run, size) metrics, then aggregate mean/std like compute_pcl_curve.
    results_by_size: Dict[int, list] = {ns: [] for ns in candidate_sizes}
    # Subtask losses: per-(run, size) for K(p) computation.
    # Structure: subtask_losses_by_run[run_idx][name] = [loss_at_size_0, ...]
    subtask_losses_by_run: list = []

    for run_idx in range(max_run_count):
        run_seed = base_seed + run_idx * 1000
        run_sampling_seed = sampling_seed + run_idx * 100

        # Per-run subtask loss tracking
        run_subtask: Dict[str, list] = {name: [] for name in attr_spec}
        run_subtask["_label"] = []

        for si, ns in enumerate(candidate_sizes):
            if ns == _max_size and num_runs_max_size is not None:
                run_count = num_runs_max_size
            elif ns <= small_data_threshold:
                run_count = num_runs_small
            else:
                run_count = num_runs_large
            if run_idx >= run_count:
                # Pad with NaN so array lengths stay consistent
                for name in attr_spec:
                    run_subtask[name].append(np.nan)
                run_subtask["_label"].append(np.nan)
                continue

            job_logger(f"[Bayes] Run {run_idx + 1}/{run_count}, Size {ns} ({si + 1}/{len(candidate_sizes)})")

            # 1. Train attribute extractors at size ns (different seed per run)
            attr_models = _train_separate_feature_extractors_for_pcl(
                bayes_train,
                bayes_val,
                hparams,
                device,
                ns,
                attr_spec=attr_spec,
                base_seed=run_seed + 999,
                config=config,
            )

            # 2. Sub-task attribute test losses
            for attr_name in attr_spec:
                loss = _eval_model_cross_entropy(
                    attr_models[attr_name], attr_test_ds[attr_name], device, batch_size
                )
                run_subtask[attr_name].append(loss)

            # 3. Sample features using full-data attr_models (not per-size ones)
            sampled_tr = sample_cmnist_tabular_inputs_from_attr_model(
                full_attr_models,
                label_train,
                attr_spec=attr_spec,
                bases=attr_bases,
                device=device,
                seed=run_sampling_seed + 1,
                mode="sample",
            )
            apply_sampled_cmnist_inputs_inplace_(label_train, sampled=sampled_tr)
            sampled_va = sample_cmnist_tabular_inputs_from_attr_model(
                full_attr_models,
                label_val,
                attr_spec=attr_spec,
                bases=attr_bases,
                device=device,
                seed=run_sampling_seed + 2,
                mode="sample",
            )
            apply_sampled_cmnist_inputs_inplace_(label_val, sampled=sampled_va)

            # Also sample features for label_test to match training distribution
            sampled_te = sample_cmnist_tabular_inputs_from_attr_model(
                full_attr_models,
                label_test,
                attr_spec=attr_spec,
                bases=attr_bases,
                device=device,
                seed=run_sampling_seed + 3,
                mode="sample",
            )
            apply_sampled_cmnist_inputs_inplace_(label_test, sampled=sampled_te)

            # 4. Train label predictor at size ns
            label_model = _train_single_model_for_pcl(
                label_train,
                label_val,
                label_hparams,
                device,
                ns,
                base_seed=run_seed + 500,
                config=config,
            )

            # 5. Sub-task label test loss (evaluated on sampled features to match training)
            label_test_loader = DataLoader(label_test, batch_size=batch_size, shuffle=False, num_workers=0)
            lbl_loss, _ = get_mean_log_loss_and_accuracy(label_model, label_test_loader, device)
            run_subtask["_label"].append(lbl_loss)

            # 6. Composite Bayes marginal on (original) test and train
            test_ll, test_acc = compute_bayes_marginal_loglikelihood(
                attr_models, label_model, test_eval_dataset, attr_spec, device
            )
            train_ll, train_acc = compute_bayes_marginal_loglikelihood(
                attr_models, label_model, train_eval_dataset, attr_spec, device
            )

            # 7. Composite Bayes marginal on extra held-out datasets
            extra_metrics: Dict[str, tuple] = {}
            if extra_eval_datasets:
                for extra_name, extra_ds in extra_eval_datasets.items():
                    e_ll, e_acc = compute_bayes_marginal_loglikelihood(
                        attr_models, label_model, extra_ds, attr_spec, device
                    )
                    extra_metrics[extra_name] = (e_ll, e_acc)
                    job_logger(f"  Bayes {extra_name}: loss={e_ll:.4f}, acc={e_acc:.4f}")

            # 8. Permutation tests on Bayes wrapper (if configured)
            perm_metrics: Dict[str, float] = {}
            if permutation_test_config is not None:
                perm_ds = permutation_test_config["dataset"]
                perm_attrs = permutation_test_config["attributes"]
                perm_n = permutation_test_config.get("n_permutations", 100)
                perm_bs = permutation_test_config.get("batch_size", batch_size)
                perm_nw = permutation_test_config.get("num_workers", 0)
                bayes_wrapper = BayesModelWrapper(
                    attr_models, label_model, attr_spec, test_eval_dataset, device
                )
                for attr in perm_attrs:
                    p_val, orig_acc_perm, mean_shuf_acc = compute_permutation_pvalue(
                        bayes_wrapper,
                        perm_ds,
                        attr,
                        device,
                        n_permutations=perm_n,
                        batch_size=perm_bs,
                        num_workers=perm_nw,
                        base_seed=run_seed + 10000 + ns,
                    )
                    perm_metrics[f"pvalue_{attr}"] = p_val
                    perm_metrics[f"acc_drop_{attr}"] = orig_acc_perm - mean_shuf_acc
                    job_logger(
                        f"  Bayes perm {attr}: p={p_val:.3f}, drop={orig_acc_perm - mean_shuf_acc:+.4f}"
                    )

            # Store run result
            run_result = {
                "test_loss": test_ll,
                "test_acc": test_acc,
                "train_loss": train_ll,
                "train_acc": train_acc,
                "original_loss": test_ll,
                "original_acc": test_acc,
            }
            for extra_name, (e_loss, e_acc) in extra_metrics.items():
                run_result[f"{extra_name}_loss"] = e_loss
                run_result[f"{extra_name}_acc"] = e_acc
            run_result.update(perm_metrics)
            results_by_size[ns].append(run_result)

            job_logger(
                f"  Bayes marginal: test_loss={test_ll:.4f}, acc={test_acc:.4f}, "
                f"train_loss={train_ll:.4f}"
            )

            del attr_models, label_model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        subtask_losses_by_run.append(run_subtask)

    # Clean up full-data attr_models
    del full_attr_models
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- aggregate results across runs -----------------------------------
    results = []
    for ns in candidate_sizes:
        run_metrics_list = results_by_size[ns]
        if not run_metrics_list:
            continue
        run_count = len(run_metrics_list)

        run_test_losses = [r["test_loss"] for r in run_metrics_list]
        run_test_accs = [r["test_acc"] for r in run_metrics_list]
        run_train_losses = [r["train_loss"] for r in run_metrics_list]
        run_train_accs = [r["train_acc"] for r in run_metrics_list]
        run_original_losses = [r["original_loss"] for r in run_metrics_list]
        run_original_accs = [r["original_acc"] for r in run_metrics_list]

        row = {
            "dataset_size": ns,
            "num_runs": run_count,
            "mean_test_log_loss": float(np.mean(run_test_losses)),
            "std_test_log_loss": float(np.std(run_test_losses, ddof=1)) if run_count > 1 else 0.0,
            "mean_test_acc": float(np.mean(run_test_accs)),
            "std_test_acc": float(np.std(run_test_accs, ddof=1)) if run_count > 1 else 0.0,
            "mean_train_log_loss": float(np.mean(run_train_losses)),
            "std_train_log_loss": float(np.std(run_train_losses, ddof=1)) if run_count > 1 else 0.0,
            "mean_train_acc": float(np.mean(run_train_accs)),
            "std_train_acc": float(np.std(run_train_accs, ddof=1)) if run_count > 1 else 0.0,
            "mean_original_log_loss": float(np.mean(run_original_losses)),
            "std_original_log_loss": float(np.std(run_original_losses, ddof=1)) if run_count > 1 else 0.0,
            "mean_original_acc": float(np.mean(run_original_accs)),
            "std_original_acc": float(np.std(run_original_accs, ddof=1)) if run_count > 1 else 0.0,
        }

        # Aggregate extra held-out evaluation metrics
        for extra_name in extra_eval_datasets:
            loss_key = f"{extra_name}_loss"
            acc_key = f"{extra_name}_acc"
            run_extra_losses = [r[loss_key] for r in run_metrics_list if loss_key in r]
            run_extra_accs = [r[acc_key] for r in run_metrics_list if acc_key in r]
            if run_extra_losses:
                ec = len(run_extra_losses)
                row[f"mean_{extra_name}_log_loss"] = float(np.mean(run_extra_losses))
                row[f"std_{extra_name}_log_loss"] = float(np.std(run_extra_losses, ddof=1)) if ec > 1 else 0.0
                row[f"mean_{extra_name}_acc"] = float(np.mean(run_extra_accs))
                row[f"std_{extra_name}_acc"] = float(np.std(run_extra_accs, ddof=1)) if ec > 1 else 0.0

        # Aggregate permutation test p-values
        if permutation_test_config is not None:
            for attr in permutation_test_config["attributes"]:
                pval_key = f"pvalue_{attr}"
                drop_key = f"acc_drop_{attr}"
                run_pvals = [
                    r[pval_key]
                    for r in run_metrics_list
                    if pval_key in r and not np.isnan(r.get(pval_key, np.nan))
                ]
                run_drops = [
                    r[drop_key]
                    for r in run_metrics_list
                    if drop_key in r and not np.isnan(r.get(drop_key, np.nan))
                ]
                if run_pvals:
                    pc = len(run_pvals)
                    row[f"mean_pvalue_{attr}"] = float(np.mean(run_pvals))
                    row[f"std_pvalue_{attr}"] = float(np.std(run_pvals, ddof=1)) if pc > 1 else 0.0
                    row[f"mean_acc_drop_{attr}"] = float(np.mean(run_drops))
                    row[f"std_acc_drop_{attr}"] = float(np.std(run_drops, ddof=1)) if pc > 1 else 0.0

        results.append(row)

    # ---- compute K(p) from sub-task curves --------------------------------
    # Average subtask losses across runs before computing K(p).
    sizes_arr = candidate_sizes
    x = np.array(sizes_arr, dtype=float)
    subtask_losses: Dict[str, np.ndarray] = {name: [] for name in attr_spec}
    subtask_losses["_label"] = []

    for si, ns in enumerate(sizes_arr):
        for name in list(attr_spec) + ["_label"]:
            vals = [
                subtask_losses_by_run[r][name][si]
                for r in range(max_run_count)
                if not np.isnan(subtask_losses_by_run[r][name][si])
            ]
            subtask_losses[name].append(float(np.mean(vals)) if vals else np.nan)

    total_kp = 0.0
    subtask_kp_dict: Dict[str, float] = {}  # store per-subtask K(p)

    for name in attr_spec:
        y = np.array(subtask_losses[name])
        kp, cidx, auc, asym_loss, w = _compute_kp_with_convergence_cutoff(x, y)
        total_kp += kp
        subtask_kp_dict[name] = kp
        job_logger(
            f"  {name} K(p): {kp:.4f} nats ({kp / np.log(2):.4f} bits) "
            f"[converged at idx {cidx}, size {int(x[cidx])}]"
        )

    y_lbl = np.array(subtask_losses["_label"])
    kp_lbl, cidx_lbl, auc_lbl, asym_loss_lbl, w_lbl = _compute_kp_with_convergence_cutoff(x, y_lbl)
    total_kp += kp_lbl
    subtask_kp_dict["_label"] = kp_lbl
    job_logger(
        f"  Label K(p): {kp_lbl:.4f} nats ({kp_lbl / np.log(2):.4f} bits) "
        f"[converged at idx {cidx_lbl}, size {int(x[cidx_lbl])}]"
    )
    job_logger(f"  Total K(p): {total_kp:.4f} nats ({total_kp / np.log(2):.4f} bits)")
    job_logger("  Breakdown: " + ", ".join(f"{k}={v:.4f}" for k, v in subtask_kp_dict.items()))

    df = pd.DataFrame(results)
    df["model_type"] = "Bayes-Optimal predictor"
    df["k_p_nats"] = total_kp
    # Store per-subtask K(p) for downstream inspection
    for st_name, st_kp in subtask_kp_dict.items():
        col = f"k_p_nats_{st_name.strip('_')}"
        df[col] = st_kp

    # cleanup
    del bayes_train, bayes_val, label_train, label_val, label_test
    for _ds in attr_test_ds.values():
        del _ds
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return df


def compute_bayes_marginal_loglikelihood(
    attr_models, label_model, dataset, attr_spec, device, *, batch_size: int = 64
):
    """Compute marginal log-likelihood p(y|x) by marginalizing over latent features.

    Args:
        attr_models: A dict mapping attribute name -> individual model for that attribute.
        label_model: Model predicting p(y|features).
        dataset: Dataset to evaluate on.
        attr_spec: List of attribute names (e.g., ['environment', 'color', 'digit']).
        device: Torch device.
        batch_size: Batch size for evaluation.

    Returns:
        (mean_loss, accuracy): Mean negative log-likelihood and accuracy.
    """
    attr_spec = [a.lower() for a in attr_spec]
    base_dataset = get_base_dataset(dataset)
    bases = [base_dataset.task_cardinalities[a] for a in attr_spec]
    num_attr_classes = int(np.prod(bases))
    combos = list(product(*[range(b) for b in bases]))
    attr_inputs = torch.tensor(combos, dtype=torch.float32, device=device)

    label_model.eval()
    with torch.no_grad():
        label_logits = label_model.predict(attr_inputs)
        label_probs = torch.softmax(label_logits, dim=-1)
    num_label_classes = label_probs.shape[1]
    p_y_given_attr = torch.zeros(num_attr_classes, num_label_classes, device=device)
    for combo_idx, combo in enumerate(combos):
        encoded_idx = _encode_attribute_combo(combo, bases)
        p_y_given_attr[encoded_idx] = label_probs[combo_idx]

    # Handle dict of separate models
    if not isinstance(attr_models, dict):
        raise ValueError("attr_models must be a dict mapping attribute name -> model")
    for m in attr_models.values():
        m.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    x_all = base_dataset.precomputed_imgs
    y_all = base_dataset.y_tensor
    total_batches = math.ceil(len(base_dataset) / batch_size)
    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(base_dataset))
        x_batch = x_all[start:end].to(device)
        y_batch = y_all[start:end].to(device)

        with torch.no_grad():
            # Separate models: get p(attr_i|x) from each model
            probs_list = []
            for name in attr_spec:
                logits = attr_models[name].predict(x_batch)
                probs_list.append(torch.softmax(logits, dim=-1))

            # Compute joint p(attr_1, attr_2, ...|x) = prod_i p(attr_i|x)
            # Use little-endian ordering to match _encode_attribute_combo:
            # encoded = v0 + base0*v1 + base0*base1*v2 + ...
            # So the first attribute varies fastest in the flattened index.
            #
            # Build joint by having earlier attributes in inner (faster) dims.
            # Start with first attribute, then expand outer dims for later attrs.
            joint_probs = probs_list[0]  # [B, base0], first attr varies fastest
            for probs in probs_list[1:]:
                # probs: [B, base_i] for attribute i
                # joint_probs: [B, acc_size] where acc_size = base0 * base1 * ... * base_{i-1}
                # We want index = v0 + v1*base0 + ... + v_i * (base0*...*base_{i-1})
                # So new dimension (v_i) should vary slowest (outer dim after reshape)
                # [B, 1, acc] * [B, base_i, 1] = [B, base_i, acc] -> reshape [B, base_i * acc]
                joint_probs = (probs.unsqueeze(-1) * joint_probs.unsqueeze(1)).reshape(
                    joint_probs.shape[0], -1
                )
            attr_probs = joint_probs

            p_y_given_attr_batch = p_y_given_attr[:, y_batch].T
            p_y = (attr_probs * p_y_given_attr_batch).sum(dim=1)
            p_y = torch.clamp(p_y, min=1e-12)
            batch_loss = -torch.log(p_y).sum().item()
            total_loss += batch_loss
            total_samples += x_batch.size(0)
            p_y1 = (attr_probs * p_y_given_attr[:, 1].unsqueeze(0)).sum(dim=1)
            y_hat = (p_y1 >= 0.5).long()
            total_correct += (y_hat == y_batch).sum().item()

    mean_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return mean_loss, accuracy


class BayesModelWrapper:
    """Wraps attr_models + label_model into a single object compatible with
    evaluate_accuracy and compute_permutation_pvalue.

    The wrapper exposes ``.predict(x)`` that returns logits (log-probabilities)
    by marginalizing over latent attributes:
        p(y|x) = sum_z p(y|z) * prod_i p(z_i|x)

    It also exposes ``.eval()`` for compatibility.
    """

    def __init__(self, attr_models, label_model, attr_spec, dataset, device):
        self.attr_models = attr_models
        self.label_model = label_model
        self.attr_spec = [a.lower() for a in attr_spec]
        self.device = device
        base_dataset = get_base_dataset(dataset)
        self.bases = [base_dataset.task_cardinalities[a] for a in self.attr_spec]
        num_attr_classes = int(np.prod(self.bases))
        combos = list(product(*[range(b) for b in self.bases]))
        attr_inputs = torch.tensor(combos, dtype=torch.float32, device=device)
        self.label_model.eval()
        with torch.no_grad():
            label_logits = self.label_model.predict(attr_inputs)
            label_probs = torch.softmax(label_logits, dim=-1)
        num_label_classes = label_probs.shape[1]
        self.p_y_given_attr = torch.zeros(num_attr_classes, num_label_classes, device=device)
        for combo_idx, combo in enumerate(combos):
            encoded_idx = _encode_attribute_combo(combo, self.bases)
            self.p_y_given_attr[encoded_idx] = label_probs[combo_idx]

    def eval(self):
        for m in self.attr_models.values():
            m.eval()
        self.label_model.eval()
        return self

    def train(self, mode=True):
        return self

    def predict(self, x):
        """Return log-probabilities p(y|x) for each class."""
        with torch.no_grad():
            probs_list = []
            for name in self.attr_spec:
                logits = self.attr_models[name].predict(x)
                probs_list.append(torch.softmax(logits, dim=-1))
            joint_probs = probs_list[0]
            for probs in probs_list[1:]:
                joint_probs = (probs.unsqueeze(-1) * joint_probs.unsqueeze(1)).reshape(
                    joint_probs.shape[0], -1
                )
            # p(y=c|x) = sum_z p(y=c|z) * p(z|x) for each class c
            p_y = joint_probs @ self.p_y_given_attr  # [B, num_classes]
            p_y = torch.clamp(p_y, min=1e-12)
            return torch.log(p_y)


def evaluate_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            i, x, y, a, e, d = batch
            x, y = x.to(device), y.to(device)

            y_pred = model.predict(x)
            y_hat = y_pred.argmax(dim=-1) if y_pred.ndim > 1 else (y_pred > 0.5).long()

            correct += (y_hat == y).sum().item()
            total += y.size(0)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def compute_permutation_pvalue(
    model,
    base_dataset,
    attribute: str,
    device,
    *,
    n_permutations: int = 100,
    batch_size: int = 64,
    num_workers: int = 0,
    base_seed: int = 42,
):
    original_loader = DataLoader(base_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    original_acc = evaluate_accuracy(model, original_loader, device)

    count_better = 0
    shuffled_accs = []

    for perm_idx in range(n_permutations):
        shuffled_view = base_dataset.create_shuffled_view(attribute, seed=base_seed + perm_idx)
        shuffled_loader = DataLoader(
            shuffled_view, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        shuffled_acc = evaluate_accuracy(model, shuffled_loader, device)
        shuffled_accs.append(shuffled_acc)

        if shuffled_acc >= original_acc:
            count_better += 1

    p_value = count_better / n_permutations
    mean_shuffled_acc = np.mean(shuffled_accs)
    return p_value, original_acc, mean_shuffled_acc
