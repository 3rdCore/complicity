import hashlib
import os
import re
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import Subset


# =============================================================================
# Dataset Utility Functions
# =============================================================================


def get_base_dataset(ds):
    """Unwrap Subset to get the underlying base dataset."""
    while isinstance(ds, Subset):
        ds = ds.dataset
    return ds


class _SubsetAttributeProxy:
    """Proxy class to access base dataset attributes through a Subset."""

    def __init__(self, subset, attr_name):
        self.subset = subset
        self.attr_name = attr_name
        self._base_attr = getattr(subset.dataset, attr_name)

    def __getitem__(self, idx):
        base_index = self.subset.indices[idx] if hasattr(self.subset, "indices") else idx
        return self._base_attr[base_index]

    def __len__(self):
        return len(self.subset)


def attach_dataset_attributes(subset):
    """Attach key attributes from the underlying dataset to a Subset object."""
    if hasattr(subset, "dataset"):
        for attr in ["INPUT_SHAPE", "num_labels", "num_attributes", "data_type"]:
            if hasattr(subset.dataset, attr):
                setattr(subset, attr, getattr(subset.dataset, attr))
        for raw_attr in ["imgs"]:
            if hasattr(subset.dataset, raw_attr) and not hasattr(subset, raw_attr):
                setattr(subset, raw_attr, _SubsetAttributeProxy(subset, raw_attr))


def denormalize_cmnist(img_tensor):
    """Denormalize CMNIST image for visualization."""
    mean = torch.tensor([0.1307, 0.1307, 0.0]).view(3, 1, 1)
    std = torch.tensor([0.3081, 0.3081, 0.3081]).view(3, 1, 1)
    return img_tensor * std + mean


# =============================================================================
# Group / stratified-sampling helpers
# =============================================================================


def build_group_index(ds):
    """Build a mapping from (label, attribute) pairs to sample indices.

    Returns a dict ``{(y, a): [idx, ...]}`` for every group present in *ds*.
    """
    groups = defaultdict(list)
    ys = getattr(ds, "labels", None)
    attrs = getattr(ds, "attributes", None)
    if ys is not None and attrs is not None and len(ys) == len(ds) and len(attrs) == len(ds):
        for i in range(len(ds)):
            y_i = int(ys[i])
            a_i = int(attrs[i])
            groups[(y_i, a_i)].append(i)
    else:
        for i in range(len(ds)):
            try:
                _, _, y, a, _ = ds[i]
                groups[(int(y), int(a))].append(i)
            except Exception:
                pass
    return groups


def seed_hash(*args):
    """Derive an integer hash from all args, for use as a random seed."""
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)


def safe_float_env(var, default):
    """Safely get a float environment variable, with fallback to default."""
    val = os.environ.get(var, "")
    try:
        return float(val) if val else default
    except ValueError:
        print(f"There was an error loading environment value {var}")
        return default


def safe_int_env(var, default):
    """Safely get an integer environment variable, with fallback to default."""
    val = os.environ.get(var, "")
    try:
        return int(val) if val else default
    except ValueError:
        print(f"There was an error loading environment value {var}")
        return default


class InfiniteDataLoader:
    """DataLoader wrapper that loops indefinitely, yielding batches with random replacement.

    Enables step-based training loops instead of epoch-based ones: callers simply
    call ``next(iter(loader))`` for as many steps as needed without worrying about
    epoch boundaries. When the underlying iterator is exhausted it silently resets.

    Supports optional per-sample weights (via WeightedRandomSampler) and a seed
    for reproducible sampling.
    """

    def __init__(self, dataset, weights, batch_size, num_workers, seed=None):
        super().__init__()

        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)

        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(
                weights, replacement=True, num_samples=len(dataset), generator=generator
            )
        else:
            sampler = torch.utils.data.RandomSampler(dataset, replacement=True, generator=generator)

        batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=batch_size, drop_last=False)

        self.dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=num_workers, batch_sampler=batch_sampler, pin_memory=False
        )
        self.iterator = iter(self.dataloader)

    def __iter__(self):
        while True:
            try:
                batch = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.dataloader)
                continue  # retry loop instead of yielding here
            yield batch

    def __len__(self):
        return len(self.dataloader)


def find_timestamp_root(path):
    """Find the root timestamp folder (format: YYYYMMDD-HHMMSS) in the path."""
    path_parts = path.split(os.sep)
    timestamp_pattern = r"^\d{8}-\d{6}$"  # YYYYMMDD-HHMMSS format

    for i, part in enumerate(path_parts):
        if re.match(timestamp_pattern, part):
            root_path = os.sep.join(path_parts[: i + 1])
            return root_path

    return path
