from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
import torch
from torchvision import datasets
import numpy as np
from typing import Tuple

DATASETS = [
    "CMNIST",
]


def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError(f"Dataset not found: {dataset_name}")
    return globals()[dataset_name]


# DEFINE
CMNIST_TRAINING_DATASET_SIZES = 250000  # using EMNIST digits split


def _normalize_cmnist_task_spec(spec, *, default):
    if spec is None:
        spec = default
    if isinstance(spec, str):
        return [spec]
    if isinstance(spec, (list, tuple)):
        return list(spec)
    raise ValueError(f"Invalid CMNIST task spec: {spec}")


def _canonical_cmnist_field(name: str) -> str:
    name = name.lower()
    alias_map = {
        "label": "label",
        "y": "label",
        "color": "color",
        "attr": "color",
        "a": "color",
        "environment": "environment",
        "env": "environment",
        "e": "environment",
        "digit": "digit",
        "d": "digit",
        "image": "image",
        "x": "image",
    }
    if name not in alias_map:
        raise ValueError(
            f"Unknown CMNIST task field: {name}. Must be one of label, color, environment, digit, image."
        )
    return alias_map[name]


def _cmnist_get_component_tensor(ds, field: str):
    if field == "label":
        return ds.y_tensor
    if field == "color":
        return ds.a_tensor
    if field == "environment":
        return ds.env_tensor
    if field == "digit":
        return ds.digit_id_tensor
    raise ValueError(f"Unsupported CMNIST component: {field}")


def _cmnist_encode_fields(values, bases):
    encoded = 0
    mult = 1
    for val, base in zip(values, bases):
        encoded += int(val) * mult
        mult *= int(base)
    return encoded


class CMNIST(Dataset):
    N_STEPS = 5001
    CHECKPOINT_FREQ = 250
    data_type = "images"
    _EMNIST_CACHE = {}

    @classmethod
    def _load_emnist_cached(cls, root: Path):
        cache_key = str(Path(root).resolve())
        cached = cls._EMNIST_CACHE.get(cache_key)
        if cached is not None:
            return cached

        emnist_train = datasets.EMNIST(root, split="digits", train=True, download=True)
        emnist_test = datasets.EMNIST(root, split="digits", train=False, download=True)

        X_train, y_train = emnist_train.data, emnist_train.targets
        X_test, y_test = emnist_test.data, emnist_test.targets

        # Rotate/Flip logic
        X_train = torch.flip(torch.rot90(X_train, k=1, dims=(1, 2)), dims=[1])
        X_test = torch.flip(torch.rot90(X_test, k=1, dims=(1, 2)), dims=[1])

        X_all = torch.cat([X_train, X_test], dim=0)
        y_all = torch.cat([y_train, y_test], dim=0)

        cls._EMNIST_CACHE[cache_key] = (X_all, y_all)
        return X_all, y_all

    def __init__(
        self,
        data_path,
        split,
        hparams,
        train_attr="yes",
        subsample_type=None,
        duplicates=None,
        input_size=None,
        dataset_size=None,
        subset_indices=None,
        subset_seed=None,
    ):
        self.hparams = hparams

        # Determine input size (32x32 images)
        self.input_size = hparams.get("input_size", 32) if input_size is None else input_size

        # Watermark configuration
        self.has_watermark = hparams.get("has_watermark", False)
        self.watermark_bank_size = int(hparams.get("watermark_bank_size", 2))
        self.grayscale = hparams.get("grayscale", False)
        self.random_watermark = hparams.get("random_watermark", False)
        self.watermark_bit_count = int(hparams.get("watermark_bits", 32))
        if not (1 <= self.watermark_bit_count <= 32):
            raise ValueError(f"watermark_bits must be in [1, 32], got {self.watermark_bit_count}")

        # 1. Load EMNIST (cached, lightweight uint8)
        root = Path(data_path) / "emnist"
        X_all, y_all = self._load_emnist_cached(root)

        # Define split sizes
        n_train = CMNIST_TRAINING_DATASET_SIZES
        n_val = 10000
        n_test = 20000

        X, y = self._select_split(X_all, y_all, split, n_train, n_val, n_test)

        # 3. Filter digits
        digits_per_class = hparams.get("digits_per_class", 5)
        X, y = self._filter_digits(X, y, digits_per_class)

        # Optional subsampling (dataset size or explicit indices)
        subset_indices = self._resolve_subset_indices(
            len(X),
            dataset_size=dataset_size,
            subset_indices=subset_indices,
            subset_seed=subset_seed,
            hparams=hparams,
        )
        if subset_indices is not None:
            X, y = self._apply_subset_indices(X, y, subset_indices)

        rng = np.random.default_rng(666)

        self.original_digits = y.numpy()
        y_np = self.original_digits  # Reuse the numpy array
        N = len(y_np)

        # Step 1: Derive binary labels from digits, then flip a proportion directly
        # We flip the binary label itself
        self.binary_label = (y_np >= 5).astype(np.int64)
        flip_mask = rng.random(N) < hparams["flip_prob"]
        self.binary_label[flip_mask] = 1 - self.binary_label[flip_mask]

        # Step 2: Sample environment for each sample
        spur_prob = hparams["spur_prob"]
        self.environment = (rng.random(N) < spur_prob).astype(np.int64)

        # Step 3: Assign color based on environment and label
        # env0: y=0 → green (color=1), y=1 → red (color=0)  => color = 1 - y
        # env1: y=0 → red (color=0), y=1 → green (color=1)  => color = y
        base_color = np.where(
            self.environment == 0, 1 - self.binary_label, self.binary_label  # env0: y=0→green, y=1→red
        )  # env1: y=0→red, y=1→green

        self.color = base_color.astype(np.int64)

        # Step 3b: Watermark correlates with environment (applied in _setup_watermarks)

        uninformative_majority = hparams.get("uninformative_majority", False)
        if uninformative_majority:
            # Majority environment is env0 (when spur_prob < 0.5) or env1 (when spur_prob > 0.5)

            majority_env = 0 if spur_prob < 0.5 else 1
            majority_mask = self.environment == majority_env
            # Assign random colors to samples in the majority environment
            self.color[majority_mask] = rng.integers(0, 2, size=majority_mask.sum())

        mnist_imgs = X.numpy()  # (N, 28, 28) uint8 numpy

        if hparams.get("random_digit", False):
            # Shuffle images to decorrelate digit from label
            perm = rng.permutation(len(mnist_imgs))
            mnist_imgs = mnist_imgs[perm]
            self.original_digits = self.original_digits[perm]

        if hparams.get("noise_digit", False):  # Replace digit content with noise
            noise = rng.normal(loc=127.0, scale=50.0, size=mnist_imgs.shape)
            mnist_imgs = np.clip(noise, 0, 255).astype(np.uint8)

        # 4. Subsampling (Re-ordering/Slicing indices and arrays)
        self._temp_mnist_imgs = mnist_imgs
        if hparams["attr_prob"] != 0.5:
            self._apply_subsampling(hparams, rng)

        mnist_imgs = self._temp_mnist_imgs
        del self._temp_mnist_imgs

        N = len(mnist_imgs)

        # Resize MNIST from 28x28 to 32x32
        resized_mnist = self._resize_mnist(mnist_imgs)
        del mnist_imgs

        # Store grayscale images for later reconstruction with shuffled attributes
        self.grayscale_imgs = resized_mnist.clone()  # (N, 32, 32) float32 [0, 1]

        # Create 32x32 RGB images as torch tensors
        self.precomputed_imgs = self._build_precomputed_imgs(resized_mnist)
        del resized_mnist

        # Initialize watermark bits storage (will be populated if watermarks are used)
        self.watermark_bits = None

        # Handle watermarks
        if self.has_watermark:
            self._setup_watermarks(N, rng)

        # Handle random watermarks (applies random bits to rightmost column)
        if self.random_watermark:
            self._apply_random_watermarks(N, rng)

        # 7. Build normalization constants as tensors for efficient __getitem__
        self._init_normalization()

        # Pre-normalize all images for fastest __getitem__
        self.precomputed_imgs = (self.precomputed_imgs - self._norm_mean) / self._norm_std

        # Convert labels to torch tensors
        self.y_tensor = torch.from_numpy(self.binary_label.astype(np.int64))
        self.a_tensor = torch.from_numpy(self.color.astype(np.int64))
        self.env_tensor = torch.from_numpy(self.environment.astype(np.int64))
        self.digits_tensor = torch.from_numpy(self.original_digits.astype(np.int64))

        # Map digit labels to contiguous ids for task encodings
        unique_digits = np.unique(self.original_digits)
        digit_value_to_index = {int(val): idx for idx, val in enumerate(unique_digits)}
        digit_ids = np.vectorize(digit_value_to_index.get)(self.original_digits).astype(np.int64)
        self.digit_id_tensor = torch.from_numpy(digit_ids)

        # Misc
        self.idx = list(range(N))
        self.digits = self.original_digits
        self.y = self.binary_label
        self.a = self.color
        self.env = self.environment
        self.num_labels = 2
        self.num_attributes = 2

        # Configure task-specific IO mapping
        self._configure_task_specs(hparams, digit_ids)

        if duplicates is not None:
            # Implement duplicate logic by repeating indices
            new_idx = []
            for i, dup in zip(self.idx, duplicates):
                new_idx.extend([i] * dup)
            self.idx = new_idx

    @staticmethod
    def _select_split(X_all, y_all, split, n_train, n_val, n_test):
        if split == "tr":
            return X_all[:n_train], y_all[:n_train]
        if split == "va":
            return X_all[n_train : n_train + n_val], y_all[n_train : n_train + n_val]
        if split == "te":
            return (
                X_all[n_train + n_val : n_train + n_val + n_test],
                y_all[n_train + n_val : n_train + n_val + n_test],
            )
        raise NotImplementedError

    @staticmethod
    def _filter_digits(X, y, digits_per_class):
        if digits_per_class >= 5:
            return X, y
        allowed_digits = list(range(digits_per_class)) + list(range(5, 5 + digits_per_class))
        mask = torch.isin(y, torch.tensor(allowed_digits))
        return X[mask], y[mask]

    @staticmethod
    def _apply_subset_indices(X, y, subset_indices):
        subset_indices = torch.tensor(subset_indices, dtype=torch.long)
        return X[subset_indices], y[subset_indices]

    @staticmethod
    def _resize_mnist(mnist_imgs, output_size=32, chunk_size=10000):
        resized_chunks = []
        for start in range(0, len(mnist_imgs), chunk_size):
            end = min(start + chunk_size, len(mnist_imgs))
            chunk = torch.from_numpy(mnist_imgs[start:end]).float().unsqueeze(1)
            chunk_resized = torch.nn.functional.interpolate(
                chunk, size=(output_size, output_size), mode="bilinear", align_corners=False
            )
            resized_chunks.append(chunk_resized.squeeze(1) / 255.0)
        return torch.cat(resized_chunks, dim=0)

    def _build_precomputed_imgs(self, resized_mnist: torch.Tensor) -> torch.Tensor:
        precomputed_imgs = torch.zeros((resized_mnist.shape[0], 3, 32, 32), dtype=torch.float32)
        if self.grayscale:
            precomputed_imgs[:, 0, :, :] = resized_mnist
            precomputed_imgs[:, 1, :, :] = resized_mnist
            precomputed_imgs[:, 2, :, :] = resized_mnist
            return precomputed_imgs

        color_tensor = torch.from_numpy(self.color.astype(np.int64))
        red_mask = (color_tensor == 0).view(-1, 1, 1)
        green_mask = (color_tensor == 1).view(-1, 1, 1)
        precomputed_imgs[:, 0, :, :] = resized_mnist * red_mask
        precomputed_imgs[:, 1, :, :] = resized_mnist * green_mask
        return precomputed_imgs

    def _init_normalization(self) -> None:
        if self.input_size == 28 or self.input_size == 32:
            self._norm_mean = torch.tensor([0.1307, 0.1307, 0.0]).view(3, 1, 1)
            self._norm_std = torch.tensor([0.3081, 0.3081, 0.3081]).view(3, 1, 1)
        else:
            self._norm_mean = torch.tensor([0.0, 0.0, 0.0]).view(3, 1, 1)
            self._norm_std = torch.tensor([1.0, 1.0, 1.0]).view(3, 1, 1)

    def _configure_task_specs(self, hparams, digit_ids) -> None:
        self.task_cardinalities = {
            "label": int(len(np.unique(self.binary_label))),
            "color": 2,
            "environment": 2,
            "digit": int(len(np.unique(digit_ids))),
        }

        task_input = _normalize_cmnist_task_spec(hparams.get("task_input"), default="image")
        task_output = _normalize_cmnist_task_spec(hparams.get("task_output"), default="label")

        self.task_input_spec = [_canonical_cmnist_field(f) for f in task_input]
        self.task_output_spec = [_canonical_cmnist_field(f) for f in task_output]

        if "image" in self.task_input_spec and len(self.task_input_spec) > 1:
            raise ValueError("CMNIST task_input cannot mix 'image' with tabular attributes.")

        if self.task_input_spec == ["image"]:
            self.data_type = "images"
            self.INPUT_SHAPE = (3, self.input_size, self.input_size)
        else:
            self.data_type = "tabular"
            self.INPUT_SHAPE = (len(self.task_input_spec),)

        if len(self.task_output_spec) == 1:
            self.num_labels = self.task_cardinalities[self.task_output_spec[0]]
        else:
            output_bases = [self.task_cardinalities[f] for f in self.task_output_spec]
            self.num_labels = int(np.prod(output_bases))

        self.task_input = self.task_input_spec
        self.task_output = self.task_output_spec

    def _setup_watermarks(self, N: int, rng: np.random.Generator):
        """Generate watermark banks and apply watermarks to the rightmost column.

        Each environment gets a bank of `watermark_bank_size` unique binary
        patterns of length `watermark_bit_count`.  Samples are assigned a
        random watermark from the bank corresponding to their environment.
        Complexity is controlled by `watermark_bank_size`.
        """
        n_wm_bits = self.watermark_bit_count
        bank_size = self.watermark_bank_size

        # Generate two non-overlapping watermark banks (one per environment)
        self.wm_bank_env0, self.wm_bank_env1 = generate_watermark_banks(
            n_bits=n_wm_bits,
            bank_size=bank_size,
            seed=667,
        )

        # Assign a random watermark from the appropriate bank to each sample
        env0_mask = self.environment == 0
        env1_mask = self.environment == 1
        n_env0 = env0_mask.sum()
        n_env1 = env1_mask.sum()

        # Sample uniformly from each bank (with replacement — many samples, few patterns)
        env0_indices = rng.integers(0, bank_size, size=n_env0)
        env1_indices = rng.integers(0, bank_size, size=n_env1)

        # Build watermark bits tensor
        self.watermark_bits = torch.zeros((N, n_wm_bits), dtype=torch.float32)
        self.watermark_bits[env0_mask] = torch.from_numpy(self.wm_bank_env0[env0_indices].astype(np.float32))
        self.watermark_bits[env1_mask] = torch.from_numpy(self.wm_bank_env1[env1_indices].astype(np.float32))

        # Apply watermark noise (env_noisiness corrupts a fraction of samples, not bits)
        env_noisiness = self.hparams.get("env_noisiness", 0.0)
        if env_noisiness > 0:
            noise_sample_mask = rng.random(N) < env_noisiness
            num_noisy = noise_sample_mask.sum()
            if num_noisy > 0:
                random_bits = torch.from_numpy(
                    rng.integers(0, 2, size=(num_noisy, n_wm_bits)).astype(np.float32)
                )
                self.watermark_bits[torch.from_numpy(noise_sample_mask)] = random_bits

        # Vectorized: Apply watermarks to the bottom segment of the rightmost column
        # Shape: (N, B) -> (N, 1, B) -> broadcast to (N, 3, B) for rightmost column
        row_start = 32 - n_wm_bits
        self.precomputed_imgs[:, :, row_start:, -1] = self.watermark_bits.unsqueeze(1)

    def _apply_random_watermarks(self, N: int, rng: np.random.Generator):
        """Apply random watermark bits to the rightmost column of each image."""

        # Generate all random watermark bits at once (N, watermark_bits) as torch tensor
        random_bits = torch.from_numpy(
            rng.integers(0, 2, size=(N, self.watermark_bit_count)).astype(np.float32)
        )

        # Initialize or overwrite watermark bits storage
        self.watermark_bits = random_bits

        # Vectorized: Apply watermarks to bottom segment of the rightmost column
        # Shape: (N, B) -> (N, 1, B) -> broadcast to (N, 3, B) for rightmost column
        row_start = 32 - self.watermark_bit_count
        self.precomputed_imgs[:, :, row_start:, -1] = random_bits.unsqueeze(1)

    def _apply_subsampling(self, hparams, rng):
        """Handle subsampling to keep __init__ clean."""
        mask = None
        n_samples = 0

        if hparams["attr_prob"] > 0.5:
            mask = self.color == 0
            n_samples = int((self.color == 1).sum() * (1 - hparams["attr_prob"]) / hparams["attr_prob"])
        elif hparams["attr_prob"] < 0.5:
            mask = self.color == 1
            n_samples = int((self.color == 0).sum() * hparams["attr_prob"] / (1 - hparams["attr_prob"]))

        if mask is not None:
            self._perform_subsample(mask, n_samples, rng)

    def _perform_subsample(self, mask, n_samples, rng):
        idxs = np.concatenate(
            (
                np.nonzero(~mask)[0],
                rng.choice(np.nonzero(mask)[0], size=n_samples, replace=False),
            )
        )
        rng.shuffle(idxs)
        # Apply shuffle to all arrays
        self._temp_mnist_imgs = self._temp_mnist_imgs[idxs]
        self.color = self.color[idxs]
        self.environment = self.environment[idxs]
        self.binary_label = self.binary_label[idxs]
        self.original_digits = self.original_digits[idxs]

    @staticmethod
    def _resolve_subset_indices(N, *, dataset_size=None, subset_indices=None, subset_seed=None, hparams=None):
        if hparams is None:
            hparams = {}

        rng = np.random.default_rng(subset_seed)
        resolved_indices = None

        if subset_indices is not None:
            resolved_indices = np.asarray(subset_indices, dtype=np.int64)
        elif dataset_size is not None:
            size = max(1, min(int(dataset_size), N))
            resolved_indices = rng.permutation(N)[:size]

        debug_mode = hparams.get("debug_mode", False)
        debug_limit = hparams.get("debug_dataset_limit")
        if debug_mode and debug_limit:
            debug_limit = max(1, min(int(debug_limit), N))
            if resolved_indices is None:
                resolved_indices = rng.permutation(N)[:debug_limit]
            elif len(resolved_indices) > debug_limit:
                resolved_indices = resolved_indices[:debug_limit]

        return resolved_indices

    @classmethod
    def from_base(
        cls,
        base_dataset: "CMNIST",
        *,
        subset_indices=None,
        shuffle_attribute: str = None,
        seed: int = None,
    ) -> "CMNIST":
        rng = np.random.default_rng(seed)
        if subset_indices is None:
            subset_indices = np.arange(len(base_dataset), dtype=np.int64)
        else:
            subset_indices = np.asarray(subset_indices, dtype=np.int64)

        n = len(subset_indices)
        idx_tensor = torch.from_numpy(subset_indices.astype(np.int64))

        grayscale_imgs = base_dataset.grayscale_imgs[idx_tensor].clone()
        y_tensor = base_dataset.y_tensor[idx_tensor].clone()
        a_tensor = base_dataset.a_tensor[idx_tensor].clone()
        env_tensor = base_dataset.env_tensor[idx_tensor].clone()
        digits_tensor = base_dataset.digits_tensor[idx_tensor].clone()
        digit_id_tensor = base_dataset.digit_id_tensor[idx_tensor].clone()

        color = base_dataset.color[subset_indices].copy()
        environment = base_dataset.environment[subset_indices].copy()
        binary_label = base_dataset.binary_label[subset_indices].copy()
        original_digits = base_dataset.original_digits[subset_indices].copy()

        watermark_bits = (
            base_dataset.watermark_bits[idx_tensor].clone()
            if base_dataset.watermark_bits is not None
            else None
        )
        if shuffle_attribute is not None:
            if shuffle_attribute not in {"color", "digit", "watermark"}:
                raise ValueError(
                    f"Unknown attribute: {shuffle_attribute}. Must be 'color', 'digit', or 'watermark'"
                )
            if shuffle_attribute == "watermark" and watermark_bits is None:
                raise ValueError("Cannot shuffle watermark: dataset has no watermarks")

            shuffle_perm = rng.permutation(n)
            if shuffle_attribute == "digit":
                grayscale_imgs = grayscale_imgs[shuffle_perm]
                digits_tensor = digits_tensor[shuffle_perm]
                digit_id_tensor = digit_id_tensor[shuffle_perm]
                original_digits = original_digits[shuffle_perm]
            elif shuffle_attribute == "color":
                color = color[shuffle_perm]
                a_tensor = a_tensor[shuffle_perm]
            elif shuffle_attribute == "watermark":
                watermark_bits = watermark_bits[shuffle_perm]

        # For color-only shuffles, we can swap the R/G channels of already-normalized images
        if shuffle_attribute == "color" and not base_dataset.grayscale:
            base_precomputed = base_dataset.precomputed_imgs[idx_tensor]
            precomputed_imgs = base_precomputed.clone()

            original_color_tensor = torch.from_numpy(base_dataset.color[subset_indices].astype(np.int64))
            new_color_tensor = torch.from_numpy(color.astype(np.int64))
            swap_mask = original_color_tensor != new_color_tensor  # bool tensor (n,)

            # Swap R and G channels where colors changed
            if swap_mask.any():
                red_channel = precomputed_imgs[:, 0, :, :].clone()
                green_channel = precomputed_imgs[:, 1, :, :].clone()
                swap_indices = swap_mask.nonzero(as_tuple=True)[0]
                precomputed_imgs[swap_indices, 0, :, :] = green_channel[swap_indices]
                precomputed_imgs[swap_indices, 1, :, :] = red_channel[swap_indices]
        else:
            # Slow path: full reconstruction needed (digit/watermark changed or grayscale mode)
            precomputed_imgs = torch.zeros((n, 3, 32, 32), dtype=torch.float32)
            if base_dataset.grayscale:
                precomputed_imgs[:, 0, :, :] = grayscale_imgs
                precomputed_imgs[:, 1, :, :] = grayscale_imgs
                precomputed_imgs[:, 2, :, :] = grayscale_imgs
            else:
                color_tensor = torch.from_numpy(color.astype(np.int64))
                red_mask = (color_tensor == 0).view(-1, 1, 1)
                green_mask = (color_tensor == 1).view(-1, 1, 1)
                precomputed_imgs[:, 0, :, :] = grayscale_imgs * red_mask
                precomputed_imgs[:, 1, :, :] = grayscale_imgs * green_mask

            if watermark_bits is not None:
                row_start = 32 - base_dataset.watermark_bit_count
                precomputed_imgs[:, :, row_start:, -1] = watermark_bits.unsqueeze(1)

            precomputed_imgs = (precomputed_imgs - base_dataset._norm_mean) / base_dataset._norm_std

        obj = cls.__new__(cls)
        obj.hparams = dict(base_dataset.hparams)
        obj.input_size = base_dataset.input_size
        obj.INPUT_SHAPE = base_dataset.INPUT_SHAPE
        obj.data_type = base_dataset.data_type
        obj.has_watermark = base_dataset.has_watermark
        obj.watermark_bank_size = base_dataset.watermark_bank_size
        obj.random_watermark = base_dataset.random_watermark
        obj.grayscale = base_dataset.grayscale
        obj.watermark_bit_count = base_dataset.watermark_bit_count

        obj._norm_mean = base_dataset._norm_mean
        obj._norm_std = base_dataset._norm_std

        obj.precomputed_imgs = precomputed_imgs
        obj.grayscale_imgs = grayscale_imgs
        obj.watermark_bits = watermark_bits

        obj.y_tensor = y_tensor
        obj.a_tensor = a_tensor
        obj.env_tensor = env_tensor
        obj.digits_tensor = digits_tensor
        obj.digit_id_tensor = digit_id_tensor

        obj.color = color
        obj.environment = environment
        obj.binary_label = binary_label
        obj.original_digits = original_digits

        obj.idx = list(range(n))
        obj.digits = original_digits
        obj.y = binary_label
        obj.a = color
        obj.env = environment
        obj.num_labels = base_dataset.num_labels
        obj.num_attributes = base_dataset.num_attributes
        obj.task_cardinalities = base_dataset.task_cardinalities
        obj.task_input_spec = base_dataset.task_input_spec
        obj.task_output_spec = base_dataset.task_output_spec
        obj.task_input = base_dataset.task_input
        obj.task_output = base_dataset.task_output

        return obj

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        # Use idx for potential duplicates support
        i = self.idx[index]

        if self.task_input_spec == ["image"]:
            x = self.precomputed_imgs[i]
        else:
            components = [_cmnist_get_component_tensor(self, f)[i].float() for f in self.task_input_spec]
            x = torch.stack(components, dim=0)

        if len(self.task_output_spec) == 1:
            y = _cmnist_get_component_tensor(self, self.task_output_spec[0])[i]
        else:
            output_values = [_cmnist_get_component_tensor(self, f)[i].item() for f in self.task_output_spec]
            output_bases = [self.task_cardinalities[f] for f in self.task_output_spec]
            y = torch.tensor(_cmnist_encode_fields(output_values, output_bases), dtype=torch.long)

        # Returns: (index, x, y, color, env, digit)
        return (index, x, y, self.a_tensor[i], self.env_tensor[i], self.digits_tensor[i])

    def override_tabular_inputs_(
        self,
        *,
        environment=None,
        color=None,
        digit=None,
        indices=None,
    ) -> None:
        """Override tabular CMNIST inputs in-place.

        Intended for the tabular subtask where `task_input` is drawn from
        {environment, color, digit}. This updates both the numpy arrays and the
        torch tensors used by `__getitem__`.

        Notes:
            - `digit` refers to the contiguous digit-id used by the CMNIST
              `digit` task field (i.e. `digit_id_tensor`), not the raw EMNIST
              digit value.
            - If `indices` is provided, only those rows are updated.
        """

        def _to_long_tensor(values, *, device):
            if values is None:
                return None
            if isinstance(values, torch.Tensor):
                return values.to(device=device, dtype=torch.long)
            return torch.as_tensor(values, device=device, dtype=torch.long)

        device = self.y_tensor.device if hasattr(self, "y_tensor") else torch.device("cpu")
        idx_t = _to_long_tensor(indices, device=device) if indices is not None else None
        if idx_t is not None and idx_t.numel() == 0:
            return

        env_t = _to_long_tensor(environment, device=device)
        col_t = _to_long_tensor(color, device=device)
        digit_id_t = _to_long_tensor(digit, device=device)

        if env_t is not None:
            if idx_t is None:
                self.environment = env_t.detach().cpu().numpy().astype(np.int64)
                self.env_tensor = env_t
            else:
                idx_np = idx_t.detach().cpu().numpy()
                self.environment[idx_np] = env_t.detach().cpu().numpy().astype(np.int64)
                self.env_tensor[idx_t] = env_t
            self.env = self.environment

        if col_t is not None:
            if idx_t is None:
                self.color = col_t.detach().cpu().numpy().astype(np.int64)
                self.a_tensor = col_t
            else:
                idx_np = idx_t.detach().cpu().numpy()
                self.color[idx_np] = col_t.detach().cpu().numpy().astype(np.int64)
                self.a_tensor[idx_t] = col_t
            self.a = self.color

        if digit_id_t is not None:
            if idx_t is None:
                self.digit_id_tensor = digit_id_t
            else:
                self.digit_id_tensor[idx_t] = digit_id_t

    def _reconstruct_image(
        self, grayscale_img: torch.Tensor, color: int, watermark_bits: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Reconstruct an RGB image from grayscale digit, color, and optional watermark.

        Args:
            grayscale_img: (32, 32) float32 grayscale digit image in [0, 1]
            color: 0 for red, 1 for green
            watermark_bits: (B,) float32 tensor of watermark bits, or None

        Returns:
            (3, 32, 32) float32 RGB image in [0, 1]
        """
        img = torch.zeros((3, 32, 32), dtype=torch.float32)

        # Apply color (red=0, green=1)
        if color == 0:
            img[0, :, :] = grayscale_img
        else:
            img[1, :, :] = grayscale_img

        # Apply watermark if provided
        if watermark_bits is not None:
            row_start = 32 - self.watermark_bit_count
            img[:, row_start:, -1] = watermark_bits.unsqueeze(0)

        return img

    def create_shuffled_view(self, attribute: str, seed: int = None) -> "CMNIST":
        """
        Create a view of the dataset with one attribute shuffled.

        Args:
            attribute: One of 'color', 'digit', or 'watermark'
            seed: Random seed for shuffling

        Returns:
            CMNIST dataset with the shuffled attribute
        """
        if attribute not in ["color", "digit", "watermark"]:
            raise ValueError(f"Unknown attribute: {attribute}. Must be 'color', 'digit', or 'watermark'")

        if attribute == "watermark" and self.watermark_bits is None:
            raise ValueError("Cannot shuffle watermark: dataset has no watermarks")

        return CMNIST.from_base(self, shuffle_attribute=attribute, seed=seed)

    def create_subset(self, n_samples: int, seed: int = None) -> "CMNIST":
        """
        Create a new native CMNIST dataset object with only n_samples samples.

        Args:
            n_samples: Number of samples to include in the subset
            seed: Random seed for selecting samples

        Returns:
            CMNIST dataset with fewer samples
        """
        n_samples = min(int(n_samples), len(self))
        rng = np.random.default_rng(seed)
        subset_indices = rng.choice(len(self), size=n_samples, replace=False)
        return CMNIST.from_base(self, subset_indices=subset_indices, seed=seed)


def generate_watermark_banks(
    n_bits: int,
    bank_size: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate two banks of unique random binary watermark patterns.

    Each bank contains `bank_size` distinct binary vectors of length `n_bits`.
    The two banks are guaranteed to have no overlap (no shared patterns).

    Args:
        n_bits: Length of each watermark pattern (number of binary pixels).
        bank_size: Number of unique watermarks per bank (controls complexity).
        seed: Random seed for reproducibility.

    Returns:
        bank_0: shape (bank_size, n_bits), uint8 in {0, 1}
        bank_1: shape (bank_size, n_bits), uint8 in {0, 1}
    """
    rng = np.random.default_rng(seed)

    # Total unique patterns needed: 2 * bank_size (non-overlapping banks)
    total_needed = 2 * bank_size
    max_possible = 2**n_bits

    if total_needed > max_possible:
        raise ValueError(
            f"Cannot create 2 non-overlapping banks of size {bank_size} "
            f"with only {n_bits} bits ({max_possible} possible patterns). "
            f"Max bank_size for {n_bits} bits = {max_possible // 2}."
        )

    if total_needed <= max_possible // 2:
        # Plenty of room: sample with rejection to ensure uniqueness
        all_patterns = set()
        patterns_list = []
        while len(patterns_list) < total_needed:
            batch = rng.integers(0, 2, size=(total_needed * 2, n_bits), dtype=np.uint8)
            for row in batch:
                key = row.tobytes()
                if key not in all_patterns:
                    all_patterns.add(key)
                    patterns_list.append(row)
                    if len(patterns_list) >= total_needed:
                        break
        all_unique = np.array(patterns_list[:total_needed])
    else:
        # Enumerate all possible patterns and sample without replacement
        all_possible = np.array(
            [[int(b) for b in format(i, f"0{n_bits}b")] for i in range(max_possible)],
            dtype=np.uint8,
        )
        indices = rng.choice(max_possible, size=total_needed, replace=False)
        all_unique = all_possible[indices]

    # Split into two non-overlapping banks
    rng.shuffle(all_unique)
    bank_0 = all_unique[:bank_size]
    bank_1 = all_unique[bank_size:]

    return bank_0, bank_1
