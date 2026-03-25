import numpy as np

from source.utils import misc


def _hparams(algorithm, random_seed):
    """Global registry of hyperparameters for CMNIST + ERM.

    Each entry is a ``(default, random)`` tuple.
    """
    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        assert name not in hparams
        random_state = np.random.RandomState(misc.seed_hash(random_seed, name))
        hparams[name] = (default_val, random_val_fn(random_state))

    _hparam("resnet18", False, lambda r: False)
    _hparam("nonlinear_classifier", False, lambda r: False)
    _hparam("group_balanced", False, lambda r: False)
    _hparam("pretrained", False, lambda r: False)

    _hparam("lr", 1e-3, lambda r: 10 ** r.uniform(-4, -2))
    _hparam("weight_decay", 1e-4, lambda r: 10 ** r.uniform(-6, -3))
    _hparam("optimizer", "adamw", lambda r: "adamw")
    _hparam("last_layer_dropout", 0.0, lambda r: 0.0)
    _hparam("batch_size", 64, lambda r: int(2 ** r.uniform(6, 7)))

    return hparams


def default_hparams(algorithm, dataset):
    return {a: b for a, (b, c) in _hparams(algorithm, 0).items()}


def random_hparams(algorithm, dataset, seed):
    return {a: c for a, (b, c) in _hparams(algorithm, seed).items()}
