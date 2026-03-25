import torch
import torch.nn as nn
from torch.optim import AdamW

from source import networks

ALGORITHMS = [
    "ERM",
]


# =============================================================================
# Optimizers
# =============================================================================


def get_adamw_optim(network, lr, weight_decay):
    no_decay = ["bias", "LayerNorm.weight"]
    decay_params = []
    no_decay_params = []
    for n, p in network.named_parameters():
        if any(nd in n for nd in no_decay):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
    return optimizer


def get_sgd_optim(network, lr, weight_decay):
    return torch.optim.SGD(
        network.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
    )


def get_adam_optim(network, lr, weight_decay):
    return torch.optim.Adam(
        network.parameters(), lr=lr, weight_decay=weight_decay
    )


get_optimizers = {"sgd": get_sgd_optim, "adam": get_adam_optim, "adamw": get_adamw_optim}


# =============================================================================
# Algorithms
# =============================================================================


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    Base class for learning algorithms.

    Subclasses should implement:
    - _init_model()
    - _compute_loss()
    - update()
    - return_feats()
    - predict()
    """

    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(Algorithm, self).__init__()
        self.hparams = hparams
        self.data_type = data_type
        self.num_classes = num_classes
        self.num_attributes = num_attributes
        self.num_examples = num_examples

    def _init_model(self):
        raise NotImplementedError

    def _compute_loss(self, i, x, y, a, step):
        raise NotImplementedError

    def update(self, minibatch, step):
        raise NotImplementedError

    def return_feats(self, x):
        raise NotImplementedError

    def predict(self, x, y=None, return_loss: bool = False):
        raise NotImplementedError

    def return_groups(self, y, a):
        """Return indices of samples belonging to each (y, a) subgroup."""
        idx_g, idx_samples = [], []
        all_g = y * self.num_attributes + a
        for g in all_g.unique():
            idx_g.append(g)
            idx_samples.append(all_g == g)
        return zip(idx_g, idx_samples)

    @staticmethod
    def return_attributes(all_a):
        """Return indices of samples belonging to each attribute value."""
        idx_a, idx_samples = [], []
        for a in all_a.unique():
            idx_a.append(a)
            idx_samples.append(all_a == a)
        return zip(idx_a, idx_samples)


class ERM(Algorithm):
    """Empirical Risk Minimization (ERM)."""

    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(ERM, self).__init__(data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)

        self.featurizer = networks.Featurizer(data_type, input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs, num_classes, self.hparams["nonlinear_classifier"]
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self._init_model()

    def _init_model(self):
        self.optimizer = get_optimizers[self.hparams["optimizer"]](
            self.network, self.hparams["lr"], self.hparams["weight_decay"]
        )
        self.loss = torch.nn.CrossEntropyLoss(reduction="none")

    def _compute_loss(self, i, x, y, a, step):
        return self.loss(self.predict(x), y).mean()

    def update(self, minibatch, step):
        # Support (i, x, y, a), (i, x, y, a, digit), or (i, x, y, a, env, digit)
        all_i, all_x, all_y, all_a = minibatch[:4]
        loss = self._compute_loss(all_i, all_x, all_y, all_a, step)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def return_feats(self, x):
        return self.featurizer(x)

    def predict(self, x, y=None, return_loss: bool = False):
        logits = self.network(x)
        if return_loss and (y is not None):
            loss_value = self.loss(logits, y).mean()
            return logits, loss_value
        return logits
