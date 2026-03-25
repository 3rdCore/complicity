import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams["mlp_width"])
        self.dropout = nn.Dropout(hparams["mlp_dropout"])
        self.hiddens = nn.ModuleList(
            [nn.Linear(hparams["mlp_width"], hparams["mlp_width"]) for _ in range(hparams["mlp_depth"] - 2)]
        )
        self.output = nn.Linear(hparams["mlp_width"], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class SimpleMLP(nn.Module):
    """Two-hidden-layer MLP featurizer with Xavier initialisation.

    Used as the default featurizer for 32×32 image inputs.  Flattens the
    input before passing it through two linear layers with ReLU activations.
    """

    def __init__(self, input_shape, hparams):
        super(SimpleMLP, self).__init__()
        self.n_outputs = hparams.get("n_outputs", 256)

        if len(input_shape) == 1:
            input_size = input_shape[0]
        else:
            input_size = input_shape[0] * input_shape[1] * input_shape[2]

        self.lin1 = nn.Linear(input_size, self.n_outputs)
        self.lin2 = nn.Linear(self.n_outputs, self.n_outputs)

        for lin in [self.lin1, self.lin2]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = x.view(x.shape[0], -1)
        out = self.relu1(self.lin1(out))
        out = self.relu2(self.lin2(out))
        return out


class SimpleCNN(nn.Module):
    """Small CNN featurizer for 32×32 images."""

    def __init__(self, input_shape, hparams):
        super(SimpleCNN, self).__init__()
        self.n_outputs = hparams.get("n_outputs", 128)

        c1 = self.n_outputs // 2
        c2 = self.n_outputs

        self.conv1 = nn.Conv2d(input_shape[0], c1, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(c2, c2, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(c2, c2, 3, 1, padding=1)

        g1 = min(8, c1)
        g2 = min(8, c2)

        self.bn0 = nn.GroupNorm(g1, c1)
        self.bn1 = nn.GroupNorm(g2, c2)
        self.bn2 = nn.GroupNorm(g2, c2)
        self.bn3 = nn.GroupNorm(g2, c2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn0(self.conv1(x)))
        x = F.relu(self.bn1(self.conv2(x)))
        x = F.relu(self.bn2(self.conv3(x)))
        x = F.relu(self.bn3(self.conv4(x)))
        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x


def Featurizer(data_type, input_shape, hparams):
    """Return the appropriate featurizer for the given data type and input shape.

    Supports:
    - ``data_type='images'`` with 32×32 inputs: ``image_arch='simple_mlp'`` (default) or ``'cnn'``
    - ``data_type='tabular'``: always uses SimpleMLP (1-D input)
    """
    arch = hparams.get("image_arch", "simple_mlp")
    if data_type == "images":
        if len(input_shape) == 1:
            return MLP(input_shape[0], hparams["mlp_width"], hparams)
        if arch == "simple_mlp":
            return SimpleMLP(input_shape, hparams)
        elif arch == "cnn":
            return SimpleCNN(input_shape, hparams)
        else:
            raise NotImplementedError(f"Unknown image_arch: {arch!r}. Choose 'simple_mlp' or 'cnn'.")
    elif data_type == "tabular":
        return SimpleMLP(input_shape, hparams)
    else:
        raise NotImplementedError(f"Unsupported data_type: {data_type!r}.")


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features),
        )
    else:
        return torch.nn.Linear(in_features, out_features)
