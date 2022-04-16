import copy
from typing import NamedTuple

from torch import nn
import torch


class BranchOutput(NamedTuple):
    y: torch.Tensor = None
    z: torch.Tensor = None
    v: torch.Tensor = None
    q: torch.Tensor = None
    q_m: torch.Tensor = None


class OnlineTargetOutput(NamedTuple):
    online: BranchOutput
    target: BranchOutput
    logits: torch.Tensor = None


class MYOW(torch.nn.Module):
    r"""
    When the projectors are cascaded, the two views are separately forwarded through the online and target networks:
    .. math::
        y = f_{\theta}(x),\  z = g_{\theta}(y), v = h_{\theta}(z)\\
        y^\prime = f_{\xi}(x^\prime),\  z^\prime = g_{\xi}(y^\prime), v^\prime = h_{\xi}(z^\prime)

    then prediction is performed either in the first projection space or the second.
    In the first, the predictor learns to predict the target projection from the online projection in order to minimize
        the following loss:
    .. math::
        \mathcal{L}_{\theta, \xi} = 2-2 \cdot \frac{\left\langle q_{\theta}\left(z\right),
        z^{\prime}\right\rangle}{\left\|q_{\theta}\left(z\right)\right\|_{2}
        \cdot\left\|z^{\prime}\right\|_{2}}

    In the second, the second predictor learns to predict the second target projection from the second online projection
        in order to minimize the following loss:
    .. math::
        \mathcal{L}_{\theta, \xi} = 2-2 \cdot \frac{\left\langle r_{\theta}\left(v\right),
        v^{\prime}\right\rangle}{\left\|v_{\theta}\left(v\right)\right\|_{2}
        \cdot\left\|v^{\prime}\right\|_{2}}.

    Args:
        encoder (torch.nn.Module): Encoder network to be duplicated and used in both online and target networks.
        projector (torch.nn.Module): Projector network to be duplicated and used in both online and target networks.
        projector_m (torch.nn.Module): Second projector network to be duplicated and used in both online and
            target networks.
        predictor (torch.nn.Module): Predictor network used to predict the target projection from the online projection.
        predictor_m (torch.nn.Module): Second predictor network used to predict the target projection from the
            online projection.
        layout (String, Optional): Defines the layout of the dual projectors. Can be either :obj:`"cascaded"` or
            :obj:`"parallel"`. (default: :obj:`"cascaded"`)
    """
    __trainable_modules = ['online_encoder', 'online_projector', 'online_projector_m', 'predictor', 'predictor_m']
    ema_module_pairs = [
        ('online_encoder', 'target_encoder'),
        ('online_projector', 'target_projector'),
        ('online_projector_m', 'target_projector_m')
    ]

    def __init__(self, encoder, projector, projector_m, predictor, predictor_m, logits=None, layout='cascaded'):
        super().__init__()
        self.online_encoder = encoder
        self.target_encoder = self._mirror_network(self.online_encoder)
        self.target_encoder.reset_parameters()
        self._stop_gradient(self.target_encoder)

        assert layout in ['cascaded', 'parallel'], "layout should be 'cascaded' or 'parallel', got {}.".format(layout)
        self.layout = layout

        # Projector and predictor for augmented views
        self.online_projector = projector
        self.target_projector = self._mirror_network(self.online_projector)
        self._stop_gradient(self.target_projector)

        self.predictor = predictor

        # Projector and predictor for mined views
        self.online_projector_m = projector_m
        self.target_projector_m = self._mirror_network(self.online_projector_m)
        self._stop_gradient(self.target_projector_m)

        self.predictor_m = predictor_m

        # This is used to provide an evaluation of the representation quality during training.
        self.logits = logits

    def _mirror_network(self, online_net):
        target_net = copy.deepcopy(online_net)
        return target_net

    def _stop_gradient(self, net):
        r"""Stops parameters of :obj:`network` of being updated through back-propagation."""
        for param in net.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _reset_moving_average(self):
        r"""Resets target network to have the same parameters as the online network."""
        for online_module, target_module in self._ema_module_pairs:
            for param_q, param_k in zip(online_module.parameters(), target_module.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False     # stop gradient

    @property
    def _ema_module_pairs(self):
        return [(self.__getattr__(ms), self.__getattr__(mt)) for ms, mt in self.ema_module_pairs]

    @torch.no_grad()
    def update_target_network(self, mm):
        r"""Performs a momentum update of the target network's weights.

        Args:
            mm (float): Momentum used in moving average update.
        """
        assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm
        for online_module, target_module in self._ema_module_pairs:
            for param_q, param_k in zip(online_module.parameters(), target_module.parameters()):
                param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    @property
    def trainable_modules(self):
        r"""Returns the list of modules that will updated via an optimizer."""
        return [self.__getattr__(m) for m in self.__trainable_modules]

    def forward(self, online_view=None, target_view=None):
        online, target, logits = None, None, None
        if online_view is not None:
            # forward online network
            online_y = self.online_encoder(online_view)
            online_z = self.online_projector(online_y)
            online_q = self.predictor(online_z)
            if self.layout == 'parallel':
                online_v = self.online_projector_m(online_y)
            elif self.layout == 'cascaded':
                online_v = self.online_projector_m(online_z)
            online_q_m = self.predictor_m(online_v)
            online =BranchOutput(y=online_y, z=online_z, v=online_v, q=online_q, q_m=online_q_m)

            if self.logits is not None:
                logits = self.logits(online_y.detach())

        if target_view is not None:
            # forward target network
            with torch.no_grad():
                target_y = self.target_encoder(target_view)
                target_z = self.target_projector(target_y)
                if self.layout == 'parallel':
                    target_v = self.target_projector_m(target_y)
                elif self.layout == 'cascaded':
                    target_v = self.target_projector_m(target_z)
            target = BranchOutput(y=target_y.detach(), z=target_z.detach(), v=target_v.detach())
        return OnlineTargetOutput(online=online, target=target, logits=logits)


class MLP3(nn.Module):
    r"""MLP class used for projector and predictor in :class:`BYOL`. The MLP has one hidden layer.

    .. note::
        The hidden layer should be larger than both input and output layers, according to the
        :class:`BYOL` paper.

    Args:
        input_size (int): Size of input features.
        output_size (int): Size of output features (projection or prediction).
        hidden_size (int, optional): Size of hidden layer. (default: :obj:`4096`).
    """
    def __init__(self, input_size, output_size, hidden_size=4096, batchnorm_mm=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size, eps=1e-5, momentum=batchnorm_mm),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size, bias=True)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                    m.reset_parameters()

    def forward(self, x):
        return self.net(x)

