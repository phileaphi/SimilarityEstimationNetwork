import torch
import torch.nn.functional as F
from torch.nn.modules.loss import *  # noqa # expose all in torch.optim through here
from torch.nn.modules.loss import _Loss, _WeightedLoss


# WIP
class ClassRectificationLoss(_WeightedLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(
        self,
        weight=None,
        size_average=None,
        ignore_index=-100,
        reduce=None,
        reduction='mean'
    ):
        super(ClassRectificationLoss,
              self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        raise NotImplementedError
        # def cross_entropy(
        #     input,
        #     target,
        #     weight=None,
        #     size_average=None,
        #     ignore_index=-100,
        #     reduce=None,
        #     reduction='mean'
        # ):
        #     if size_average is not None or reduce is not None:
        #         reduction = _Reduction.legacy_get_string(size_average, reduce)
        #     return nll_loss(
        #         log_softmax(input, 1), target, weight, None, ignore_index, None, reduction
        #     )

        # n_bs = len(target)
        # pn_bs = 0.5 * n_bs
        # h = torch.sum(target, dim=0)


class VAELoss(_Loss):
    def forward(self, input, target):
        recon_x, mu, logvar = input
        x = target / 255
        if torch.isinf(recon_x).any() or torch.isnan(recon_x).any():
            raise ValueError
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        # BCE = F.mse_loss(recon_x, x, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        if torch.isinf(KLD) or torch.isnan(KLD).any():
            raise ValueError
        return BCE + KLD
