import torch
import numpy as np

def kl_criterion_unit(mu, logvar):
    kld = ((torch.exp(logvar) + mu ** 2) - logvar - 1) / 2
    return kld.sum() / np.prod(mu.shape)

def kl_criterion(mu1, logvar1, mu2, logvar2):

    sigma1 = logvar1.mul(0.5).exp()
    sigma2 = logvar2.mul(0.5).exp()
    kld = torch.log(sigma2 / sigma1) + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (
            2 * torch.exp(logvar2)) - 1 / 2
    return kld.sum() / np.prod(mu1.shape)