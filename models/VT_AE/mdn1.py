# -*- coding: utf-8 -*-
"""
Reference: https://github.com/pankajmishra000/VT-ADL

@author: Pankaj Mishra

Refrence: https://github.com/moonl1ght/MDN/blob/master/MDN.ipynb
for the no of parameters - sum(p.numel() for p in model.parameters() if p.requires_grad)
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class MDN(nn.Module):

    def __init__(self, input_dim=512, out_dim=512, layer_size=512, coefs=10):
        super(MDN, self).__init__()
        self.in_features = input_dim

        self.pi = nn.Linear(layer_size, coefs, bias=False)
        self.mu = nn.Linear(layer_size, out_dim * coefs, bias=False)  # mean
        self.sigma_sq = nn.Linear(layer_size, out_dim * coefs, bias=False)  # isotropic independent variance
        self.out_dim = out_dim
        self.coefs = coefs

    def forward(self, x):
        ep = np.finfo(float).eps
        x = torch.clamp(x, ep)

        pi = F.softmax(self.pi(x), dim=-1)
        sigma_sq = F.softplus(self.sigma_sq(x)).view(x.size(0), x.size(1), self.in_features, -1)  # logvar
        mu = self.mu(x).view(x.size(0), x.size(1), self.in_features, -1)  # mean
        return pi, mu, sigma_sq


'''
    functions to compute the log-likelihood of data according to a gaussian mixture model.
    All the computations are done with log because exp will lead to numerical underflow errors.
'''


def log_gaussian(x, mean, logvar):
    """
    Computes the Gaussian log-likelihoods

    Parameters:
        x: [samples,features]  data samples
        mean: [features]  Gaussian mean (in a features-dimensional space)
        logvar: [features]  the logarithm of variances [no linear dependance hypotesis: we assume one variance per dimension]

    Returns:
         [samples]   log-likelihood of each sample
    """

    x = x.unsqueeze(-1).expand_as(logvar)
    a = (x - mean) ** 2  # works on multiple samples thanks to tensor broadcasting
    log_p = (logvar + a / (torch.exp(logvar))).sum(2)
    log_p = -0.5 * (np.log(2 * np.pi) + log_p)

    return log_p


def log_gmm(x, means, logvars, weights, total=True):
    """
    Computes the Gaussian Mixture Model log-likelihoods

    Parameters:
        x: [samples,features]  data samples
        means:  [K,features]   means for K Gaussians
        logvars: [K,features]  logarithm of variances for K Gaussians  [no linear dependance hypotesis: we assume one variance per dimension]
        weights: [K]  the weights of each Gaussian
        total:   wether to sum the probabilities of each Gaussian or not (see returning value)

    Returns:
        [samples]  if total=True. Log-likelihood of each sample
        [K,samples] if total=False. Log-likelihood of each sample for each model

    """
    res = -log_gaussian(x, means, logvars)  # negative of log likelihood

    res = weights * res

    if total:
        return torch.sum(res, 2)
    else:
        return res


def mdn_loss_function(x, means, logvars, weights, test=False):
    if test:
        res = log_gmm(x, means, logvars, weights)
    else:
        res = torch.mean(torch.sum(log_gmm(x, means, logvars, weights), 1))
    return res


if __name__ == "__main__":
    model = MDN()
    print(model)
