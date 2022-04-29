# -*- coding: utf-8 -*-
"""
Reference: https://github.com/pankajmishra000/VT-ADL

@author: Pankaj Mishra
"""

import torch
import torch.nn as nn

from models.VT_AE.student_transformer import ViT
import models.VT_AE.model_res18 as M
import models.VT_AE.spatial as S


class VT_AE(nn.Module):
    def __init__(self, image_size=512,
                 patch_size=64,
                 num_classes=1,
                 dim=512,
                 depth=6,
                 heads=8,
                 mlp_dim=1024,
                 train=True):

        super(VT_AE, self).__init__()
        self.vt = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim)
        out_dim_caps = 512
        if image_size == 512:
            self.decoder = M.decoder512(out_dim_caps // 64)
        elif image_size == 28:
            self.decoder = M.decoder28(out_dim_caps//64)
        else:
            ...
        # self.G_estimate= mdn1.MDN() # Trained in modular fashion
        self.Digcap = S.DigitCaps(in_num_caps=((image_size // patch_size) ** 2) * 8 * 8,
                                  in_dim_caps=dim//64, out_dim_caps=out_dim_caps)
        self.mask = torch.ones(1, image_size // patch_size, image_size // patch_size).bool()
        self.Train = train

        if self.Train:
            print("\nInitializing network weights.........")
            S.initialize_weights(self.vt, self.decoder)

    def forward(self, x):
        b = x.size(0)
        encoded = self.vt(x, self.mask.to(x.device))
        if self.training:
            encoded = add_noise(encoded)
        encoded1, vectors = self.Digcap(encoded.view(b, encoded.size(1) * 8 * 8, -1))
        recons = self.decoder(encoded1.view(b, -1, 8, 8))
        # pi, mu, sigma = self.G_estimate(encoded)       
        # return encoded, pi, sigma, mu, recons

        return encoded, recons


##### Adding Noise ############

def add_noise(latent, noise_type="gaussian", sd=0.2):
    """Here we add noise to the latent features concatenated from the 4 autoencoders.
    Arguements:
    'gaussian' (string): Gaussian-distributed additive noise.
    'speckle' (string) : Multiplicative noise using out = image + n*image, where n is uniform noise with specified mean & variance.
    'sd' (float) : standard deviation used for generating noise

    Input :
        latent : numpy array or cuda tensor.

    Output:
        Array: Noise added input, can be np array or cuda tnesor.
    """
    assert sd >= 0.0
    if noise_type == "gaussian":
        noise = torch.randn_like(latent) * sd
        latent = latent + noise
        return latent

    if noise_type == "speckle":
        noise = torch.randn_like(latent)
        latent = latent + latent * noise
        return latent



if __name__ == "__main__":
    from torchsummary import summary

    mod = VT_AE()
    print(mod)
    summary(mod, (3, 512, 512))
