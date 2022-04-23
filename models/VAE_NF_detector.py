import easydict
import torch
import torchvision
from torch import nn
import numpy as np
import torch.nn.functional as F

from models.VAE_detector import View
import normflow as nf


def sample_gaussian(mu, logvar):
    eps = torch.randn_like(mu)
    eps.requires_grad = False
    sample = mu + eps * torch.exp(logvar / 2)
    return sample


class VAE_NF_Detector(nn.Module):

    def __init__(self, config_model, device, *args, **kwargs):
        super(VAE_NF_Detector, self).__init__()
        image_size = config_model.image_size
        in_channels = config_model.in_channels
        latent_dim = config_model.latent_dim
        param_mlp_hidden = config_model.param_mlp_hidden
        emb_dim = config_model.emb_dim
        x_emb_dim = config_model.x_emb_dim

        # new params
        flow_type = config_model.flow_type
        n_flows = config_model.n_flows
        self.num_samples = config_model.num_samples

        n_bottleneck = latent_dim

        # H_out = (H-F+2P)/S+1
        if image_size == 512:
            cnn_params = [
                # in_dim, out_dim, kernel_size, stride, (padding)
                [in_channels, 4, 8, 2],  # 253
                [4, 8, 5, 2],  # 125
                [8, 16, 5, 2],  # 61
                [16, 32, 7, 2],  # 28
                [32, 32, 3, 1],  # 26
                [32, 32, 3, 1],  # 24
                [32, 32, 4, 2],  # 11
                [32, 64, 3, 1],  # 9
                [64, 64, 3, 2],  # 4
                [64, emb_dim, 4, 1],  # 1, 1, emb_dim
            ]
            # cnn_params = [
            #     [in_channels, 8, 11, 1],  # 502
            #     [8, 16, 6, 1],  # 497
            #     [16, 32, 9, 2],  # 245
            #     [32, 32, 7, 5, 1],  # 49
            #     [32, 16, 9, 3, 1],  # 15
            #     [16, 16, 3, 2, 1],  # 8
            #     [16, 32, 4, 2],  # 3
            #     [32, emb_dim, 3, 1],  # 1
            # ]
        elif image_size == 28:
            assert in_channels == 1
            # 784 - [32C3-32C3-32C5S2] - [64C3-64C3-64C5S2] - 128 - 10
            # ref: https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist
            cnn_params = [
                [in_channels, 32, 3, 1],  # 26, 26, 32
                [32, 32, 3, 1],  # 24, 24, 32
                [32, 32, 4, 2],  # 11, 11, 32
                [32, 64, 3, 1],  # 9, 9, 64
                [64, 64, 3, 2],  # 4, 4, 64
                [64, emb_dim, 4, 1],  # 1, 1, emb_dim
            ]
        else:
            raise NotImplementedError()
        prior = torch.distributions.MultivariateNormal(torch.zeros(n_bottleneck, device=device),
                                                       torch.eye(n_bottleneck, device=device))
        encoder_cnns = []
        cur_size = image_size
        for arg in cnn_params:
            new_size = (cur_size - arg[2]) / arg[3] + 1
            print(f'in {cur_size}, out {new_size}')
            cur_size = new_size
            encoder_cnns.append(nn.Conv2d(*arg))
            encoder_cnns.append(nn.BatchNorm2d(arg[1]))
            encoder_cnns.append(nn.ReLU(True))
        encoder_cnns = encoder_cnns[:-2]
        encoder_cnns.append(nn.Flatten())
        encoder_cnns.append(nf.nets.MLP([emb_dim, 2 * n_bottleneck], init_zeros=True))
        encoder_nn = nn.Sequential(*encoder_cnns)
        # cnn_params[0] = [x_emb_dim] + list(cnn_params[0])[1:]
        # cnn_params[-1] = [latent_dim] + list(cnn_params[0])[1:]
        cnn_params[0][0] = x_emb_dim
        cnn_params[-1][1] = latent_dim
        decoder_cnns = [View([latent_dim, 1, 1])]
        for arg in reversed(cnn_params):
            decoder_cnns.append(nn.ConvTranspose2d(arg[1], arg[0], *arg[2:]))
            decoder_cnns.append(nn.BatchNorm2d(arg[0]))
            decoder_cnns.append(nn.ReLU(True))
        decoder_nn = nn.Sequential(*decoder_cnns[:-2])

        encoder = nf.distributions.NNDiagGaussian(encoder_nn)
        decoder = nf.distributions.NNBernoulliDecoder(decoder_nn)

        if flow_type == 'Planar':
            flows = [nf.flows.Planar((n_bottleneck,)) for k in range(n_flows)]
        elif flow_type == 'Radial':
            flows = [nf.flows.Radial((n_bottleneck,)) for k in range(n_flows)]
        elif flow_type == 'RealNVP':
            b = torch.tensor(n_bottleneck // 2 * [0, 1] + n_bottleneck % 2 * [0], device=device)
            flows = []
            for i in range(n_flows):
                s = nf.nets.MLP([n_bottleneck, n_bottleneck], init_zeros=True)
                t = nf.nets.MLP([n_bottleneck, n_bottleneck], init_zeros=True)
                if i % 2 == 0:
                    flows += [nf.flows.MaskedAffineFlow(b, t, s)]
                else:
                    flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
        else:
            raise NotImplementedError

        self.nfm = nf.NormalizingFlowVAE(prior, encoder, flows, decoder)

        self.result_cache = easydict.EasyDict({
            'mu_z': None, 'logvar_z': None, 'mu_x': None, 'logvar_x': None, 'nll_pixel': None
        })
        self.loss_cache = easydict.EasyDict({
            'nll': None, 'kl_qp': None, 'loss': None
        })

    def write_loss(self, writer, epoch):
        writer.add_scalar('mean_log_q', self.loss_cache.mean_log_q.item(), epoch)
        writer.add_scalar('mean_log_p', self.loss_cache.mean_log_p.item(), epoch)
        writer.add_scalar('loss', self.loss_cache.loss.item(), epoch)

    def write_hist(self, writer, epoch):
        writer.add_histogram('z', self.result_cache.z, epoch)

    def write_reconstructions(self, writer, epoch):
        reconstructions = self.result_cache.z
        writer.add_image(
            'Reconstructed Sample',
            torchvision.utils.make_grid(reconstructions.clamp(0., 1.)),
            epoch, dataformats='CHW')

    def forward(self, x, *args, **kwargs):
        z, log_q, log_p = self.nfm(x, self.num_samples)
        mean_log_q = torch.mean(log_q)
        mean_log_p = torch.mean(log_p)
        loss = mean_log_q - mean_log_p
        self.result_cache.update({
            'z': z.mean(dim=1)
        })
        self.loss_cache.update({
            'mean_log_q': mean_log_q, 'mean_log_p': mean_log_p, 'loss': loss,
            # 'score': log_p.mean(dim=1).sum()  # mean over all samples
        })
        return loss

    def get_ano_score(self):
        return -self.loss_cache.mean_log_p.detach().cpu().numpy()

    def get_ano_loc_score(self):
        pass
