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
        self.prior = torch.distributions.MultivariateNormal(torch.zeros(n_bottleneck, device=device),
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
        decoder_cnns.append(nn.Conv2d(x_emb_dim, 2*in_channels, 1, 1))
        nn.init.zeros_(decoder_cnns[-1].weight)
        nn.init.zeros_(decoder_cnns[-1].bias)
        decoder_nn = nn.Sequential(*decoder_cnns)

        self.encoder = nf.distributions.NNDiagGaussian(encoder_nn)
        self.decoder = nf.distributions.NNDiagGaussian(decoder_nn)

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
        self.flows = nn.ModuleList(flows)

        # self.nfm = nf.NormalizingFlowVAE(prior, encoder, flows, decoder)

        self.result_cache = easydict.EasyDict({
            'mu_x': None, 'logvar_x': None, 'nll_pixel': None
        })
        self.loss_cache = easydict.EasyDict({
            'nll': None, 'kl_qp': None, 'loss': None
        })

    def write_loss(self, writer, epoch):
        writer.add_scalar('nll', self.loss_cache.nll.item(), epoch)
        writer.add_scalar('kl_qp', self.loss_cache.kl_qp.item(), epoch)
        writer.add_scalar('loss', self.loss_cache.loss.item(), epoch)

    def write_reconstructions(self, writer, epoch):
        mu_x = self.result_cache.mu_x
        logvar_x = self.result_cache.logvar_x
        reconstructions = sample_gaussian(mu_x, logvar_x)
        writer.add_image(
            'Reconstructed Sample',
            torchvision.utils.make_grid(reconstructions.clamp(0., 1.)),
            epoch, dataformats='CHW')
        writer.add_image(
            'Mean',
            torchvision.utils.make_grid(mu_x.clamp(0., 1.)),
            epoch, dataformats='CHW')


    def write_hist(self, writer, epoch):
        # writer.add_histogram('mu_z', self.result_cache.mu_z, epoch)
        # writer.add_histogram('logvar_z', self.result_cache.logvar_z, epoch)
        writer.add_histogram('mu_x', self.result_cache.mu_x, epoch)
        writer.add_histogram('logvar_x', self.result_cache.logvar_x, epoch)

    def forward(self, x, *args, **kwargs):
        # z, log_q, log_p = self.nfm(x, self.num_samples)
        num_samples = self.num_samples
        z, log_q = self.encoder(x, num_samples=num_samples)
        # Flatten batch and sample dim
        z = z.view(-1, *z.size()[2:])
        log_q = log_q.view(-1, *log_q.size()[2:])
        # size: [batch_size*num_samples,]

        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        log_p = self.prior.log_prob(z)  # [batch_size*num_samples]

        mean_std = self.decoder.net(z)
        n_hidden = mean_std.size()[1] // 2
        mu_x = mean_std[:, :n_hidden, ...]
        mu_x = mu_x.view(-1, num_samples, *mu_x.size()[1:])
        logvar_x = mean_std[:, n_hidden:(2 * n_hidden), ...]
        logvar_x = logvar_x.view(-1, num_samples, *logvar_x.size()[1:])
        log_p_x_given_z = - 0.5 * x.size(2) * np.log(2 * np.pi) \
            - 0.5 * torch.sum(logvar_x + (x.unsqueeze(1) - mu_x) ** 2 / logvar_x.exp(), dim=2)
        # [batch_size, num_samples, W, H]

        # log_p_x_given_z = self.decoder.log_prob(x, z)  # [batch_size*num_samples, ]
        # Separate batch and sample dimension again
        # z = z.view(-1, num_samples, *z.size()[1:])

        mean_log_q = torch.mean(log_q)  # mean over [batch_size*num_samples,]
        mean_log_p = torch.mean(log_p)  # mean over [batch_size*num_samples,]

        mean_log_p_x_given_z = torch.sum(log_p_x_given_z) / log_p_x_given_z.shape[0] / log_p_x_given_z.shape[1]
        # mean_log_p_x_given_z = log_p_x_given_z.mean()
        # mean over [batch_size, num_samples]
        # sum over [W, H]
        kl_qp = (mean_log_q - mean_log_p)
        nll = - mean_log_p_x_given_z
        loss = nll + kl_qp
        self.loss_cache.update({
            'nll': nll, 'kl_qp': kl_qp, 'loss': loss
        })
        self.result_cache.update({
            'nll_pixel': -log_p_x_given_z.detach().mean(dim=1).cpu(),
            'mu_x': mu_x.mean(dim=1).detach().cpu(), 'logvar_x': logvar_x.mean(dim=1).detach().cpu()
        })
        return loss

    def get_ano_score(self):
        return self.loss_cache.nll.detach().cpu().numpy()

    def get_ano_loc_score(self):
        return self.result_cache.nll_pixel.detach().cpu().numpy()
