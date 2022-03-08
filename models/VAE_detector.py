import easydict
import torch
import torchvision
from torch import nn
import torch.nn.functional as F


def sample_gaussian(mu, logvar):
    eps = torch.randn_like(mu)
    eps.requires_grad = False
    sample = mu + eps * torch.exp(logvar / 2)
    return sample


class VAE_Detector(nn.Module):

    def __init__(self, config_model, *args, **kwargs):
        super(VAE_Detector, self).__init__()
        image_size = config_model.image_size
        in_channels = config_model.in_channels
        latent_dim = config_model.latent_dim
        param_mlp_hidden = config_model.param_mlp_hidden
        emb_dim = config_model.emb_dim
        x_emb_dim = config_model.x_emb_dim

        # H_out = (H-F+2P)/S+1
        if image_size == 512:
            cnn_params = [
                [in_channels, 8, 11, 1],  # 502
                [8, 16, 6, 1],  # 497
                [16, 32, 9, 2],  # 245
                [32, 32, 7, 5, 1],  # 49
                [32, 16, 9, 3, 1],  # 15
                [16, 16, 3, 2, 1],  # 8
                [16, 32, 4, 2],  # 3
                [32, emb_dim, 3, 1],  # 1
            ]
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
        encoder_cnns = []
        for arg in cnn_params:
            encoder_cnns.append(nn.Conv2d(*arg))
            encoder_cnns.append(nn.BatchNorm2d(arg[1]))
            encoder_cnns.append(nn.ReLU(True))
        encoder_cnns = encoder_cnns[:-2]
        encoder_cnns.append(nn.Flatten())
        self.encoder = nn.Sequential(*encoder_cnns)
        # cnn_params[0] = [x_emb_dim] + list(cnn_params[0])[1:]
        # cnn_params[-1] = [latent_dim] + list(cnn_params[0])[1:]
        cnn_params[0][0] = x_emb_dim
        cnn_params[-1][1] = latent_dim
        decoder_cnns = []
        for arg in reversed(cnn_params):
            decoder_cnns.append(nn.ConvTranspose2d(arg[1], arg[0], *arg[2:]))
            decoder_cnns.append(nn.BatchNorm2d(arg[0]))
            decoder_cnns.append(nn.ReLU(True))
        self.decoder = nn.Sequential(*decoder_cnns[:-2])

        self.mu_z_given_x = nn.Sequential(nn.Linear(emb_dim, param_mlp_hidden),
                                          nn.ReLU(True), nn.Linear(param_mlp_hidden, latent_dim))
        self.logvar_z_given_x = nn.Sequential(nn.Linear(emb_dim, param_mlp_hidden),
                                              nn.ReLU(True), nn.Linear(param_mlp_hidden, latent_dim))

        # self.mu_x_given_z = nn.Linear(x_emb_dim, in_channels)
        # self.logvar_x_given_z = nn.Linear(x_emb_dim, in_channels)
        self.mu_x_given_z = nn.Sequential(nn.Linear(x_emb_dim, param_mlp_hidden),
                                          nn.ReLU(True), nn.Linear(param_mlp_hidden, in_channels))
        self.logvar_x_given_z = nn.Sequential(nn.Linear(x_emb_dim, param_mlp_hidden),
                                              nn.ReLU(True), nn.Linear(param_mlp_hidden, in_channels))

        self.result_cache = easydict.EasyDict({
            'mu_z': None, 'logvar_z': None, 'mu_x': None, 'logvar_x': None, 'nll_pixel': None
        })
        self.loss_cache = easydict.EasyDict({
            'nll': None, 'kl_qp': None, 'loss': None
        })

    def write_loss(self, writer, epoch):
        writer.add_scalar('nll', self.loss_cache.nll.item(), epoch)
        writer.add_scalar('kl_qp', self.loss_cache.kl_qp.item(), epoch)
        writer.add_scalar('loss', self.loss_cache.loss.item(), epoch)

    def write_hist(self, writer, epoch):
        writer.add_histogram('mu_z', self.result_cache.mu_z, epoch)
        writer.add_histogram('logvar_z', self.result_cache.logvar_z, epoch)
        writer.add_histogram('mu_x', self.result_cache.mu_x, epoch)
        writer.add_histogram('logvar_x', self.result_cache.logvar_x, epoch)

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

    def forward(self, x, *args, **kwargs):
        emb = self.encoder(x)
        mu_z = self.mu_z_given_x(emb)
        logvar_z = self.logvar_z_given_x(emb)

        z_sample = sample_gaussian(mu_z, logvar_z)

        x_emb = self.decoder(z_sample[:, :, None, None])
        x_emb = x_emb.permute(0, 2, 3, 1)
        mu_x = self.mu_x_given_z(x_emb).permute(0, 3, 1, 2)
        logvar_x = self.logvar_x_given_z(x_emb).permute(0, 3, 1, 2)

        nll_pixel = 0.5 * logvar_x + 0.5 * (x - mu_x) ** 2 / logvar_x.exp()
        kl_qp_pixel = 0.5 * (logvar_z.exp() + mu_z.pow(2) - 1. - logvar_z)

        nll = nll_pixel.sum() / x.shape[0]
        kl_qp = kl_qp_pixel.sum() / x.shape[0]
        loss = nll + kl_qp
        self.result_cache.update({
            'mu_z': mu_z, 'logvar_z': logvar_z, 'mu_x': mu_x, 'logvar_x': logvar_x, 'nll_pixel': nll_pixel
        })
        self.loss_cache.update({
            'nll': nll, 'kl_qp': kl_qp, 'loss': loss
        })
        return loss

    def get_ano_score(self):
        return self.loss_cache.nll.detach().cpu().numpy()

    def get_ano_loc_score(self):
        return self.result_cache.nll_pixel.detach().cpu().numpy()
