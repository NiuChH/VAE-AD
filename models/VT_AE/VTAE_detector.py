import easydict
import torchvision
from torch import nn
import torch.nn.functional as F

import pytorch_ssim
from models.VT_AE import mdn1
from models.VT_AE.VT_AE import VT_AE


class VT_AE_Detector(nn.Module):

    def __init__(self, config_model, train=True):
        super(VT_AE_Detector, self).__init__()
        self.vt_ae = VT_AE(train=train, **config_model.vt_ae)
        self.gmm = mdn1.MDN(**config_model.mdn)
        self.lambda_mse = float(config_model.lambda_mse)
        self.lambda_ssim = float(config_model.lambda_ssim)
        self.ssim_loss_func = pytorch_ssim.SSIM(**config_model.ssim)  # SSIM Loss
        self.result_cache = easydict.EasyDict({
            'vector': None, 'reconstructions': None,
            'pi': None, 'mu': None, 'sigma': None
        })
        self.loss_cache = easydict.EasyDict({
            'mse_loss': None, 'ssim_loss': None, 'll_loss': None, 'sum_loss': None
        })

    def forward(self, x, test=False):
        vector, reconstructions = self.vt_ae(x)
        pi, mu, sigma = self.gmm(vector)  # vector: b,H*W/p^2,vt_ae.dim
        self.result_cache.update({
            'vector': vector, 'reconstructions': reconstructions,
            'pi': pi, 'mu': mu, 'sigma': sigma
        })

        mse_loss = F.mse_loss(reconstructions, x, reduction='mean')  # Rec Loss
        ssim_loss = -self.ssim_loss_func(x, reconstructions)  # SSIM loss for structural similarity
        ll_loss = mdn1.mdn_loss_function(vector, mu, sigma, pi, test=test)  # MDN loss for gaussian approximation

        sum_loss = self.lambda_mse * mse_loss + self.lambda_ssim * ssim_loss + ll_loss
        self.loss_cache.update({
            'mse_loss': mse_loss, 'ssim_loss': ssim_loss, 'll_loss': ll_loss, 'sum_loss': sum_loss
        })
        return sum_loss

    def get_ano_score(self):
        # TODO: - or +
        score = self.loss_cache.mse_loss - self.loss_cache.ssim_loss + self.loss_cache.ll_loss.max()
        return score.detach().cpu().numpy()

    def get_ano_loc_score(self):
        return self.loss_cache.ll_loss.detach().cpu().numpy()

    def write_loss(self, writer, epoch):
        writer.add_scalar('recon-loss', self.loss_cache.mse_loss.item(), epoch)
        writer.add_scalar('ssim loss', self.loss_cache.ssim_loss.item(), epoch)
        writer.add_scalar('Gaussian loss', self.loss_cache.ll_loss.item(), epoch)

    def write_hist(self, writer, epoch):
        writer.add_histogram('Vectors', self.result_cache.vector, epoch)
        writer.add_histogram('Pi', self.result_cache.pi, epoch)
        writer.add_histogram('Variance', self.result_cache.sigma, epoch)
        writer.add_histogram('Mean', self.result_cache.mu, epoch)

    def write_reconstructions(self, writer, epoch):
        writer.add_image(
            'Reconstructed Image',
            torchvision.utils.make_grid(self.result_cache.reconstructions),
            epoch, dataformats='CHW')

