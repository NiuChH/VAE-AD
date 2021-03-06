import logging
import os
import pickle
import time

import numpy as np
import torch.optim.lr_scheduler
from easydict import EasyDict as edict
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
from tensorboardX import SummaryWriter

from models import VT_AE_Detector, VAE_Detector, VAE_NF_Detector
from utils.arg_helper import set_seed_and_logger, get_config, parse_arguments, process_config, edict2dict
from utils.load_helper import log_model_params, get_model, load_data
from utils.utility_fun import Filter, plot


def fit(config, data, model, optimizer, scheduler, writer):
    logger = logging.getLogger(config.logger_name)
    dev = config.dev
    model.train()
    min_loss = 1e10
    best_epoch = -1
    epoch_times = []
    for epoch in range(config.train.epochs):
        batch_loss_ls = []
        time_st = time.time()
        for x_batch, _ in data.train_loader:
            x_batch = x_batch.to(dev)
            model.zero_grad()
            loss = model.forward(x_batch)
            model.write_loss(writer, epoch)
            batch_loss_ls.append(loss.item())
            loss.backward()
            optimizer.step()
        epoch_loss = np.mean(batch_loss_ls)
        if epoch % 5 == 0:
            model.write_hist(writer, epoch)
            model.write_reconstructions(writer, epoch)
        writer.add_scalar('Mean Epoch loss', epoch_loss, epoch)
        writer.close()

        if epoch_loss < min_loss:
            min_loss = epoch_loss
            best_epoch = epoch
        if best_epoch == epoch or config.train.save_all_models:
            to_save = {
                'model': model.state_dict(),
                'config': edict2dict(config),
                'epoch': epoch,
                'epoch_loss': epoch_loss,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            torch.save(to_save, os.path.join(config.model_save_dir, f"model.pth"))
        epoch_times.append(time.time() - time_st)
        avg_time = np.mean(epoch_times)
        eta = (config.train.epochs - epoch - 1) * avg_time
        logger.info(f'epoch={epoch:04d}, loss={epoch_loss:.4f}, best_epoch={best_epoch:04d}, min_loss={min_loss:.4f}, '
                    + f'time={epoch_times[-1]:.2f}, eta={time.strftime("%H:%M:%S", time.gmtime(eta))}')


def train_main(config):
    model = get_model(config.model, train=True, device=config.dev)
    log_model_params(config, model)
    optimizer = torch.optim.Adam(model.parameters(), **config.train.optim)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.train.lr_dacey)
    writer = SummaryWriter(logdir=config.tensorboard_dir)
    data = load_data(config)
    model = model.to(config.dev)
    fit(config, data, model, optimizer, scheduler, writer)


def evaluate_auc(config, data, model, writer):
    model.eval()
    patch_size = config.dataset.patch_size
    all_score_ls = [[], []]
    loc_ls = []
    mask_ls = []
    for dl, label in zip((data.test_norm_loader, data.test_anom_loader), (0, 1)):
        for idx, (img_b, mask_b) in enumerate(dl):
            loss = model.forward(img_b.to(config.dev), test=True)
            ano_score = model.get_ano_score()
            all_score_ls[label].append(ano_score)
            if config.dataset.name.lower() != 'mvtec':
                continue
            if isinstance(model, VT_AE_Detector):
                ano_loc = model.get_ano_loc_score()
                m = torch.nn.UpsamplingBilinear2d((512, 512))
                norm_score = ano_loc.reshape(-1, 1, 512 // patch_size, 512 // patch_size)
                score_map = m(torch.tensor(norm_score))
                score_map = Filter(score_map, type=0)
            elif isinstance(model, VAE_Detector) or isinstance(model, VAE_NF_Detector):
                ano_loc = model.get_ano_loc_score()
                score_map = Filter(ano_loc)
            else:
                continue
            loc_ls.append(score_map)  # Storing all score maps
            mask_ls.append(mask_b.squeeze(0).squeeze(0).cpu().numpy())
            if idx % 5 == 0:
                plot(img_b, mask_b, score_map[0][0])

    # PRO Score
    if config.dataset.name.lower() != 'mvtec' or len(loc_ls) == 0:
        PRO_score = 0.0
    else:
        loc_np = np.asarray(loc_ls).flatten()
        mask_np = np.asarray(mask_ls).flatten()
        PRO_score = roc_auc_score(mask_np, loc_np, max_fpr=0.3)

    # Image Anomaly Classification Score (AUC)
    roc_scores = np.concatenate(all_score_ls)
    roc_targets = np.concatenate((np.zeros(len(all_score_ls[0])), np.ones(len(all_score_ls[1]))))

    fpr, tpr, thresholds = roc_curve(roc_targets, roc_scores)
    pickle.dump((fpr, tpr, thresholds), open(os.path.join(config.save_dir, 'roc.pickle'), 'wb'))

    AUC_Score_total = roc_auc_score(roc_targets, roc_scores)

    # AUC Precision Recall Curve
    precision, recall, thres = precision_recall_curve(roc_targets, roc_scores)
    pickle.dump((precision, recall, thres), open(os.path.join(config.save_dir, 'pr.pickle'), 'wb'))
    AUC_PR = auc(recall, precision)

    return PRO_score, AUC_Score_total, AUC_PR


def test_main(config):
    save_dict = torch.load(config.model.load_path, map_location=config.dev)
    ori_config = edict(save_dict['config'])

    model = get_model(ori_config.model, device=config.dev, train=False).to(config.dev)
    log_model_params(config, model)
    model.load_state_dict(save_dict['model'])
    writer = SummaryWriter(logdir=config.tensorboard_dir)
    # optimizer = torch.optim.Adam(model.parameters(), **config.train.optim)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.train.lr_dacey)
    # optimizer.load_state_dict(save_dict['optimizer'])
    # scheduler.load_state_dict(save_dict['scheduler'])
    data = load_data(config)
    with torch.no_grad():
        PRO, AUC, AUC_PR = evaluate_auc(config, data, model, writer)
    print(f'PRO Score: {PRO} \nAUC Total: {AUC} \nPR_AUC Total: {AUC_PR}')


if __name__ == "__main__":
    args = parse_arguments('configs/mnist_train_vae_nf.yaml')
    config_dict = get_config(args.config_file)
    process_config(config_dict)
    set_seed_and_logger(config_dict)
    if args.testing:
        test_main(config_dict)
    else:
        train_main(config_dict)
