import logging
import os
import time

import numpy as np
import torch.optim.lr_scheduler
from easydict import EasyDict as edict
from tensorboardX import SummaryWriter

from utils.arg_helper import set_seed_and_logger, get_config, parse_arguments, process_config, edict2dict
from utils.load_helper import log_model_params, get_model, load_data


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
            model.write_hist(writer, epoch)
            batch_loss_ls.append(loss.item())
            loss.backward()
            optimizer.step()
        epoch_loss = np.mean(batch_loss_ls)
        epoch_times.append(time.time() - time_st)
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
        avg_time = np.mean(epoch_times)
        eta = (config.train.epochs - epoch - 1) * avg_time
        logger.info(f'epoch={epoch}, loss={epoch_loss}, best_epoch={best_epoch}, min_loss={min_loss}, '
                    f'time={epoch_times[-1]:.2f}, eta={time.strftime("%H:%M:%S", time.gmtime(eta))}')


def train_main(config):
    set_seed_and_logger(config)
    model = get_model(config)
    log_model_params(config, model)
    optimizer = torch.optim.Adam(model.parameters(), **config.train.optim)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.train.lr_dacey)
    writer = SummaryWriter(logdir=config.tensorboard_dir)
    data = load_data(config)
    fit(config, data, model, optimizer, scheduler, writer)


if __name__ == "__main__":
    args = parse_arguments('configs/mvtech_train.yaml')
    ori_config_dict = get_config(args.config_file)
    config_dict = edict(ori_config_dict.copy())
    process_config(config_dict)
    print(config_dict)
    # noinspection PyTypeChecker
    train_main(config_dict)
