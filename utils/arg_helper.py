import argparse
import logging
import os
import random
import sys
import time
from pprint import pformat

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict


def parse_arguments(default_config="configs/train.yaml"):
    parser = argparse.ArgumentParser(description="Running Experiments")
    parser.add_argument(
        '-c',
        '--config_file',
        type=str,
        default=default_config,
        help="Path of config file")
    parser.add_argument(
        '-l',
        '--log_level',
        type=str,
        default='INFO',
        help="Logging Level, one of: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    parser.add_argument('-m', '--comment', type=str,
                        default="", help="A single line comment for the experiment")
    args = parser.parse_args()

    return args


def get_config(config_file):
    print('config_file:', config_file)
    """ Construct and snapshot hyper parameters """
    config = edict(yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader))
    process_config(config, comment=config.comment)
    return config


def process_config(config, comment=''):
    # create hyper parameters
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config.dev = dev
    config.run_id = str(os.getpid())
    config.folder_name = '_'.join([
        config.model.name, comment.replace(' ', '_'),
        time.strftime('%b-%d-%H-%M-%S')
    ])

    if 'save_dir' not in config:
        config.save_dir = os.path.join(config.exp_dir, config.exp_name, config.folder_name)
    if 'result_dir' not in config:
        config.result_dir = os.path.join(config.save_dir, 'results')
    if 'model_save_dir' not in config:
        config.model_save_dir = os.path.join(config.save_dir, 'models')
    config.tensorboard_dir = os.path.join(config.save_dir, 'runs')

    # snapshot hyper-parameters and code
    mkdir(config.exp_dir)
    mkdir(config.save_dir)
    mkdir(config.result_dir)
    mkdir(config.model_save_dir)

    # mkdir(config.save_dir + '/code')
    # os.system('cp ./*py ' + config.save_dir + '/code')
    # os.system('cp -r ./model ' + config.save_dir + '/code')
    # os.system('cp -r ./utils ' + config.save_dir + '/code')

    save_name = os.path.join(config.save_dir, 'config.yaml')
    yaml.dump(edict2dict(config), open(save_name, 'w'), default_flow_style=False)
    print('config: \n' + pformat(config))


def edict2dict(edict_obj):
    dict_obj = {}

    for key, vals in edict_obj.items():
        if isinstance(vals, edict):
            dict_obj[key] = edict2dict(vals)
        else:
            dict_obj[key] = vals

    return dict_obj


def mkdir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


def set_seed_and_logger(config):
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.RandomState(config.seed)

    log_file = os.path.join(config.save_dir, config.log_level.lower() + ".log")

    FORMAT = config.comment + '| %(asctime)s %(message)s'
    fh = logging.FileHandler(log_file)
    sh = logging.StreamHandler(sys.stdout)
    fh.setFormatter(logging.Formatter(FORMAT))
    sh.setFormatter(logging.Formatter(FORMAT))

    logger = logging.getLogger(config.logger_name)
    logger.setLevel(config.log_level)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info('EXPERIMENT BEGIN: ' + config.comment)
    logger.info('logging into %s', log_file)
    return logger
