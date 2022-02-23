import logging

from utils.mvtech import Mvtec
from models import *


def log_model_params(model):
    param_strings = []
    max_string_len = 75
    for name, param in model.named_parameters():
        if param.requires_grad:
            line = '.' * max(0, max_string_len - len(name) - len(str(list(param.size()))))
            param_strings.append(f"{name} {line} {list(param.size())}")
    param_string = '\n'.join(param_strings)
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info('model: \n' + str(model))
    logging.info(f"Parameters: \n"
                 f"{'-' * max_string_len} \n"
                 f"{param_string} \n"
                 f"{'-' * max_string_len} \n"
                 f"Parameters Count: {total_params}, Trainable: {total_trainable_params} \n"
                 f"{'-' * max_string_len}")


def get_model(config, **kwargs):
    model = eval(config.model.name)(config)
    model.to(config.dev)
    return model


def load_data(config):
    if config.dataset.name.lower() == 'mvtec':
        return Mvtec(config.dataset)
    ...

