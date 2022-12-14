import setup_path

import os
import torch as th

device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
FloatTensor = th.FloatTensor if not th.cuda.is_available() else th.cuda.FloatTensor
ByteTensor = th.ByteTensor if not th.cuda.is_available() else th.cuda.ByteTensor


def get_folder(folder, root='trained', has_log=False, has_graph=False, has_model=False, allow_exist=False):
    folder = os.path.join(root, folder)
    if os.path.exists(folder):
        if not allow_exist:
            raise FileExistsError

    if has_log:
        log_path = os.path.join(folder, 'logs/')
        if not os.path.exists(log_path):
            os.makedirs(log_path)
    else:
        log_path = None

    if has_graph:
        graph_path = os.path.join(folder, 'graph/')
        if not os.path.exists(graph_path):
            os.makedirs(graph_path)
    else:
        graph_path = None

    if has_model:
        model_path = os.path.join(folder, 'model/')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
    else:
        model_path = None

    return {'folder': folder,
            'log_path': log_path,
            'graph_path': graph_path,
            'model_path': model_path}


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(source_param.data)
