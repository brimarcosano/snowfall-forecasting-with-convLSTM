import os
import torch


def torch_save(obj, path):
    torch.save(obj, path)

def load_torch_save(path):
    return torch.load(path)

def datasets_exist(path):
    return os.path.isfile(path)
