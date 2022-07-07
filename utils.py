import torch
import json
import numpy as np
from constants import OBJ_AFFS
import os

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

def check_gpu_mem():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    return t, r, a, f #total, reserved, allocated, free = r-a
    

def read_json(path):
    fp = open(path, 'r')
    data = json.load(fp)
    fp.close()
    return data

def save_checkpoint(state, filename, prefix='checkpoints/'):
    outpath = os.path.join(prefix,filename)
    os.makedirs(prefix, exist_ok=True)
    torch.save(state, outpath)

def load_checkpoint(model, name, prefix='checkpoints/'):
    checkpoint = torch.load(os.path.join(prefix, name))
    model.load_state_dict(checkpoint['model'])
    return model

def aff_labels_from_objnames(objnames):
    affs = []
    for name in objnames:
        affs.append(OBJ_AFFS[name])
    return np.array(affs)
