from torch.autograd import Variable
import torch
import numpy as np


def to_var(x):
    return Variable(torch.from_numpy(np.asarray(x).astype('float32')).cuda())


def to_np(x):
    return x.data.cpu().numpy()
