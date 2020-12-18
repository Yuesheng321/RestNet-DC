import torch
import numpy as np


def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch._C._TensorBase):
        return data.data.cpu().numpy()
    if isinstance(data, torch.autograd.Variable):
        return tonumpy(data.data)


def totensor(data, cuda=True):
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    if isinstance(data, torch._C._TensorBase):
        tensor = data
    if isinstance(data, torch.autograd.Variable):
        tensor = data.data
    if cuda:
        tensor = tensor.cuda()
    return tensor


def tovariable(data, cuda=True):
    if isinstance(data, np.ndarray):
        return tovariable(totensor(data, cuda=cuda), cuda=cuda)
    if isinstance(data, torch._C._TensorBase):
        if cuda:
            return torch.autograd.Variable(data).cuda()
        else:
            return torch.autograd.Variable(data)
    if isinstance(data, torch.autograd.Variable):
        return data
    else:
        raise ValueError("UnKnow data type: %s, input should be {np.ndarray,Tensor,Variable}" % type(data))
