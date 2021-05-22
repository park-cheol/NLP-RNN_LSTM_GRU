import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.
def batchify(data, bsz):
    nbatch = data.size(0) //bsz
    # Trim off any extra elements that wouldn't cleanly fit
    data = data.narrow(0, 0, nbatch * bsz)
    # torch.narrow(input, dim, start, length) → Tensor
    data = data.view(bsz, -1).t().contiguous()

    return data

# https://discuss.pytorch.org/t/help-clarifying-repackage-hidden-in-word-language-model/226/5
def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    # sequence가 끝난 후에 backprop 원하지 않으므로 detach()
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h) # todo 이런경우 어떻게동작



















