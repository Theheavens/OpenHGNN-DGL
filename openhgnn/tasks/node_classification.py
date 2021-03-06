import argparse
import copy
import dgl
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from openhgnn.models import build_model

from . import BaseTask, register_task
from ..dataset import build_dataset
from ..utils import Evaluator


@register_task("node_classification")
class NodeClassification(BaseTask):
    """Node classification tasks."""
    def __init__(self, args):
        super(NodeClassification, self).__init__(args)
        self.dataset = build_dataset(args.dataset, 'node_classification')
        # self.evaluator = Evaluator()
        if hasattr(args, 'validation'):
            self.train_idx, self.val_idx, self.test_idx = self.dataset.get_idx(args.validation)
        else:
            self.train_idx, self.val_idx, self.test_idx = self.dataset.get_idx()
        self.evaluator = Evaluator(args.seed)

    def get_graph(self):
        return self.dataset.g

    def get_loss_fn(self):
        return F.cross_entropy

    def get_evaluator(self, name):
        if name == 'acc':
            return self.evaluator.cal_acc
        elif name == 'f1_lr':
            return self.evaluator.nc_with_LR

    def get_idx(self):
        return self.train_idx, self.val_idx, self.test_idx

    def get_labels(self):
        return self.dataset.get_labels()

