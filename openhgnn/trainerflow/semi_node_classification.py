import argparse
import copy
import dgl
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from openhgnn.models import build_model, MLP_follow_model


from . import BaseFlow, register_flow
from ..tasks import build_task
from ..utils import extract_embed


@register_flow("semi_supervised_node_classification")
class SemiSupervisedNode(BaseFlow):
    """Node classification flows."""

    def __init__(self, args):
        super(SemiSupervisedNode, self).__init__(args)

        self.args = args
        self.model_name = args.model
        self.device = args.device
        self.task = build_task(args)

        self.hg = self.task.get_graph().to(self.device)
        self.args.num_classes = self.task.dataset.num_classes

        self.model = build_model(self.model_name).build_model_from_args(self.args, self.hg)
        if not hasattr(args, 'out_dim') or args.out_dim == self.args.num_classes:
            pass
        else:
            self.model = MLP_follow_model(self.model, args.out_dim, args.num_classes)
        self.model = self.model.to(self.device)

        self.evaluator = self.task.get_evaluator('acc')
        self.loss_fn = self.task.get_loss_fn()
        self.optimizer = (
            torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            if not hasattr(self.model, "get_optimizer")
            else self.model.get_optimizer(args)
        )

        self.patience = args.patience
        self.max_epoch = args.max_epoch



    def preprocess(self):
        self.category = self.task.dataset.category
        self.train_idx, self.val_idx, self.test_idx = self.task.get_idx()
        self.labels = self.task.get_labels()
        if self.args.mini_batch_flag:
            # sampler = dgl.dataloading.MultiLayerNeighborSampler([self.args.fanout] * self.args.n_layers)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.args.n_layers)
            self.loader = dgl.dataloading.NodeDataLoader(
                self.hg.to('cpu'), {self.category: self.train_idx.to('cpu')}, sampler,
                batch_size=self.args.batch_size, shuffle=True, num_workers=0)
        return

    def train(self):
        self.preprocess()
        patience = 0
        best_score = 0
        best_loss = np.inf
        max_score = 0
        min_loss = np.inf
        best_model = copy.deepcopy(self.model)
        epoch_iter = tqdm(range(self.max_epoch))
        for epoch in epoch_iter:
            if self.args.mini_batch_flag:
                self._mini_train_step()
            else:
                self._full_train_setp()

            acc, losses = self._test_step()

            train_acc = acc["train"]
            val_acc = acc["val"]
            val_loss = losses["val"]
            epoch_iter.set_description(
                f"Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, ValLoss:{val_loss: .4f}"
            )
            if val_loss <= min_loss or val_acc >= max_score:
                if val_loss <= best_loss:  # and val_acc >= best_score:
                    best_loss = val_loss
                    best_score = val_acc
                    best_model = copy.deepcopy(self.model)
                min_loss = np.min((min_loss, val_loss.cpu()))
                max_score = np.max((max_score, val_acc))
                patience = 0
            else:
                patience += 1
                if patience == self.patience:
                    epoch_iter.close()
                    break
        print(f"Valid accurracy = {best_score: .4f}")
        self.model = best_model
        test_acc, _ = self._test_step(split="test")
        val_acc, _ = self._test_step(split="val")
        print(f"Test accuracy = {test_acc:.4f}")
        return dict(Acc=test_acc, ValAcc=val_acc)

    def _mini_train_step(self,):
        for i, (input_nodes, seeds, blocks) in enumerate(self.loader):
            blocks = [blk.to(self.device) for blk in blocks]
            seeds = seeds[self.category]  # out_nodes, we only predict the nodes with type "category"
            # batch_tic = time.time()
            emb = extract_embed(self.model.embed_layer(), input_nodes)
            lbl = self.labels[seeds].to(self.device)

            logits = self.model(blocks, emb)[self.category]
            loss = self.loss_fn(logits, lbl)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return

    def _full_train_setp(self):
        logits = self.model(self.hg)[self.category]
        loss = self.loss_fn(logits[self.train_idx], self.labels[self.train_idx])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        pass

    def _test_step(self, split=None, logits=None):
        self.model.eval()
        with torch.no_grad():
            logits = logits if logits else self.model(self.hg)[self.category].to('cpu')
        if split == "train":
            mask = self.train_idx
        elif split == "val":
            mask = self.val_idx
        elif split == "test":
            mask = self.test_idx
        else:
            mask = None

        if mask is not None:
            loss = self.loss_fn(logits[mask], self.labels[mask])
            metric = self.evaluator(self.labels[mask], logits[mask])
            return metric, loss
        else:
            masks = {'train': self.train_idx, 'val': self.val_idx, 'test': self.test_idx}
            metrics = {key: self.evaluator(self.labels[mask], logits[mask]) for key, mask in masks.items()}
            losses = {key: self.loss_fn(logits[mask], self.labels[mask]) for key, mask in masks.items()}
            return metrics, losses


