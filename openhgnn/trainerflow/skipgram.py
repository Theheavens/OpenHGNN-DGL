import argparse
import copy
import dgl
import torch as th
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from openhgnn.models import build_model

from . import BaseFlow, register_flow
from ..tasks import build_task
from torch.utils.data import IterableDataset, DataLoader
from openhgnn.utils.sampler import SkipGramBatchSampler, HetGNNCollator, NeighborSampler


@register_flow("skipgram")
class SkipGram(BaseFlow):
    """SkipGram flows."""

    def __init__(self, args):
        super(SkipGram, self).__init__(args)

        self.args = args
        self.model_name = args.model
        self.device = args.device
        self.task = build_task(args)

        self.hg = self.task.get_graph().to(self.device)
        self.args.num_classes = self.task.dataset.num_classes

        self.model = build_model(self.model_name).build_model_from_args(self.args, self.hg)
        #self.model.set_device(self.device)
        self.model.cuda()
        self.evaluator = self.task.get_evaluator('f1_lr')

        self.optimizer = (
            torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            if not hasattr(self.model, "get_optimizer")
            else self.model.get_optimizer(args)
        )

        self.model = self.model.to(self.device)
        self.patience = args.patience
        self.max_epoch = args.max_epoch

    def preprocess(self):
        self.category = self.task.dataset.category
        self.labels = self.task.get_labels()
        if self.args.task == 'node_classification':
            self.train_idx, self.val_idx, self.test_idx = self.task.get_idx()
        if self.args.mini_batch_flag:
            if self.args.model == 'HetGNN':
                from openhgnn.utils.dgl_graph import hetgnn_graph
                hetg = hetgnn_graph(self.hg, self.args.dataset)
                self.hg = self.hg.to('cpu')
                self.het_graph = hetg.get_hetgnn_graph(self.args.rw_length, self.args.rw_walks, self.args.rwr_prob).to('cpu')

                batch_sampler = SkipGramBatchSampler(self.hg, self.args.batch_size, self.args.window_size)
                neighbor_sampler = NeighborSampler(self.het_graph, self.hg.ntypes, batch_sampler.num_nodes, self.args.device)
                collator = HetGNNCollator(neighbor_sampler, self.hg)
                dataloader = DataLoader(
                    batch_sampler,
                    collate_fn=collator.collate_train,
                    num_workers=self.args.num_workers)
                self.dataloader_it = iter(dataloader)
                self.hg = self.hg.to(self.args.device)
                self.het_graph = self.het_graph.to(self.args.device)
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
                loss = self._mini_train_step()
            else:
                loss = self._full_train_setp()

            self._test_step()

            if loss <= min_loss:
                best_model = copy.deepcopy(self.model)
                min_loss = loss
                patience = 0
            else:
                patience += 1
                if patience == self.patience:
                    epoch_iter.close()
                    break
        self.model = best_model
        metrics = self._test_step()
        print(f"Test accuracy = {metrics:.4f}")
        return dict(metrics=metrics)

    def loss_fn(self, pos_score, neg_score):
        # an example hinge loss
        loss = []
        for i in pos_score:
            loss.append(F.logsigmoid(pos_score[i]))
            loss.append(F.logsigmoid(-neg_score[i]))
        loss = th.cat(loss)
        return -loss.mean()

    def _mini_train_step(self, ):
        self.model.train()
        all_loss = 0
        for batch_id in range(self.args.batches_per_epoch):
            positive_graph, negative_graph, blocks = next(self.dataloader_it)
            blocks = [b.to(self.device) for b in blocks]
            positive_graph = positive_graph.to(self.device)
            negative_graph = negative_graph.to(self.device)
            # we need extract multi-feature
            x = self.model(blocks[0])
            loss = self.loss_fn(self.ScorePredictor(positive_graph, x), self.ScorePredictor(negative_graph, x))
            all_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return all_loss/self.args.batches_per_epoch

    def ScorePredictor(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(
                    dgl.function.u_dot_v('x', 'x', 'score'), etype=etype)
            return edge_subgraph.edata['score']

    def _full_train_setp(self):
        self.model.train()

        logits = self.model(self.he)[self.category]
        loss = self.loss_fn(logits[self.train_idx], self.labels[self.train_idx])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        pass

    def _test_step(self, logits=None):
        self.model.eval()
        with torch.no_grad():
            h = self.model.extract_feature(self.hg, self.hg.ntypes)
            logits = logits if logits else self.model(self.het_graph, h)[self.category].to('cpu')
        if self.args.task == 'node_classification':

            #loss = self.loss_fn(logits[self.train_idx], self.labels[self.test_idx])
            metric = self.evaluator(logits, self.labels, self.train_idx, self.test_idx)
            return metric



