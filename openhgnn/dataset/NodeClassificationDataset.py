from dgl.data import DGLDataset
import torch as th
from openhgnn.dataset import BaseDataset, register_dataset
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
from dgl.data.utils import load_graphs, save_graphs
from ogb.nodeproppred import DglNodePropPredDataset


@register_dataset('node_classification')
class NodeClassificationDataset(BaseDataset):
    """
    metric: Accuracy, multi-label f1 or multi-class f1. Default: `accuracy`
    """

    def __init__(self, dataset_name):
        super(NodeClassificationDataset, self).__init__()
        if dataset_name in ['aifb', 'mutag', 'bgs', 'am']:
            self.g, self.category, self.num_classes = self.load_RDF_dgl(dataset_name)
        elif dataset_name in ['acm', 'imdb', 'acm1', 'academic']:
            self.g, self.category, self.num_classes = self.load_HIN(dataset_name)
        elif dataset_name in 'ogbn-mag':
            dataset = DglNodePropPredDataset(name='ogbn-mag')
            split_idx = dataset.get_idx_split()
            self.num_classes = dataset.num_classes
            self.train_idx, self.valid_idx, self.test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
            self.g, self.label = dataset[0]
            self.category = 'paper'# graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)

    def get_labels(self):
        if 'labels' in self.g.nodes[self.category].data:
            labels = self.g.nodes[self.category].data.pop('labels')
        elif 'label' in self.hg.nodes[self.category].data:
            labels = self.g.nodes[self.category].data.pop('label')
        else:
            raise ValueError('label in not in the hg.nodes[category].data')
        return labels

    def get_idx(self,):
        train_mask = self.g.nodes[self.category].data.pop('train_mask')
        test_mask = self.g.nodes[self.category].data.pop('test_mask')
        self.train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
        self.test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
        return self.train_idx, self.test_idx


@register_dataset('rdf_node_classification')
class RDF_NodeCLassification(NodeClassificationDataset):
    def __init__(self, dataset_name):
        super(NodeClassificationDataset, self).__init__()
        self.g, self.category, self.num_classes = self.load_RDF_dgl(dataset_name)

    def load_RDF_dgl(self, dataset):
        # load graph data
        if dataset == 'aifb':
            kg_dataset = AIFBDataset()
        elif dataset == 'mutag':
            kg_dataset = MUTAGDataset()
        elif dataset == 'bgs':
            kg_dataset = BGSDataset()
        elif dataset == 'am':
            kg_dataset = AMDataset()
        else:
            raise ValueError()

        # Load from hetero-graph
        kg = kg_dataset[0]
        category = kg_dataset.predict_category
        num_classes = kg_dataset.num_classes
        return kg, category, num_classes

    def get_idx(self, validation=True):
        train_mask = self.g.nodes[self.category].data.pop('train_mask')
        test_mask = self.g.nodes[self.category].data.pop('test_mask')
        train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
        test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
        if validation:
            val_idx = train_idx[:len(train_idx) // 10]
            train_idx = train_idx[len(train_idx) // 10:]
        else:
            val_idx = train_idx
            train_idx = train_idx
        return train_idx, val_idx, test_idx

    def get_labels(self):
        if 'labels' in self.g.nodes[self.category].data:
            labels = self.g.nodes[self.category].data.pop('labels')
        else:
            raise ValueError('label in not in the hg.nodes[category].data')
        return labels


@register_dataset('hin_node_classification')
class HIN_NodeCLassification(NodeClassificationDataset):
    def __init__(self, dataset_name):
        super(NodeClassificationDataset, self).__init__()
        self.g, self.category, self.num_classes = self.load_HIN(dataset_name)

    def load_HIN(self, dataset):
        if dataset == 'acm':
            data_path = './openhgnn/dataset/acm_graph.bin'
            category = 'paper'
            num_classes = 3
        elif dataset == 'imdb':
            data_path = './openhgnn/dataset/imdb_graph.bin'
            category = 'movie'
            num_classes = 3
        elif dataset == 'acm1':
            data_path = './openhgnn/dataset/acm_graph1.bin'
            category = 'paper'
            num_classes = 3
        elif dataset == 'academic':
            # which is used in HetGNN
            data_path = './openhgnn/dataset/academic.bin'
            category = 'author'
            num_classes = 4
        g, _ = load_graphs(data_path)
        g = g[0].long()
        return g, category, num_classes

    def get_idx(self, validation=True):
        train_mask = self.g.nodes[self.category].data.pop('train_mask')
        test_mask = self.g.nodes[self.category].data.pop('test_mask')
        train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
        test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
        if validation:
            val_idx = train_idx[:len(train_idx) // 10]
            train_idx = train_idx[len(train_idx) // 10:]
        else:
            val_idx = train_idx
            train_idx = train_idx
        return train_idx, val_idx, test_idx

    def get_labels(self):
        if 'labels' in self.g.nodes[self.category].data:
            labels = self.g.nodes[self.category].data.pop('labels')
        elif 'label' in self.g.nodes[self.category].data:
            labels = self.g.nodes[self.category].data.pop('label')
        else:
            raise ValueError('label in not in the hg.nodes[category].data')
        return labels

@register_dataset('ogb_node_classification')
class OGB_NodeCLassification(NodeClassificationDataset):
    def __init__(self, dataset_name):
        super(NodeClassificationDataset, self).__init__()
        if dataset_name == 'ogb-mag':
            dataset = DglNodePropPredDataset(name='ogbn-mag')
        else:
            raise ValueError
        split_idx = dataset.get_idx_split()
        self.num_classes = dataset.num_classes
        self.train_idx, self.valid_idx, self.test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        self.g, self.label = dataset[0]
        self.category = 'paper'  # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)