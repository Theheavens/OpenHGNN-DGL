# @Time   : 2021/1/28
# @Author : Tianyu Zhao
# @Email  : tyzhao@bupt.edu.cn


import argparse
from openhgnn.config import Config
from openhgnn.start import OpenHGNN


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='CompGCN', type=str, help='name of models')
    parser.add_argument('--task', '-t', default='link_prediction', type=str, help='name of task')
    # link_prediction / node_classification
    parser.add_argument('--dataset', '-d', default='wn18', type=str, help='name of datasets')
    # ['wn18', 'FB15k', 'FB15k-237']
    parser.add_argument('--gpu', '-g', default='0', type=int, help='-1 means cpu')
    args = parser.parse_args()

    config_file = ["./openhgnn/config.ini"]
    config = Config(file_path=config_file, model=args.model, dataset=args.dataset, gpu=args.gpu)
    config.task = args.task
    OpenHGNN(args=config)
