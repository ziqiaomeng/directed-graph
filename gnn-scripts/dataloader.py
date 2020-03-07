import os
import pickle as pkl
import random

import networkx as nx
import numpy as np
import pandas as pd


def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def set_random(seed):
    np.random.seed(seed)
    random.seed(seed)


class Loader():
    def __init__(self, args):
        self.dataname = args.dataset
        print('==' * 20)
        print('Processing dataset {}'.format(self.dataname))
        self.load()
        self.info()
        self.split()
        self.save()

    def load(self):
        print('1. loading ...')
        self.G = nx.DiGraph()
        if self.dataname in ['cora', 'citeseer', 'pubmed']:
            dataset_dir = '../dataset/raw/{}/ind.{}.graph'.format(self.dataname, self.dataname)
            with open(dataset_dir, 'rb') as f:
                adj_dict = pkl.load(f, encoding='latin1')
                edges = nx.from_dict_of_lists(adj_dict).edges
                edges = np.array(list(edges))
                idx_map = {j: i for i, j in enumerate(np.unique(edges))}
                edges = np.array(list(map(idx_map.get, edges.flatten())), dtype=np.int32).reshape(edges.shape)
                self.G.add_edges_from(edges)
        if self.dataname in ['cit-HepTh', 'cit-HepPh']:
            dataset_dir = '../dataset/raw/{}/out.{}'.format(self.dataname, self.dataname)
            self.df = pd.read_csv(dataset_dir, sep='\t| ', names=['src', 'dst'],
                                  header=None, comment='%', engine='python')
            print('==' * 20)
            print(self.df.head(5))
            print('==' * 20)
            links = self.df.to_numpy()[:, 0:2]
            idx_map = {j: i for i, j in enumerate(np.unique(links))}
            links = np.array(list(map(idx_map.get, links.flatten())), dtype=np.int32).reshape(links.shape)
            self.train_dataset = links
            self.save()
            exit()

        if self.dataname in ['slashdot']:
            dataset_dir = '../dataset/raw/{}/{}.txt'.format(self.dataname, self.dataname)
            links = np.loadtxt(dataset_dir, dtype=int, delimiter='\t', comments='#', )
            idx_map = {j: i for i, j in enumerate(np.unique(links))}
            links = np.array(list(map(idx_map.get, links.flatten())), dtype=np.int32).reshape(links.shape)
            self.G.add_edges_from(links)

    def info(self):
        print('2. info')
        print('>> # number of total nodes: {}'.format(nx.number_of_nodes(self.G)))
        print('>> # number of total links: {}'.format(nx.number_of_edges(self.G)))

    def split(self):
        self.testG = nx.DiGraph()
        hidden_in = []
        hidden_out = []
        for node in self.G.nodes:
            successors = list(self.G.successors(node))
            if len(successors) > 1:
                suc = np.random.choice(successors, 1)[0]
                hidden_out.append((node, suc))
            predecessors = list(self.G.predecessors(node))
            if len(predecessors) > 1:
                pre = np.random.choice(predecessors, 1)[0]
                hidden_in.append((pre, node))
        hidden_pairs = list(set(hidden_out).union(set(hidden_in)))
        self.hidden_in = hidden_in
        self.hidden_out = hidden_out

        for (node1, node2) in hidden_pairs:
            self.G.remove_edge(node1, node2)
        train_edges = list(self.G.edges)
        print('number of nodes in train dataset: {}'.format(nx.number_of_nodes(self.G)))
        print('number of links in train dataset: {}'.format(nx.number_of_edges(self.G)))
        self.train_dataset = np.array(sorted(train_edges, key=lambda x: x[0]))

    def save(self):
        print('==' * 20)
        dataset_dir = '../dataset/processed/{}/'.format(self.dataname)
        mkdir(dataset_dir)
        train_path = dataset_dir + '{}_train.txt'.format(self.dataname)
        hidden_in_path = dataset_dir + '{}_hidden_in.txt'.format(self.dataname)
        hidden_out_path = dataset_dir + '{}_hidden_out.txt'.format(self.dataname)
        mkdir(dataset_dir)

        if to_save:
            np.savetxt(train_path, self.train_dataset, delimiter=' ', fmt='%d')
            if self.dataname in ['cora', 'citeseer', 'pubmed']:
                np.savetxt(hidden_in_path, self.hidden_in, delimiter=' ', fmt='%d')
                np.savetxt(hidden_out_path, self.hidden_out, delimiter=' ', fmt='%d')
            print('>> {} data saved!,dir is {}'.format(self.dataname, dataset_dir))
        else:
            print('>> no saved')


if __name__ == '__main__':
    import argparse
    set_random(1024)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    to_save = True
    args = parser.parse_args()
    loader = Loader(args)
