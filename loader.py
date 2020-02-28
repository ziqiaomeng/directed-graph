import os
import random

import numpy as np
import torch


def get_successors(dataset, nb_nodes):
    successor = {}
    perm = []
    for node1, node2 in dataset:
        if node1 not in successor.keys():
            successor[node1] = [node2]
        else:
            successor[node1] += [node2]
        perm.append(node1 * nb_nodes + node2)
    return successor, perm


def remap(nodeids_npy1, nodeids_npy2):
    all_nodeids = np.unique(np.concatenate([nodeids_npy1, nodeids_npy2]))
    nb_nodes = all_nodeids.shape[0]
    print('>> number of nodes:{}'.format(nb_nodes))
    dicts = {}
    for ix, nodeid in enumerate(np.unique(all_nodeids)):
        dicts[nodeid] = ix
    remaped_nodeid1 = np.array(list(map(dicts.get, nodeids_npy1.flatten()))).reshape(nodeids_npy1.shape)
    remaped_nodeid2 = np.array(list(map(dicts.get, nodeids_npy2.flatten()))).reshape(nodeids_npy2.shape)
    return remaped_nodeid1, remaped_nodeid2, nb_nodes


def load1(dataset):
    dir = '../node_embedding/dataset/processed/'
    print('===' * 20)
    print('>> dataset is {}'.format(dataset))
    train_file = dir + '{}/{}_train.txt'.format(dataset, dataset)
    test_file = dir + '{}/{}_test.txt'.format(dataset, dataset)
    train_npy = np.loadtxt(train_file, delimiter=' ', dtype=np.int)
    test_npy = np.loadtxt(test_file, delimiter=' ', dtype=np.int)
    train_npy, test_npy, nb_nodes = remap(train_npy, test_npy)

    successors1 = get_successors(train_npy, nb_nodes)
    successors2 = get_successors(test_npy, nb_nodes)

    train_data = torch.from_numpy(train_npy).transpose(0, 1)
    test_data = torch.from_numpy(test_npy).transpose(0, 1)

    return train_data, test_data, nb_nodes, successors1, successors2


def hidden_edges(edge_index, nb_nodes, dataset):
    cache_path = './data/processed/{}/{}.processed.pt'.format(dataset, dataset)
    adj = np.zeros([nb_nodes, nb_nodes])
    adj[edge_index[0], edge_index[1]] = 1
    h_out = []
    h_in = []
    for i in range(adj.shape[0]):
        node_out = np.where(adj[i, :] != 0)[0]
        node_in = np.where(adj[:, i] != 0)[0]
        if len(node_out) != 0:
            r_node_out = random.choice(node_out)
            h_out.append((i, r_node_out))
        if len(node_in) != 0:
            r_node_in = random.choice(node_in)
            h_in.append((r_node_in, i))
    hidden_ = list(set(h_out).union(set(h_in)))
    k1 = k2 = 0
    # 统计没有out link的node, adj, adj.T
    for i in adj:
        if i.max() == 0:
            k1 += 1
    for j in adj.T:
        if j.max() == 0:
            k2 += 1
    print('{} {}'.format(k1, k2))
    # 把选中的hidden从edge_index中隐藏
    hidden_perm = np.array([a[0] * nb_nodes + a[1] for a in hidden_])
    all_perm = edge_index[0] * nb_nodes + edge_index[1]
    train_perm = np.setdiff1d(all_perm, hidden_perm)
    print('number of train links:{}'.format(train_perm.shape[0]))
    print('number of test links:{}'.format(hidden_perm.shape[0]))
    print('number of all links:{}'.format(all_perm.shape[0]))
    # saved
    torch.save((train_perm, h_in, h_out, nb_nodes), cache_path)
    print('>> saved!')
    mask1 = train_perm // nb_nodes
    mask2 = train_perm % nb_nodes
    adj[mask1, mask2] = 0
    return train_perm, h_in, h_out, nb_nodes


def load_feat(raw_file, node_index):
    import pandas as pd
    node_index_list = []
    for key, value in node_index.items():
        node_index_list.append([key, value])
    df_node_index = pd.DataFrame(node_index_list, columns=['id', 'index'])
    f = open(raw_file, 'r')
    feat = []
    for line in f.readlines():
        line_ = line.strip('\n').split('\t')[:-1]
        feat.append(line_)
    df_feat = pd.DataFrame(feat)
    df_feat = df_feat.rename(columns={0: 'id'})
    df = pd.merge(df_node_index, df_feat, how='left', on='id')
    df.sort_values(by='index', ascending=True, inplace=True)
    df_feat_reindex = df.drop(['id', 'index'], axis=1)
    np_feat = df_feat_reindex.astype(np.float64).to_numpy()
    return np_feat


def load(dataset):
    dir = './data/raw/'
    print('===' * 20)
    print('>> dataset is {}'.format(dataset))
    raw_file = dir + '{}/{}.cites'.format(dataset, dataset)
    raw_feat_file = dir + '{}/{}.content'.format(dataset, dataset)
    processed_file = './data/processed/{}/{}.processed.pt'.format(dataset, dataset)
    edge_list = []
    if os.path.isfile(processed_file):
        train_perm, h_in, h_out, nb_nodes = torch.load(processed_file)
        print('>> number of nodes: {}'.format(nb_nodes))
    else:
        with open(raw_file, 'r') as f:
            node_index = {}
            ix = 0
            for line in f.readlines():
                src, dst = line.strip('\n').split('\t')[:2]
                if src not in node_index.keys():
                    node_index[src] = ix
                    ix += 1
                if dst not in node_index.keys():
                    node_index[dst] = ix
                    ix += 1
                edge_list.append([node_index[dst], node_index[src]])
        nb_nodes = len(node_index)
        nb_links = len(edge_list)
        print('>> number of nodes: {}'.format(nb_nodes))
        print('>> number of links: {}'.format(nb_links))
        edge_index = np.transpose(np.array(edge_list))
        train_perm, h_in, h_out, nb_nodes = hidden_edges(edge_index, nb_nodes, dataset)
    feat = load_feat(raw_feat_file, node_index)
    return train_perm, h_in, h_out, nb_nodes, feat


if __name__ == '__main__':
    dataset = 'cora'
    dataset = 'citeseer'
    load(dataset)

