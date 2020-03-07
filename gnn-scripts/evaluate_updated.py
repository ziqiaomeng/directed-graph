import copy
import os

import numpy as np
import torch


def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def prediction_with_recall(pred, train_label, hidden_in, hidden_out, top_k=20):
    '''
    :param pred: [n,n] matrix, p(i,j) means (i->j)'s probability
    :param train_label: [n,n] matrix
    :param hidden_in:[m,2] list, each row means a edge from hidden_in[i,0] to hidden_in[i,1]
    :param hidden_out: [m,2] list, each row means a edge from hidden_out[i,0] to hidden_out[i,1]
    :param top_k: check top k scores to see if the link in the
    :return:
    hidden_in_recall: float, means the recall rate in hidden in  link
    hidden_out_recall: float, means the recall rate in hidden out link
    '''
    assert len(hidden_in) > 0
    assert len(hidden_out) > 0
    num_correct_in = np.zeros((top_k))

    for link_in in hidden_in:
        s_node = link_in[0]
        e_node = link_in[1]
        train_node = np.where(train_label[:, e_node] > 0)
        p = copy.deepcopy(pred[:, e_node])
        # exclude train link
        p[train_node[0]] = 0
        node_rank = np.argsort(p)[::-1]
        rank_k = np.where(node_rank == s_node)[0][0]
        if rank_k < top_k:
            num_correct_in[rank_k:] += 1

    num_correct_out = np.zeros((top_k))
    for link_out in hidden_out:
        s_node = link_out[0]
        e_node = link_out[1]
        train_node = np.where(train_label[s_node, :] > 0)
        p = copy.deepcopy(pred[s_node, :])
        # exclude train link
        p[train_node[0]] = 0
        node_rank = np.argsort(p)[::-1]
        rank_k = np.where(node_rank == e_node)[0][0]
        if rank_k < top_k:
            num_correct_out[rank_k:] += 1
    return num_correct_in / len(hidden_in), num_correct_out / len(hidden_out)


def load_embedding(filename):
    # load embedding into memory, skip first line
    file = open(filename, 'r')
    lines = file.readlines()[1:]
    file.close()
    # create a map of words to vectors
    embedding = dict()
    for line in lines:
        parts = line.split()
        # key is string word, value is numpy array for vector
        embedding[parts[0]] = np.asarray(parts[1:], dtype='float32')
    return embedding


def get_sim_matrix(raw_embedding, vocab_size):
    # step vocab, store vectors using the Tokenizer's integer mapping
    weight_matrix = np.random.uniform(-0.05, 0.05, (vocab_size, 128))
    for i, (node, value) in enumerate(raw_embedding.items()):
        weight_matrix[int(float(node))] = value
    z = torch.from_numpy(weight_matrix).cuda()
    z = torch.nn.functional.normalize(z)
    sim_matrix = torch.matmul(z, z.transpose(0, 1))
    return sim_matrix.cpu().numpy()


if __name__ == '__main__':

    datasets = ['cora', 'citeseer', 'pubmed']
    models = ['deepwalk', 'node2vec', 'LINE']
    recall_outs = []

    for dataset in datasets:
        for model in models:
            # print('loading network ...')
            train_links = np.loadtxt('../dataset/processed/{}/{}_train.txt'.format(dataset, dataset), dtype=np.int)
            nodes = np.unique(train_links)
            raw_embedding = load_embedding('./output/{}/{}.embeddings'.format(model, dataset))
            num_nodes = np.max(nodes) + 1
            print('{}, {}'.format(len(nodes), len(raw_embedding)))
            sim_matrix = get_sim_matrix(raw_embedding, num_nodes)

            hidden_in = np.loadtxt('../dataset/processed/{}/{}_hidden_in.txt'.format(dataset, dataset), dtype=np.int)
            hidden_out = np.loadtxt('../dataset/processed/{}/{}_hidden_out.txt'.format(dataset, dataset), dtype=np.int)
            # 生成标签
            labels = np.zeros((num_nodes, num_nodes))
            for [id1, id2] in train_links:
                labels[id1, id2] = 1
            r_in, r_out = prediction_with_recall(pred=sim_matrix, train_label=labels, hidden_in=hidden_in,
                                                 hidden_out=hidden_out)
            mkdirs('./output/recall/{}/'.format(dataset))
            np.savetxt('./output/recall/{}/{}_{}_recall_out.txt'.format(dataset, dataset, model), r_out,
                       delimiter='\t', fmt='%.6f')
            np.savetxt('./output/recall/{}/{}_{}_recall_in.txt'.format(dataset, dataset, model), r_in,
                       delimiter='\t', fmt='%.6f')
