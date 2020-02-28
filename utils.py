import argparse
import copy
import math
import os

import gensim
import numpy as np
import torch


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

def xavier_init(shape):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = np.random.uniform(low=-init_range, high=init_range, size=shape)
    return torch.Tensor(initial)


def prediction_with_recall(pred, train_label, hidden_in, hidden_out, top_k=1):
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


def convert(file_path):
    z = torch.load(file_path, map_location=torch.device('cpu')).detach().numpy()
    nodeids = list(range(z.shape[0]))
    nfile_path = file_path + '.txt'
    f = open(nfile_path, 'w')
    header = '{} {}\n'.format(z.shape[0], z.shape[1])
    f.write(header)
    for nodeid, vector in zip(nodeids, z):
        f.write('{} {}\n'.format(nodeid, ' '.join([str(v) for v in (vector)])))
    f.close()
    print('saved!')
    embedding = gensim.models.KeyedVectors.load_word2vec_format(nfile_path)
    print('success')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--losstype', type=str, default='I')
    args = parser.parse_args()
    embedding_path = './output/{}/{}/'.format(args.dataset, args.losstype)
    embedding_file = embedding_path + 'embeddings.{}.pt'.format(args.dataset)
    convert(embedding_file)
