import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GCNConv

from loader import load
from lossfun import Lossfun
from utils import mkdirs, glorot, zeros, truncated_normal_, prediction_with_recall

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.manual_seed(12345)
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Encoder(torch.nn.Module):
    def __init__(self, nb_nodes, in_channels, out_channels, bias=True):
        super(Encoder, self).__init__()
        self.dropout = 0.01
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv3 = GCNConv(2 * out_channels, 1, cached=True)
        self.conv4 = GCNConv(2 * out_channels, 1, cached=True)
        self.feat = Parameter(torch.randn(nb_nodes, in_channels), requires_grad=False).to(dev)
        self.pos_weights1 = Parameter(torch.rand(1, nb_nodes), requires_grad=True).to(dev)
        self.pos_weights2 = Parameter(torch.rand(nb_nodes, 1), requires_grad=True).to(dev)
        if bias:
            self.bias = Parameter(torch.rand(in_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.feat)
        truncated_normal_(self.pos_weights1, std=5e-2)
        truncated_normal_(self.pos_weights2, std=5e-2)
        zeros(self.bias)

    def update(self, z):
        self.feat = z.clone().detach().requires_grad_(False)

    def get_sim_matrix(self, x, pos_weights1=None, pos_weights2=None):
        sim_matrix = torch.matmul(x, x.transpose(0, 1))
        sim_matrix = sim_matrix-torch.max(sim_matrix, dim=1, keepdim=True)[0]
        x = sim_matrix.exp() * pos_weights1
        pred = x / torch.sum(x, dim=1, keepdim=True)
        pred = pred * pos_weights2 if pos_weights2 else pred
        return pred

    def forward(self, feat, edge_index):
        x = feat
        x = F.relu(self.conv1(x, edge_index))
        pos_weights1 = self.conv3(x, edge_index).sigmoid()
        x = self.conv2(x, edge_index)
        pred = self.get_sim_matrix(x, pos_weights1)
        return pred.sigmoid()

def train(model, feat, train_perm, nb_nodes, h_in, h_out):
    minloss = 100
    top_k = 20
    edge_index = torch.tensor([train_perm // nb_nodes, train_perm % nb_nodes]).long()
    labels = torch.zeros(nb_nodes, nb_nodes)
    labels[edge_index[0], edge_index[1]] = 1
    labels = labels.to(dev)
    edge_index = edge_index.to(dev)
    feat = torch.tensor(feat).float().to(dev)
    model.train()
    model.to(dev)
    lossfun = Lossfun()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    max_epoch = 800
    best_mean_out_recall = 0
    result_data = []
    for epoch in range(1, max_epoch):
        optimizer.zero_grad()
        sim_mat = model(feat, edge_index)
        loss = lossfun(sim_mat, edge_index, labels)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print('Epoch {} mean loss value is {:.8f}'.format(epoch, loss.item()))
        # if epoch == max_epoch - 1:
        if epoch % 10 == 0:
            torch.save(sim_mat, embedding_file)
            with torch.no_grad():
                recall_in, recall_out = prediction_with_recall(sim_mat.detach().cpu().numpy(),
                                                               labels.cpu().numpy(),
                                                               hidden_in=h_in,
                                                               hidden_out=h_out,
                                                               top_k=20)
                print(recall_out)
                # if np.mean(recall_out) > best_mean_out_recall:
                #     best_out_recall = recall_out
                #     best_in_recall = recall_in
                #     best_step = epoch
                # result_data.append([epoch + 1] + list(np.concatenate((recall_in, recall_out))))
                # print("Epoch:", '%04d' % (epoch + 1),
                #       "test_recall_in=", "{:.5f}".format(recall_in[0]),
                #       "test_recall_out=", "{:.5f}".format(recall_out[0]))


def evaluate1(edge_index, suc_train, suc_test, K):
    suc1, perm1 = suc_train
    suc2, perm2 = suc_test
    z = torch.load(embedding_file)
    nb_nodes = z.size(0)
    nodesid = range(nb_nodes)
    src_list = edge_index[0, :].cpu().numpy()
    recall_list = []
    for ix, src in enumerate(np.unique(src_list)):
        print(src)
        if src not in suc1.keys():
            continue
        sim = (z[src] * z).sum(dim=1).sigmoid()
        sorted_sim = sorted(zip(nodesid, sim), key=lambda x: x[1], reverse=True)
        topkid = list(zip(*sorted_sim))[0]
        topk_perm = src * nb_nodes + np.array(topkid)
        print(len(topk_perm[np.where(np.isin(topk_perm, perm2) == True)]))
        exit()
        topk_perm = topk_perm[np.where(np.isin(topk_perm, perm1) == False)][:K]
        hit = np.intersect1d(topk_perm, perm2).shape[0]
        print(hit)
        exit()
        recall = hit * 1.0 / K
        recall_list.append(recall)
        print(ix, recall)
    mr = np.mean(recall_list)
    print('mean recall is {:.4f}'.format(mr))


def main(args):
    channels = 16
    K = 20
    # train_data, test_data, nb_nodes, suc_train, suc_test = load(args.dataset)
    train_perm, h_in, h_out, nb_nodes, feat = load(args.dataset)
    model = Encoder(nb_nodes=nb_nodes, in_channels=feat.shape[1], out_channels=channels)
    train(model, feat, train_perm, nb_nodes, h_in, h_out)
    # evaluate(test_data, suc_train, suc_test, K)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GAE')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--losstype', type=str, default='I')
    args = parser.parse_args()
    embedding_path = './output/{}/{}/'.format(args.dataset, args.losstype)
    mkdirs(embedding_path)
    embedding_file = embedding_path + 'embeddings.{}.mat'.format(args.dataset)
    main(args)
