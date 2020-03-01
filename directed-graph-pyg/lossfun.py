import torch
import torch.nn as nn
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Lossfun(nn.Module):
    def __init__(self):
        super(Lossfun, self).__init__()

    def forward(self, sim_matrix, pos_links, labels):
        # location = torch.where(torch.ge(labels, 0) & torch.ge(sim_matrix, 0))
        pos_pred = sim_matrix[pos_links[0], pos_links[1]]
        loss = torch.mean(-torch.log(pos_pred))
        return loss
