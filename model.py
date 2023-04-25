import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter, LayerNorm, InstanceNorm2d

class T_cheby_conv(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ,tem_size] - input of all time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]
    '''

    def __init__(self, c_in, c_out, K, Kt):
        super(T_cheby_conv, self).__init__()
        c_in_new = (K) * c_in
        self.conv1 = Conv2d(c_in_new, c_out, kernel_size=(1, Kt), padding=(0, 1),
                            stride=(1, 1), bias=True)
        self.K = K

    def forward(self, x, adj):
        nSample, feat_in, nNode, length = x.shape
        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).cuda()
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 * torch.matmul(adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 0)  # [K,nNode, nNode]
        Lap = Lap.transpose(-1, -2)
        # print(Lap)
        x = torch.einsum('bcnl,knq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out

class TATT(nn.Module):
    def __init__(self, c_in, num_nodes, tem_size):
        super(TATT, self).__init__()
        self.conv1 = Conv2d(c_in, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.conv2 = Conv2d(num_nodes, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.w = nn.Parameter(torch.rand(num_nodes, c_in), requires_grad=True)
        nn.init.xavier_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros(tem_size, tem_size), requires_grad=True)

        self.v = nn.Parameter(torch.rand(tem_size, tem_size), requires_grad=True)
        nn.init.xavier_uniform_(self.v)
        self.bn = BatchNorm1d(tem_size)

    def forward(self, seq):
        c1 = seq.permute(0, 1, 3, 2)  # b,c,n,l->b,c,l,n
        f1 = self.conv1(c1).squeeze()  # b,l,n

        c2 = seq.permute(0, 2, 1, 3)  # b,c,n,l->b,n,c,l
        f2 = self.conv2(c2).squeeze()  # b,c,n

        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b)
        logits = torch.matmul(self.v, logits)
        ##normalization
        # logits=tf_util.batch_norm_for_conv1d(logits, is_training=training,
        #                                   bn_decay=bn_decay, scope='bn')
        # a,_ = torch.max(logits, 1, True)
        # logits = logits - a

        logits = logits.permute(0, 2, 1).contiguous()
        logits = self.bn(logits).permute(0, 2, 1).contiguous()
        coefs = torch.softmax(logits + B, -1)
        return coefs

class ST_BLOCK(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, tem_size, K, Kt):
        super(ST_BLOCK, self).__init__()

        self.conv1 = Conv2d(c_in, c_out, kernel_size=(1, 1),
                            stride=(1, 1), bias=True)
        self.TATT = TATT(c_out, num_nodes, tem_size)
        self.dynamic_gcn = T_cheby_conv(c_out, 2 * c_out, K, Kt)
        self.K = K
        self.time_conv = Conv2d(c_in, c_out, kernel_size=(1, Kt), padding=(0, 1),
                                stride=(1, 1), bias=True)
        # self.bn=BatchNorm2d(c_out)
        self.c_out = c_out
        self.bn = LayerNorm([c_out, num_nodes, tem_size])

    def forward(self, x, supports):
        x_input = self.conv1(x)
        x_1 = self.time_conv(x)
        x_1 = F.leaky_relu(x_1)
        x_1 = F.dropout(x_1, 0.5, self.training)
        x_1 = self.dynamic_gcn(x_1, supports)
        filter, gate = torch.split(x_1, [self.c_out, self.c_out], 1)
        x_1 = torch.sigmoid(gate) * F.leaky_relu(filter)
        x_1 = F.dropout(x_1, 0.5, self.training)
        T_coef = self.TATT(x_1)
        T_coef = T_coef.transpose(-1, -2)
        x_1 = torch.einsum('bcnl,blq->bcnq', x_1, T_coef)
        out = self.bn(F.leaky_relu(x_1) + x_input)
        return out, supports, T_coef

class DGCN_Res(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, week, day, recent, K, Kt):
        super(DGCN_Res, self).__init__()
        tem_size = week + day + recent
        self.block1 = ST_BLOCK(c_in, c_out, num_nodes, tem_size, K, Kt)
        self.block2 = ST_BLOCK(c_out, c_out, num_nodes, tem_size, K, Kt)
        self.bn = BatchNorm2d(c_in, affine=False)
        self.conv1 = Conv2d(c_out, 1, kernel_size=(1, 1), padding=(0, 0),
                            stride=(1, 1), bias=True)
        self.conv2 = Conv2d(c_out, 1, kernel_size=(1, 1), padding=(0, 0),
                            stride=(1, 1), bias=True)
        self.conv3 = Conv2d(c_out, 1, kernel_size=(1, 1), padding=(0, 0),
                            stride=(1, 1), bias=True)
        self.conv4 = Conv2d(c_out, 1, kernel_size=(1, 2), padding=(0, 0),
                            stride=(1, 2), bias=True)

        self.h = Parameter(torch.zeros(num_nodes, num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)

    def forward(self, x_w, x_d, x_r, supports):
        x_w = self.bn(x_w)
        x_d = self.bn(x_d)
        x_r = self.bn(x_r)
        x = torch.cat((x_w, x_d, x_r), -1)
        A = self.h + supports
        d = 1 / (torch.sum(A, -1) + 0.0001)
        D = torch.diag_embed(d)
        A = torch.matmul(D, A)
        A1 = F.dropout(A, 0.5, self.training)

        x, _, _ = self.block1(x, A1)
        x, d_adj, t_adj = self.block2(x, A1)

        x1 = x[:, :, :, 0:12]
        x2 = x[:, :, :, 12:24]
        x3 = x[:, :, :, 24:36]
        x4 = x[:, :, :, 36:60]

        x1 = self.conv1(x1).squeeze()
        x2 = self.conv2(x2).squeeze()
        x3 = self.conv3(x3).squeeze()
        x4 = self.conv4(x4).squeeze()  # b,n,l
        x = x1 + x2 + x3 + x4
        return x, d_adj, A