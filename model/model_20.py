import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import os
import numpy as np

"""
在model_19基础上引入 Non-Local Block 捕捉全局特征；
"""


def drop_edge(adj, drop_prob=0.1):
    """Randomly drop edges in the adjacency matrix."""
    if drop_prob <= 0.0:
        return adj
    mask = torch.rand_like(adj, dtype=torch.float32) > drop_prob
    return adj * mask


class NonLocalBlock(nn.Module):
    """
    Non-Local Block to capture global features
    """

    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = in_channels

        # Define the layers for Non-Local operation
        self.theta = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.g = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self.W = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x: [B, C, N]
        """
        batch_size, channels, num_nodes = x.size()

        # Compute theta, phi, and g
        theta = self.theta(x)  # [B, C, N]
        phi = self.phi(x)  # [B, C, N]
        g = self.g(x)  # [B, C, N]

        # Compute pairwise similarity matrix
        theta_phi = torch.matmul(theta.transpose(1, 2), phi)  # [B, N, N]
        attention = self.softmax(theta_phi)  # [B, N, N]

        # Apply attention on g
        out = torch.matmul(attention, g.transpose(1, 2))  # [B, N, C]
        out = out.transpose(1, 2)  # [B, C, N]

        # Apply the final weight matrix
        out = self.W(out)

        return out + x  # Residual connection


class GraphConvolution(nn.Module):
    """
    Simple GCN layer with Residual Connection
    """

    def __init__(self, in_features, out_features, mat_path, bias=True, drop_prob=0.1):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # Load adjacency matrix
        adj_mat = np.load(mat_path)
        self.register_buffer('adj', torch.from_numpy(adj_mat))

        # DropEdge probability
        self.drop_prob = drop_prob

        # Residual connection layer
        if in_features != out_features:
            self.residual_layer = nn.Linear(in_features, out_features, bias=False)
        else:
            self.residual_layer = None

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        b, n, c = input.shape

        # Apply DropEdge to adjacency matrix
        adj = drop_edge(self.adj, drop_prob=self.drop_prob)  # Apply drop_edge

        support = torch.bmm(input, self.weight.unsqueeze(0).repeat(b, 1, 1))
        output = torch.bmm(adj.unsqueeze(0).repeat(b, 1, 1), support)

        if self.bias is not None:
            output += self.bias

        # Residual connection
        if self.residual_layer is not None:
            residual = self.residual_layer(input)
        else:
            residual = input

        return F.relu(output + residual)  # Residual connection after ReLU


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer with Residual Connection
    """

    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, drop_prob=0.1):
        super(GraphAttentionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.dropout = dropout
        self.alpha = alpha
        self.drop_prob = drop_prob  # DropEdge probability

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.zeros(1, out_features * 2))  # Attention weights

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.softmax = nn.Softmax(dim=1)  # Softmax is computed over the neighbors

    def forward(self, h, adj):
        B, N, F = h.size()

        # Apply DropEdge to adjacency matrix
        adj = drop_edge(adj, drop_prob=self.drop_prob)  # Apply drop_edge

        # Linear transformation of node features
        h_prime = self.W(h)  # [B, N, F'']

        # Compute attention coefficients
        e = torch.matmul(h_prime, h_prime.transpose(1, 2))  # [B, N, N]
        e = self.leakyrelu(e)  # [B, N, N]
        attention = self.softmax(e)  # Softmax on each row [B, N, N]

        # Apply attention mechanism
        h_prime = h_prime.unsqueeze(2).repeat(1, 1, N, 1)  # [B, N, N, F'']
        h_prime = h_prime * attention.unsqueeze(-1)  # [B, N, N, F'']

        # Aggregate neighbor information
        output = torch.sum(h_prime, dim=2)  # [B, N, F'']

        return output


class TCNBlock(nn.Module):
    """
    TCN layer with Residual Connection
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=1,
            padding=(kernel_size - 1) * dilation // 2, dilation=dilation
        )
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        # Residual connection
        if in_channels != out_channels:
            self.residual_layer = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.residual_layer = None

    def forward(self, x):
        residual = x  # Save residual
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Residual connection
        if self.residual_layer is not None:
            residual = self.residual_layer(residual)

        return F.relu(x + residual)  # Residual connection after activation


class GCNWithGATAndTCN(nn.Module):
    """
    GCN with GAT, TCN, and Non-Local Block
    """

    def __init__(self, nfeat, nhid, nout, mat_path, dropout=0.3):
        super(GCNWithGATAndTCN, self).__init__()

        # First layer GCN
        self.gc1 = GraphConvolution(nfeat, nhid, mat_path)

        # First layer GAT and TCN
        self.gat1 = GraphAttentionLayer(nhid, nout, dropout)
        self.tcn1 = TCNBlock(nout, nout, kernel_size=3, dilation=1, dropout=0.2)

        # Add Non-Local Block after TCN
        self.non_local = NonLocalBlock(nout)

        # BatchNorm layers
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nout)

    def forward(self, x, adj):
        # Apply GCN
        x = self.gc1(x)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)  # BatchNorm
        x = F.relu(x)

        # Apply GAT and TCN
        x = self.gat1(x, adj)
        x = self.tcn1(x.transpose(1, 2)).transpose(1, 2)

        # Apply Non-Local Block
        x = self.non_local(x.transpose(1, 2)).transpose(1, 2)

        return x


class AUwGCNWithGATAndTCN(torch.nn.Module):
    """
    AU detection model with GCN, GAT, and one TCN layer
    """

    def __init__(self, opt):
        super().__init__()

        mat_dir = '/kaggle/working/ME-GCN-Project'
        self.mat_path = os.path.join(mat_dir, 'assets', '{}.npy'.format(opt['dataset']))

        # 使用修改后的 GCNWithGATAndTCN
        self.graph_embedding = GCNWithGATAndTCN(2, 16, 16, self.mat_path)

        in_dim = 192  # GCN 和 GAT 输出的维度
        self._sequential = torch.nn.Sequential(
            torch.nn.Conv1d(in_dim, 64, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
        )

        self._classification = torch.nn.Conv1d(
            64, 3 + 3 + 2 + 2, kernel_size=3, stride=1, padding=2, dilation=2, bias=False
        )

        self._init_weight()

    def forward(self, x):
        b, t, n, c = x.shape

        x = x.reshape(b * t, n, c)  # (b*t, n, c)

        # 获取邻接矩阵 adj
        adj = self.graph_embedding.gc1.adj  # 从 graph_embedding 中获取 adj
        # 调用 GCNWithGATAndTCN 进行图卷积、图注意力和 TCN 操作
        x = self.graph_embedding(x, adj)
        # reshape 处理为适合卷积输入的维度
        x = x.reshape(b, t, -1).transpose(1, 2)
        # 卷积操作
        x = self._sequential(x)
        # 分类层
        x = self._classification(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.Parameter):
                m.data.uniform_(-0.1, 0.1)
