import warnings

import networkx as nx

from utils import *
from NMF_layers import NMF, NMF2
from dataloader import load_data
from GAE_layers import GAAE
from loss_func import loss_fun

import time
import torch
import pandas as pd
import numpy as np
import community as community_louvain
import matplotlib.cm as cm
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib


matplotlib.use('TkAgg')
warnings.filterwarnings('ignore')


def NMFGAAE(graph, A, features, k, max_iter):
    # _, X = NMF(adj=A, k=256, epochs=200)  # 可以使用单位阵、或任何嵌入方法来初始化
    # print(X.shape)
    # X = X.t()

    #n_in_feat = X.shape[1]  # gcn输入特征的维度
    n_in_feat = features.shape[1]  #  gcn输入特征的维度
    n_hidden = 256  # gcn隐藏层的维度
    n_out_feat = 128 # gcn学习到的表示的维度

    lr = 1e-4  # gcn学习率
    num_header = 8  # 多头注意力的头数
    weight_decay = 3e-4 # 训练权重

    model = GAAE(graph, n_in_feat, n_hidden, n_out_feat, num_heads=num_header, numcommunity=k)
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)

    u = torch.rand(graph.number_of_nodes(), k)
    v = torch.rand(k, n_out_feat)

    for epoch in range(max_iter):

        Z, H, A_build = model(features, u.detach())
        Z = torch.sigmoid(Z)
        u, v = NMF2(Z, k, 200, u.detach(), v.detach())
        Z1 = torch.sigmoid(u.matmul(v))
        loss = loss_fun(A, A_build.detach(), Z, Z1.detach(), u.detach(), H)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = torch.softmax(u, dim=1).argmax(dim=1).detach().numpy()
        Q = Modularity(A, pred)
        #print(H)
        print('epoch={:d}, loss={:.4f}, Q={:.8f}'.format(epoch, loss, Q))
    np.set_printoptions(suppress=True)

    cluster_martix = cluster_get(u.detach().numpy(), threshold=0)
    adj_list, community_dict, list32, partition = community_matrix_to_adj_list(cluster_martix)

    # G = nx.Graph()
    # G = nx.from_numpy_matrix(A.detach().numpy())
    # Q = community_louvain.modularity(new_partition2, G)
    # print(Q)
    # #colors = ["#ed1299", "#09f9f5", "#246b93", "#cc8e12", "#d561dd", "#c93f00", "#ddd53e"]
    # colors = ["#ed1299", "#09f9f5", "#246b93", "#cc8e12", "#d561dd"]
    # cmap = {i: colors[i] for i in range(len(colors))}
    #
    # node_colors = []
    # for node in G.nodes():
    #     color_set = partition[node]
    #     if len(color_set) == 1:
    #         node_colors.append(cmap[list(color_set)[0] % len(colors)])
    #     else:
    #         # 如果节点属于两个社区，将颜色设置为两个社区颜色的拼接
    #         color1 = cmap[list(color_set)[0] % len(colors)]
    #         color2 = cmap[list(color_set)[1] % len(colors)]
    #         mixed_color = blend_colors(color1, color2)
    #         node_colors.append(mixed_color)
    #
    # # 绘制网络图
    # pos = nx.spring_layout(G)
    # labels = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13,
    #             13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 42, 23: 43, 24: 44,
    #             25: 45, 26: 46, 27: 47, 28: 48, 29: 49, 30: 50, 31: 51, 32: 52}
    # nx.draw(G, pos, node_color=node_colors, labels=labels, with_labels=True)
    # plt.show()
    # plt.savefig('my_figure.svg', dpi=150, format='svg', bbox_inches='tight')


if __name__ == '__main__':


    time_start = time.time()

    data = pd.read_csv(r'H:\TE\d00.dat', sep='  ', header=None, engine='python')
    data = data.values[:, :].T
    X1 = data[:, :22]
    X2 = data[:, -11:]
    X = np.concatenate((X1, X2), axis=1)

    g, A, features, k = load_data(X, lamda=1)
    NMFGAAE(g, A, features, k, max_iter=100)

    time_end = time.time()
    time_sum = time_end - time_start
    print(time_sum)