import torch
import networkx as nx
import numpy as np
import community  # pip install  python-louvain


def Modularity(adj, part, classes_num=None):
    graph = nx.from_numpy_matrix(adj.numpy())
    part = part.tolist()
    index = range(0, len(part))
    dic = zip(index, part)
    part = dict(dic)
    modur = community.modularity(part, graph)
    return modur

def Modular(array, cluster):
    """"使用矩阵法计算模块度"""
    m = sum(sum(array))/2
    k1 = array.sum(axis=1)
    k2 = k1.reshape(k1.shape[0], 1)
    k1k2 = k1 * k2
    Eij = k1k2 / (2 * m)
    B = array - Eij
    node_cluster = np.dot(cluster, np.transpose(cluster))
    results = np.dot(B, node_cluster)
    sum_results = np.trace(results)
    Q = sum_results / (2 * m)
    return Q

def calculate_modularity(Q, adj_matrix, communities):
    m = np.sum(adj_matrix) / 2
    c = len(communities)

    for nodes_in_community in communities:
        internal_edges = adj_matrix[np.ix_(nodes_in_community, nodes_in_community)].sum() / 2
        degree_sum = adj_matrix[np.ix_(nodes_in_community)].sum()
        external_degree_sum = adj_matrix[nodes_in_community, :].sum()
        O = external_degree_sum / (2 * m)
        Q += internal_edges / m - degree_sum * external_degree_sum / (2 * m * m) + O * O

    return Q

def get_communities(pred):
    partition = dict(zip(range(len(pred)), pred))
    communities = [[] for _ in range(max(pred) + 1)]
    for node, comm in partition.items():
        communities[comm].append(node)
    return communities, partition

def cluster_get(u, threshold):
    #用于构建重叠社区（阈值为0的时候就是不重叠社区）的社区成员指示矩阵
    cluster_martix = np.zeros((u.shape[0], u.shape[1]))
    max_indexes = np.argmax(u, axis=1)
    #print(max_indexes)
    for i in range(u.shape[0]):
        row = u[i]
        max_val = row[max_indexes[i]]
        row_change = cluster_martix[i]
        row_change[max_indexes[i]] = 1
        for j in range(len(row)):
            if abs(max_val - row[j]) < threshold:
                row_change[j] = 1

    return cluster_martix

def community_matrix_to_adj_list(cluster_matrix):
    num_nodes, num_communities = cluster_matrix.shape
    adj_list = {i: [] for i in range(num_nodes)}
    community_list = {i: [] for i in range(num_communities)}
    list = []
    partition={}
    for i in range(num_communities):
        community_members = np.where(cluster_matrix[:, i])[0]
        community_list[i] = community_members.tolist()
        list.append(community_members.tolist())
        for node in community_members:
            if node in adj_list:
                adj_list[node].append(i)

    partition = {key: set(value) for key, value in adj_list.items()}
    return adj_list, community_list, list, partition

def blend_colors(color1, color2):
    r1, g1, b1 = tuple(int(color1.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    r2, g2, b2 = tuple(int(color2.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    r = int((r1 + r2) / 2)
    g = int((g1 + g2) / 2)
    b = int((b1 + b2) / 2)
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def add_n_to_list_values(dct, n, inc):
    """
    给定一个字典 dct、一个数字 n 和一个增量 inc，
    将 dct 中所有值为列表的键对应的列表中所有大于 n 的元素加上 inc。
    """
    for key, value in dct.items():
        if isinstance(value, list):
            dct[key] = [x + inc if x > n else x for x in value]
    return dct