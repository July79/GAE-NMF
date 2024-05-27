import torch

mse_loss = torch.nn.MSELoss()


def NMF(adj, k, epochs, U=None, V=None):
    if not isinstance(adj, torch.Tensor):
        adj = torch.tensor(adj, dtype=torch.float)
    u = U if U is not None else torch.rand(adj.shape[0], k)
    v = V if V is not None else torch.rand(k, adj.shape[1])


    for epoch in range(epochs):
        u = u * ((adj.matmul(v.t())) / (u.matmul(v).matmul(v.t()))) + 1e-20
        v = v * ((u.t().matmul(adj)) / (u.t().matmul(u).matmul(v))) + 1e-20
    return u, v


def NMF2(adj, k, epochs, U=None, V=None):
    if not isinstance(adj, torch.Tensor):
        adj = torch.tensor(adj, dtype=torch.float)
    u = U if U is not None else torch.rand(adj.shape[0], k)
    v = V if V is not None else torch.rand(k, adj.shape[1])

    for epoch in range(epochs):
        u = u * ((adj.matmul(v.t())) / (u.matmul(u.t()).matmul(adj).matmul(v.t()))) + 1e-20
        v = v * ((u.t().matmul(adj)) / (u.t().matmul(u).matmul(v))) + 1e-20
    return u, v