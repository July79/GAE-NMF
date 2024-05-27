import torch
from utils import Modular
bce_loss = torch.nn.BCELoss(reduction='sum')


def loss_fun(A, A1, Z, Z1, u, H):
    GAE_loss = bce_loss(A.view(-1), A1.view(-1))
    NMF_loss = bce_loss(Z.view(-1), Z1.view(-1))

    u_sigmoid = torch.sigmoid(u)
    max_Q_loss = Modular(A, H.detach())
    loss = bce_loss(u_sigmoid.view(-1), H.view(-1))

    Loss = 10**(0) * GAE_loss + 10**(0) * NMF_loss - 10**(4) * max_Q_loss+ 10**(0) * loss
    Loss1 = GAE_loss + 1 * NMF_loss
    return Loss