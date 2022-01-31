import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h=8, mask=False):
        """
        :param d_model: int, embedding dimension
        :param d_k: int, key dimension
        :param d_v: int, value dimension
        :param h: int=8, number of heads
        :param mask: bool=false, whether to mask positions > i
        :return: None
        """
        super(MultiHeadAttention, self).__init__()
        self.mask = mask
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.W_K = nn.Parameter(torch.Tensor(d_model, d_k * h))
        self.W_Q = nn.Parameter(torch.Tensor(d_model, d_k * h))
        self.W_V = nn.Parameter(torch.Tensor(d_model, d_v * h))
        self.W_O = nn.Parameter(torch.Tensor(d_v*h, d_model))
        self.layer_norm = nn.LayerNorm(d_model)
        self.reset_parameters()  # weight initialization

    def reset_parameters(self):
        nn.init.xavier_uniform(self.W_V)
        nn.init.xavier_uniform(self.W_Q)
        nn.init.xavier_uniform(self.W_K)
        nn.init.xavier_uniform(self.W_O)

    def scaled_dot_product_attention(self):


    def forward(self, Q, K, V):
        """
        :param Q: tensor dimension batch x len_q x d_model
        :param K: tensor dimension batch x len_k x d_model
        :param V: tensor dimension batch x len_k x d_model
        :return: tensor dimension batch x len_q x d_model
        """
        Q_t = torch.matmul(Q, self.W_Q)
        K_t = torch.matmul(K, self.W_K)
        V_t = torch.matmul(V, self.W_V)





class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()
        pass

    def forward(self):
        pass


class PositionalEncoder:
    def __init__(self):
        pass
