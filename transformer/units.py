import torch
import torch.nn as nn

INFINITY = 1e9


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

    def apply_mask(self, QKT):
        """
        when applying mask, we want to mask the upper triangular (not including diagonal) part to negative inf
        this is because QKT is len_q x len_k, and we want q only look at index smaller than it
        only works for self attention, so len_q == len_k
        """
        len_q = QKT.size(-1)
        assert QKT.size(-1) == QKT.size(-2)
        mask_tensor = torch.triu(torch.ones((len_q, len_q)) * -1 * INFINITY, diagonal=1)
        return QKT + mask_tensor

    def scaled_dot_product(self, Q_t, K_t):
        """
        :param Q_t: tensor dimension batch x len_q x (d_k * h)
        :param K_t: tensor dimension batch x len_k x (d_k * h)
        :return: tensor dimension batch x h x len_q x len_k
        """
        batch_size, len_q = Q_t.size(0), Q_t.size(1)
        Q_t = Q_t.view(batch_size, len_q, self.h, self.d_k).transpose(1, 2)  # batch x h x len_q x d_k
        K_t = K_t.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # batch x h x len_k x d_k
        QKT = 1/(self.d_k**(1/2)) * torch.matmul(Q_t, K_t.transpose(3, 4)) # batch x h x len_q x len_k
        if not self.mask:
            return nn.Softmax(dim=-1)(QKT)
        else:
            QKT = self.apply_mask(QKT)
            return nn.Softmax(dim=-1)(QKT)

    def forward(self, Q, K, V):
        """
        :param Q: tensor dimension batch x len_q x d_model
        :param K: tensor dimension batch x len_k x d_model
        :param V: tensor dimension batch x len_k x d_model
        :return: tensor dimension batch x len_q x d_model
        """
        batch_size = Q.size(0)
        Q_t = torch.matmul(Q, self.W_Q)
        K_t = torch.matmul(K, self.W_K)
        V_t = torch.matmul(V, self.W_V)  # batch x len_k x (d_v * h)
        #  compute context
        V_t = V_t.view(batch_size, -1, self.h, self.d_v).transpose(1, 2)  # batch x h x len_k x d_v
        attention = self.scaled_dot_product(Q_t, K_t)  # batch x h x len_q x len_k
        context = torch.matmul(attention, V_t)  # batch x h x len_q x d_v
        context = context.transpose(1, 2).view(batch_size, -1, self.h*self.d_v)  # batch x len_q x (h*d_v)
        output = torch.matmul(context, self.W_O)  # batch x len_q x d_model
        return self.layer_norm(Q + output)


class FeedForward(nn.Module):
    def __init__(self, d_model):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(in_features=d_model, out_features=d_model)
        self.linear2 = nn.Linear(in_features=d_model, out_features=d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, X):
        """
        :param X: tensor dimension batch x len_q x d_model
        :return out: tensor dimension batch x len_q x d_model
        """
        out = self.linear2(nn.ReLU()(self.linear1(X)))
        return self.layer_norm(out+X)


class PositionalEncoder:
    def __init__(self):
        pass
