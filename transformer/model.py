import torch
import torch.nn as nn
import numpy as np
from units import MultiHeadAttention, FeedForward, PositionalEncoder


class Encoder(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048):
        super(Encoder, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, d_k, d_v, h=h, mask=False)
        self.feed_forward = FeedForward(d_model, d_ff)

    def forward(self, embeddings):
        """
        :param embeddings: tensor dimension batch x len_q x d_model
        :return : tensor dimension batch x len_q x d_model
        """
        out = self.self_attention(embeddings, embeddings, embeddings)
        return self.feed_forward(out)


class Decoder(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048):
        super(Decoder, self).__init__()
        self.masked_self_attention = MultiHeadAttention(d_model, d_k, d_v, h=h, mask=True)
        self.self_attention = MultiHeadAttention(d_model, d_k, d_v, h=h, mask=False)
        self.feed_forward = FeedForward(d_model, d_ff)

    def forward(self, out_embed, in_embed):
        """
        :param out_embed: tensor dimension batch x len_q x d_model
        :param in_embed: tensor dimension batch x len_k x d_model
        :return: tensor dimension batch x len_q x d_model
        """
        out = self.masked_self_attention(out_embed, out_embed, out_embed)
        out = self.self_attention(Q=out, K=in_embed, V=in_embed)
        return self.feed_forward(out)


class Transformer(nn.Module):
    def __init__(self, vocab_size, max_sentence_len, n_encoders=6, n_decoders=6, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048):
        super(Transformer, self).__init__()
        self.position_encoder = PositionalEncoder(d_model, max_sentence_len)
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        for i in range(n_encoders):
            self.encoder_layers.append(Encoder(d_model=d_model, d_k=d_k, d_v=d_v, h=h, d_ff=d_ff))
        for i in range(n_decoders):
            self.decoder_layers.append(Decoder(d_model=d_model, d_k=d_k, d_v=d_v, h=h, d_ff=d_ff))
        self.linear = nn.Linear(in_features=d_model, out_features=vocab_size, bias=False)

    def forward(self, out_embed, in_embed):
        """
        :param out_embed: tensor dimension batch x len_q x d_model
        :param in_embed: tensor dimension batch x len_k x d_model
        :return: tensor dimension batch x len_q x vocab_size (probability distribution of word index)
        """
        in_embed = in_embed + self.position_encoder.apply(in_embed.size(-2))
        out_embed = out_embed + self.position_encoder.apply(out_embed.size(-2))
        for layer in self.encoder_layers:
            in_embed = layer(in_embed)
        for layer in self.decoder_layers:
            out_embed = layer(out_embed, in_embed)
        out = self.linear(out_embed)  # batch x len_q x vocab_size
        return nn.Softmax(out)
