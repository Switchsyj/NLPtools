import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy
import math
import numpy as np


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Transformer(nn.Module):
    """
    Transformer Encoder
    """
    def __init__(self, encoder, enc_embedder, decoder, dec_embedder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.enc_embedder = enc_embedder
        self.decoder = decoder
        self.dec_embedder = dec_embedder

    def forward(self, enc_input, enc_input_mask, dec_input, dec_mask):
        return self.decoder(self.dec_embedder(dec_input), self.encoder(self.enc_embedder(enc_input), enc_input_mask),
                            dec_mask)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, x_mask):
        for layer in self.layers:
            x = layer(x, x_mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.size = layer.size

    def forward(self, x, memory, x_src_mask, x_tgt_mask):
        tgt_attn, src_attn = None, None
        batch_size, length, _ = x.size()
        for layer in self.layers:
            x, tgt_attn, src_attn = layer(x, memory, x_src_mask, x_tgt_mask)
        coverage = src_attn
        coverage_records = torch.matmul(torch.triu(x.new_ones(length, length), diagonal=1).transpose(0, 1).unsqueeze(0),
                                        src_attn)
        # shape(1, 1, 1) (batch, 1, src_length)
        output = self.norm(x)
        return dict(
            decoder_hidden_states=output.clone(),
            rnn_hidden_states=output.clone(),
            target_copy_attentions=tgt_attn,
            source_copy_attentions=src_attn,
            coverage_records=coverage_records,
            coverage=coverage
        )


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.ones(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.linears = clones(nn.Linear(d_model, d_model, bias=False), 4)
        self.h = h
        self.d_k = d_model // h
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        :params query: shape(batch, len, d_mdoel)
        """
        batch_size, length, d_model = query.size()
        # mask shape(batch, 1, 1, seq_len)
        # if mask is not None:
        #     mask = mask.unsqueeze(1)
        query, key, value = [proj(x).view(batch_size, -1, self.h * self.d_k) for x, proj in
                             zip((query, key, value), self.linears)]
        attn_score = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(d_model)
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, -1e9)
        attn_score = self.dropout(F.softmax(attn_score, dim=-1))
        out_put = torch.bmm(attn_score, value).view(batch_size, -1, self.d_k * self.h)
        # final projection on output space
        return self.linears[-1](out_put), attn_score


class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FFN, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear_2(self.dropout(F.relu(self.linear_1(x))))


class ResNet(nn.Module):
    def __init__(self, size, dropout):
        super(ResNet, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(size)

    def forward(self, x, layer, self_attn_flag):
        if self_attn_flag:
            # TODO: must copy
            output, attn_score = layer(self.norm(x))
            return x + self.dropout(output), attn_score
        else:
            return x + self.dropout(layer(self.norm(x)))


class EncoderSubLayer(nn.Module):
    def __init__(self, size, attn, ffn, dropout):
        super(EncoderSubLayer, self).__init__()
        self.attn = attn
        self.ffn = ffn
        self.size = size
        self.dropout = dropout
        self.resnet = clones(ResNet(size, dropout), 2)

    def forward(self, x, x_mask):
        # the second parameter is a function
        x, _ = self.resnet[0](x, lambda x: self.attn(x, x, x, x_mask), True)
        return self.resnet[1](x, self.ffn, False)


class DecoderSublayer(nn.Module):
    def __init__(self, size, attn, ffn, dropout):
        super(DecoderSublayer, self).__init__()
        self.attn = attn
        self.ffn = ffn
        self.size = size
        self.dropout = dropout
        self.resnet = clones(ResNet(size, dropout), 3)

    def forward(self, x, memory, x_src_mask, x_tgt_mask):
        # the second parameter is a function
        x, tgt_attn = self.resnet[0](x, lambda x: self.attn(x, x, x, x_tgt_mask), True)
        x, src_attn = self.resnet[1](x, lambda x: self.attn(x, memory, memory, x_src_mask), True)
        return self.resnet[2](x, self.ffn, False), tgt_attn, src_attn


class PositionalEmbedding(nn.Module):
    """
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, input_size, d_model, dropout, max_len=5000, pos_embedding=True):
        """
        d_model: over_all embedding dimension
        """
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.embed_proj = nn.Linear(input_size, d_model, bias=False)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        # the same 'i' for odd and even position
        factor = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(pos * factor)
        pe[:, 1::2] = torch.cos(pos * factor)
        # unsqueeze for batch dim
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        self.pos_embedding = pos_embedding
        # TODO: training embedding
        self.train_pos_embedding = nn.Parameter(torch.Tensor(max_len, d_model), requires_grad=True)
        nn.init.normal_(self.train_pos_embedding)

    def forward(self, x):
        """
        x for embedding matrix shape(batch, seq_len, embed_dim)
        """
        x = self.embed_proj(x)
        if self.pos_embedding:
            # x += Variable(self.pe[:, :x.shape[1]], requires_grad=False)
            x += self.train_pos_embedding.data[:x.shape[1]].unsqueeze(0)
        return self.dropout(x)
