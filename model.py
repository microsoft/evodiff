import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from sequence_models.layers import PositionFeedForward, PositionFeedForward2d, DoubleEmbedding
from sequence_models.convolutional import MaskedConv1d, MaskedCausalConv1d, ByteNetBlock

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=8, length=500):
        super().__init__()
        self.d_model = d_model
        self.length = length

    def forward(self, x):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if self.d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(self.d_model))
        pe = torch.zeros(self.length, self.d_model)
        position = torch.arange(0, self.length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, self.d_model, 2, dtype=torch.float) *
                              -(np.log(10000.0) / self.d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        return pe[x].to(x.device)


class ByteNetTime(nn.Module):
    """Stacked residual blocks from ByteNet paper defined by n_layers

         Shape:
            Input: (N, L,)
            input_mask: (N, L, 1), optional
            Output: (N, L, d)
    """

    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r, rank=None, n_frozen_embs=None,
                 padding_idx=None, causal=False, dropout=0.0, slim=True, activation='relu', down_embed=True,
                 timesteps=None):
        """
        :param n_tokens: number of tokens in token dictionary
        :param d_embedding: dimension of embedding
        :param d_model: dimension to use within ByteNet model, //2 every layer
        :param n_layers: number of layers of ByteNet block
        :param kernel_size: the kernel width
        :param r: used to calculate dilation factor
        :padding_idx: location of padding token in ordered alphabet
        :param causal: if True, chooses MaskedCausalConv1d() over MaskedConv1d()
        :param rank: rank of compressed weight matrices
        :param n_frozen_embs: number of frozen embeddings
        :param slim: if True, use half as many dimensions in the NLP as in the CNN
        :param activation: 'relu' or 'gelu'
        :param down_embed: if True, have lower dimension for initial embedding than in CNN layers
        :param timesteps: None or int providing max timesteps in DM model
        """
        super().__init__()
        self.timesteps = timesteps
        if self.timesteps is not None:
            self.time_encoding = PositionalEncoding(d_embedding, timesteps)
        else:
            self.up_embedder = nn.Identity()
        if n_tokens is not None:
            if n_frozen_embs is None:
                self.embedder = nn.Embedding(n_tokens, d_embedding, padding_idx=padding_idx)
            else:
                self.embedder = DoubleEmbedding(n_tokens - n_frozen_embs, n_frozen_embs,
                                                d_embedding, padding_idx=padding_idx)
        else:
            self.embedder = nn.Identity()
        if down_embed:
            self.up_embedder = PositionFeedForward(d_embedding, d_model)
        else:
            self.up_embedder = nn.Identity()
            assert n_tokens == d_embedding
        log2 = int(np.log2(r)) + 1
        dilations = [2 ** (n % log2) for n in range(n_layers)]
        d_h = d_model
        if slim:
            d_h = d_h // 2
        layers = [
            ByteNetBlock(d_model, d_h, d_model, kernel_size, dilation=d, causal=causal, rank=rank,
                         activation=activation)
            for d in dilations
        ]
        self.layers = nn.ModuleList(modules=layers)
        self.dropout = dropout

    def forward(self, x, y, input_mask=None):
        """
        :param x: (batch, length)
        :param y: (batch)
        :param input_mask: (batch, length, 1)
        :return: (batch, length,)
        """
        e = self._embed(x, y, timesteps=self.timesteps)
        return self._convolve(e, input_mask=input_mask)

    def _embed(self, x, y, timesteps=None):
        e1 = self.embedder(x)
        if timesteps is not None:
            e2 = self.time_encoding(y)
            # expand dim of e2 to match e1
            e2 = e2.expand(e1.shape[1], e2.shape[0], e2.shape[1])
            e2 = e2.reshape(e1.shape[0], e1.shape[1], e1.shape[2])
            e = torch.add(e2,e1)
        else:
            e = e1
        e = self.up_embedder(e)
        return e

    def _convolve(self, e, input_mask=None):
        for layer in self.layers:
            e = layer(e, input_mask=input_mask)
            if self.dropout > 0.0:
                e = F.dropout(e, self.dropout)
        return e


class ByteNetLMTime(nn.Module):

    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r, rank=None, n_frozen_embs=None,
                 padding_idx=None, causal=False, dropout=0.0, final_ln=False, slim=True, activation='relu',
                 tie_weights=False, down_embed=True, timesteps=None):
        super().__init__()
        self.embedder = ByteNetTime(n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                                padding_idx=padding_idx, causal=causal, dropout=dropout, down_embed=down_embed,
                                slim=slim, activation=activation, rank=rank, n_frozen_embs=n_frozen_embs,
                                timesteps=timesteps)
        if tie_weights:
            self.decoder = nn.Linear(d_model, n_tokens, bias=False)
            self.decoder.weight = self.embedder.embedder.weight
        else:
            self.decoder = PositionFeedForward(d_model, n_tokens)
        if final_ln:
            self.last_norm = nn.LayerNorm(d_model)
        else:
            self.last_norm = nn.Identity()

    def forward(self, x, y, input_mask=None):
        e = self.embedder(x, y, input_mask=input_mask)
        e = self.last_norm(e)
        return self.decoder(e)