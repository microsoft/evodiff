import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from sequence_models.layers import PositionFeedForward, PositionFeedForward2d, DoubleEmbedding


class MaskedConv1d(nn.Conv1d):
    """ A masked 1-dimensional convolution layer.
    Takes the same arguments as torch.nn.Conv1D, except that the padding is set automatically.
         Shape:
            Input: (N, L, in_channels)
            input_mask: (N, L, 1), optional
            Output: (N, L, out_channels)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, dilation: int = 1, groups: int = 1,
                 bias: bool = True):
        """
        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size: the kernel width
        :param stride: filter shift
        :param dilation: dilation factor
        :param groups: perform depth-wise convolutions
        :param bias: adds learnable bias to output
        """
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                         groups=groups, bias=bias, padding=padding)

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class MaskedConv2d(nn.Conv2d):
    """ A masked 2-dimensional convolution layer.
    Takes the same arguments as torch.nn.Conv2D, except that the padding is set automatically.
         Shape:
            Input: (N, L, L, in_channels)
            input_mask: (N, L, L, 1), optional
            Output: (N, L, L, out_channels)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, dilation: int = 1, groups: int = 1,
                 bias: bool = True):
        """
        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size: the kernel width
        :param stride: filter shift
        :param dilation: dilation factor
        :param groups: perform depth-wise convolutions
        :param bias: adds learnable bias to output
        """
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                         groups=groups, bias=bias, padding=padding)

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class MaskedCausalConv1d(nn.Module):
    """Masked Causal 1D convolution based on https://github.com/Popgun-Labs/PopGen/.

         Shape:
            Input: (N, L, in_channels)
            input_mask: (N, L, 1), optional
            Output: (N, L, out_channels)
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, dilation=1, groups=1, init=None):
        """
        Causal 1d convolutions with caching mechanism for O(L) generation,
        as described in the ByteNet paper (Kalchbrenner et al, 2016) and "Fast Wavenet" (Paine, 2016)
        Usage:
            At train time, API is same as regular convolution. `conv = CausalConv1d(...)`
            At inference time, set `conv.sequential = True` to enable activation caching, and feed
            sequence through step by step. Recurrent state is managed internally.
        References:
            - Neural Machine Translation in Linear Time: https://arxiv.org/abs/1610.10099
            - Fast Wavenet: https://arxiv.org/abs/1611.09482
        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size: the kernel width
        :param dilation: dilation factor
        :param groups: perform depth-wise convolutions
        :param init: optional initialisation function for nn.Conv1d module (e.g xavier)
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.groups = groups

        # if `true` enables fast generation
        self.sequential = False

        # compute required amount of padding to preserve the length
        self.zeros = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, groups=groups)

        # use supplied initialization function
        if init:
            init(self.conv)

    def forward(self, x, input_mask=None):
        """
        :param x: (batch, length, in_channels)
        :param input_mask: (batch, length, 1)
        :return: (batch, length, out_channels)
        """
        if input_mask is not None:
            x = x * input_mask
        # training mode
        x = torch.transpose(x, 1, 2)
        if not self.sequential:
            # no padding for kw=1
            if self.kernel_size == 1:
                return self.conv(x).transpose(1, 2)

            # left-pad + conv.
            out = self._pad(x)
            return self._unpad(self.conv(out)).transpose(1, 2)

        # sampling mode
        else:
            # note: x refers to a single timestep (batch, features, 1)
            if not hasattr(self, 'recurrent_state'):
                batch_size = x.size(0)
                self._init_recurrent_state(batch_size)

            return self._generate(x).transpose(1, 2)

    def _pad(self, x):
        return F.pad(x, [self.zeros, 0])

    def _unpad(self, x):
        return x

    def clear_cache(self):
        """
        Delete the recurrent state. Note: this should be called between runs, to prevent
        leftover state bleeding into future samples. Note that we delete state (instead of zeroing) to support
        changes in the inference time batch size.
        """
        if hasattr(self, 'recurrent_state'):
            del self.recurrent_state

    def _init_recurrent_state(self, batch):
        """
        Initialize the recurrent state for fast generation.
        :param batch: the batch size to generate
        """

        # extract weights and biases from nn.Conv1d module
        state = self.conv.state_dict()
        self.weight = state['weight']
        self.bias = state['bias']

        # initialize the recurrent states to zeros
        self.recurrent_state = torch.zeros(batch, self.in_channels, self.zeros, device=self.bias.device)

    def _generate(self, x_i):
        """
        Generate a single output activations, from the input activation
        and the cached recurrent state activations from previous steps.
        :param x_i: features of a single timestep (batch, in_channels, 1)
        :return: the next output value in the series (batch, out_channels, 1)
        """

        # if the kernel_size is greater than 1, use recurrent state.
        if self.kernel_size > 1:
            # extract the recurrent state and concat with input column
            recurrent_activations = self.recurrent_state[:, :, :self.zeros]
            f = torch.cat([recurrent_activations, x_i], 2)

            # update the cache for this layer
            self.recurrent_state = torch.cat(
                [self.recurrent_state[:, :, 1:], x_i], 2)
        else:
            f = x_i

        # perform convolution
        activations = F.conv1d(f, self.weight, self.bias,
                               dilation=self.dilation, groups=self.groups)

        return activations


class ByteNetBlock(nn.Module):
    """Residual block from ByteNet paper (https://arxiv.org/abs/1610.10099).

         Shape:
            Input: (N, L, d_in)
            input_mask: (N, L, 1), optional
            Output: (N, L, d_out)
    """

    def __init__(self, d_in, d_h, d_out, kernel_size, dilation=1, groups=1, causal=False, activation='relu', rank=None):
        super().__init__()
        if causal:
            self.conv = MaskedCausalConv1d(d_h, d_h, kernel_size=kernel_size, dilation=dilation, groups=groups)
        else:
            self.conv = MaskedConv1d(d_h, d_h, kernel_size=kernel_size, dilation=dilation, groups=groups)
        if activation == 'relu':
            act = nn.ReLU
        elif activation == 'gelu':
            act = nn.GELU
        layers1 = [
            nn.LayerNorm(d_in),
            act(),
            PositionFeedForward(d_in, d_h, rank=rank),
            nn.LayerNorm(d_h),
            act()
        ]
        layers2 = [
            nn.LayerNorm(d_h),
            act(),
            PositionFeedForward(d_h, d_out, rank=rank),
        ]
        self.sequence1 = nn.Sequential(*layers1)
        self.sequence2 = nn.Sequential(*layers2)

    def forward(self, x, input_mask=None):
        """
        :param x: (batch, length, in_channels)
        :param input_mask: (batch, length, 1)
        :return: (batch, length, out_channels)
        """
        return x + self.sequence2(
            self.conv(self.sequence1(x), input_mask=input_mask)
        )


class ByteNet(nn.Module):
    """Stacked residual blocks from ByteNet paper defined by n_layers

         Shape:
            Input: (N, L,)
            input_mask: (N, L, 1), optional
            Output: (N, L, d)
    """

    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r, rank=None, n_frozen_embs=None,
                 padding_idx=None, causal=False, dropout=0.0, slim=True, activation='relu', down_embed=True):
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
        """
        super().__init__()
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

    def forward(self, x, input_mask=None):
        """
        :param x: (batch, length)
        :param input_mask: (batch, length, 1)
        :return: (batch, length,)
        """
        e = self._embed(x)
        return self._convolve(e, input_mask=input_mask)

    def _embed(self, x):
        e = self.embedder(x)
        e = self.up_embedder(e)
        return e

    def _convolve(self, e, input_mask=None):
        for layer in self.layers:
            e = layer(e, input_mask=input_mask)
            if self.dropout > 0.0:
                e = F.dropout(e, self.dropout)
        return e


class ByteNetLM(nn.Module):

    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r, rank=None, n_frozen_embs=None,
                 padding_idx=None, causal=False, dropout=0.0, final_ln=False, slim=True, activation='relu',
                 tie_weights=False, down_embed=True):
        super().__init__()
        self.embedder = ByteNet(n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                                padding_idx=padding_idx, causal=causal, dropout=dropout, down_embed=down_embed,
                                slim=slim, activation=activation, rank=rank, n_frozen_embs=n_frozen_embs)
        if tie_weights:
            self.decoder = nn.Linear(d_model, n_tokens, bias=False)
            self.decoder.weight = self.embedder.embedder.weight
        else:
            self.decoder = PositionFeedForward(d_model, n_tokens)
        if final_ln:
            self.last_norm = nn.LayerNorm(d_model)
        else:
            self.last_norm = nn.Identity()

    def forward(self, x, input_mask=None):
        #print("embedder weight dim", len(self.embedder.embedder.weight))
        e = self.embedder(x, input_mask=input_mask)
        e = self.last_norm(e)
        return self.decoder(e)


class ConditionedByteNetLM(nn.Module):

    def __init__(self, n_tokens, d_embedding, d_conditioning, d_model, n_layers, kernel_size, r,
                 padding_idx=None, causal=False):
        super().__init__()
        self.embedder = ConditionedByteNetDecoder(n_tokens, d_embedding, d_conditioning,
                                                  d_model, n_layers, kernel_size, r,
                                                  padding_idx=padding_idx, causal=causal)
        self.decoder = PositionFeedForward(d_model, n_tokens)

    def forward(self, x, input_mask=None):
        e = self.embedder(x, input_mask=input_mask)
        return self.decoder(e)


class ConditionedByteNetDecoder(ByteNet):
    """ A conditioned, ByteNet decoder.
    Inputs:
        x (n, ell)
        c: (n, d_conditioning)
    """

    def __init__(self, n_tokens, d_embedding, d_conditioning, d_model, n_layers, kernel_size, r,
                 padding_idx=None, causal=False):
        """
        :param n_tokens: number of tokens in token dictionary
        :param d_embedding: dimension of embedding
        :param d_conditioning: dimension for conditioning, subtract from d_model
        :param d_model: dimension to use within ByteNet model, //2 every layer
        :param n_layers: number of layers of ByteNet block
        :param kernel_size: the kernel width
        :param r: used to calculate dilation factor
        """
        super().__init__(n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                         padding_idx=padding_idx, causal=causal)
        self.up_embedder = PositionFeedForward(d_embedding, d_model - d_conditioning)

    def _embed(self, inputs):
        x, c = inputs
        e = self.embedder(x)
        e = self.up_embedder(e)  # (n, ell, d_model - d_conditioning)
        # Concatenate the conditioning
        _, ell = x.shape
        if len(c.shape) == 2:
            c = c.unsqueeze(1)
            c_ = torch.repeat_interleave(c, ell, dim=1)  # (n, ell, d_conditioning)
        else:
            c_ = c
        e = torch.cat([e, c_], dim=2)  # (n, ell, d_model)
        return e


class ByteNetBlock2d(nn.Module):
    """Residual block from ByteNet paper (https://arxiv.org/abs/1610.10099).
         Shape:
            Input: (N, L, L, d_in)
            input_mask: (N, L, L, 1), optional
            Output: (N, L, L, d_out)
    """

    def __init__(self, d_in, d_h, d_out, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.conv = MaskedConv2d(d_h, d_h, kernel_size=kernel_size, dilation=dilation, groups=groups)
        layers1 = [
            nn.LayerNorm(d_in),
            nn.ReLU(),
            PositionFeedForward2d(d_in, d_h),
            nn.LayerNorm(d_h),
            nn.ReLU()
        ]
        layers2 = [
            nn.LayerNorm(d_h),
            nn.ReLU(),
            PositionFeedForward2d(d_h, d_out),
        ]
        self.sequence1 = nn.Sequential(*layers1)
        self.sequence2 = nn.Sequential(*layers2)

    def forward(self, x, input_mask=None):
        """
        :param x: (batch, length, length, in_channels)
        :param input_mask: (batch, length, length, 1)
        :return: (batch, length, length, out_channels)
        """
        return x + self.sequence2(
            self.conv(self.sequence1(x), input_mask=input_mask)
        )


class ByteNet2d(nn.Module):
    """Stacked residual blocks from ByteNet paper defined by n_layers
         Shape:
            Input: (N, L, L, d_in)
            input_mask: (N, L, L, 1), optional
            Output: (N, L, d_model)
    """

    def __init__(self, d_in, d_model, n_layers, kernel_size, r, dropout=0.0):
        """
        :param d_in: number of input dimensions
        :param d_model: dimension to use within ByteNet model, // 2 every layer
        :param n_layers: number of layers of ByteNet block
        :param kernel_size: the kernel width
        :param r: used to calculate dilation factor
        """
        super().__init__()
        self.up_embedder = PositionFeedForward2d(d_in, d_model)
        log2 = int(np.log2(r)) + 1
        dilations = [2 ** (n % log2) for n in range(n_layers)]
        layers = [
            ByteNetBlock2d(d_model, d_model // 2, d_model, kernel_size, dilation=d)
            for d in dilations
        ]
        self.layers = nn.ModuleList(modules=layers)
        self.dropout = dropout

    def forward(self, x, input_mask=None):
        e = self._embed(x)
        return self._convolve(e, input_mask=input_mask)

    def _embed(self, x):
        e = self.up_embedder(x)
        return e

    def _convolve(self, e, input_mask=None):
        for layer in self.layers:
            e = layer(e, input_mask=input_mask)
            if self.dropout > 0.0:
                e = F.dropout(e, self.dropout)
        return e
