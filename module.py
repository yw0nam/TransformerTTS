# ========================================== #
#              Thanks to soobinseo           #
# ========================================== #
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import copy
import math

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Linear(nn.Module):
    """
    Linear Module
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=True, w_init='linear'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              bias=bias)

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        x = self.conv(x)
        return x


class EncoderPrenet(nn.Module):
    """
    Pre-network for Encoder consists of convolution networks.
    """

    def __init__(self, embedding_size, num_hidden, symbols_length):
        super(EncoderPrenet, self).__init__()
        self.embedding_size = embedding_size
        self.embed = nn.Embedding(symbols_length, embedding_size, padding_idx=0)

        self.conv1 = Conv(in_channels=embedding_size,
                          out_channels=num_hidden,
                          kernel_size=5,
                          padding=int(np.floor(5 / 2)),
                          w_init='relu')
        self.conv2 = Conv(in_channels=num_hidden,
                          out_channels=num_hidden,
                          kernel_size=5,
                          padding=int(np.floor(5 / 2)),
                          w_init='relu')

        self.conv3 = Conv(in_channels=num_hidden,
                          out_channels=num_hidden,
                          kernel_size=5,
                          padding=int(np.floor(5 / 2)),
                          w_init='relu')

        self.batch_norm1 = nn.BatchNorm1d(num_hidden)
        self.batch_norm2 = nn.BatchNorm1d(num_hidden)
        self.batch_norm3 = nn.BatchNorm1d(num_hidden)

        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.projection = Linear(num_hidden, num_hidden)

    def forward(self, input_):
        input_ = self.embed(input_)
        input_ = input_.transpose(1, 2)
        input_ = self.dropout1(torch.relu(self.batch_norm1(self.conv1(input_))))
        input_ = self.dropout2(torch.relu(self.batch_norm2(self.conv2(input_))))
        input_ = self.dropout3(torch.relu(self.batch_norm3(self.conv3(input_))))
        input_ = input_.transpose(1, 2)
        input_ = self.projection(input_)

        return input_
    
class DecoderPrenet(nn.Module):
    """
    Prenet before passing through the network
    """
    def __init__(self, input_size, hidden_size, output_size, p=0.5):
        """
        :param input_size: dimension of input
        :param hidden_size: dimension of hidden unit
        :param output_size: dimension of output
        """
        super(DecoderPrenet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer = nn.Sequential(OrderedDict([
             ('fc1', Linear(self.input_size, self.hidden_size)),
             ('relu1', nn.ReLU()),
             ('dropout1', nn.Dropout(p)),
             ('fc2', Linear(self.hidden_size, self.output_size)),
             ('relu2', nn.ReLU()),
             ('dropout2', nn.Dropout(p)),
        ]))

    def forward(self, input_):

        out = self.layer(input_)

        return out
    

class PostConvNet(nn.Module):
    """
    Post Convolutional Network (mel --> mel)
    """

    def __init__(self, num_hidden, in_channels, out_channels):
        """
        
        :param num_hidden: dimension of hidden 
        """
        super(PostConvNet, self).__init__()
        self.conv1 = Conv(in_channels=in_channels,
                          #in_channels=hp.num_mels * hp.outputs_per_step,
                          out_channels=num_hidden,
                          kernel_size=5,
                          padding=4,
                          w_init='tanh')
        self.conv_list = clones(Conv(in_channels=num_hidden,
                                     out_channels=num_hidden,
                                     kernel_size=5,
                                     padding=4,
                                     w_init='tanh'), 3)
        self.conv2 = Conv(in_channels=num_hidden,
                          out_channels=out_channels,
                          #out_channels=hp.num_mels * hp.outputs_per_step,
                          kernel_size=5,
                          padding=4)

        self.batch_norm_list = clones(nn.BatchNorm1d(num_hidden), 3)
        self.pre_batchnorm = nn.BatchNorm1d(num_hidden)

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout_list = nn.ModuleList(
            [nn.Dropout(p=0.1) for _ in range(3)])

    def forward(self, input_, mask=None):
        # Causal Convolution (for auto-regressive)
        input_ = self.dropout1(
            torch.tanh(self.pre_batchnorm(self.conv1(input_)[:, :, :-4])))
        for batch_norm, conv, dropout in zip(self.batch_norm_list, self.conv_list, self.dropout_list):
            input_ = dropout(torch.tanh(batch_norm(conv(input_)[:, :, :-4])))
        input_ = self.conv2(input_)[:, :, :-4]
        return input_


class Attention(nn.Module):

    def __init__(self):
        super(Attention, self).__init__()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None, dk=64):
        # |Q| = (batch_size, m, hidden_size)
        # |K| = |V| = (batch_size, n, hidden_size)
        # |mask| = (batch_size, m, n)

        w = torch.bmm(Q, K.transpose(1, 2))
        # |w| = (batch_size, m, n)
        if mask is not None:
            assert w.size() == mask.size()
            w.masked_fill_(mask, -float('inf'))

        w = self.softmax(w / (dk**.5))
        c = torch.bmm(w, V)
        # |c| = (batch_size, m, hidden_size)

        return c


class MultiHeadAttention(nn.Module):

    def __init__(self, hidden_size, n_splits):
        super(MultiHeadAttention, self).__init__()

        self.hidden_size = hidden_size
        self.n_splits = n_splits

        # Note that we don't have to declare each linear layer, separately.
        self.Q_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.K_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

        self.attn = Attention()

    def forward(self, Q, K, V, mask=None):
        # |Q|    = (batch_size, m, hidden_size)
        # |K|    = (batch_size, n, hidden_size)
        # |V|    = |K|
        # |mask| = (batch_size, m, n)

        QWs = self.Q_linear(Q).split(self.hidden_size // self.n_splits, dim=-1)
        KWs = self.K_linear(K).split(self.hidden_size // self.n_splits, dim=-1)
        VWs = self.V_linear(V).split(self.hidden_size // self.n_splits, dim=-1)
        # |QW_i| = (batch_size, m, hidden_size / n_splits)
        # |KW_i| = |VW_i| = (batch_size, n, hidden_size / n_splits)

        QWs = torch.cat(QWs, dim=0)
        KWs = torch.cat(KWs, dim=0)
        VWs = torch.cat(VWs, dim=0)
        # |QWs| = (batch_size * n_splits, m, hidden_size / n_splits)
        # |KWs| = |VWs| = (batch_size * n_splits, n, hidden_size / n_splits)

        if mask is not None:
            mask = torch.cat([mask for _ in range(self.n_splits)], dim=0)
            # |mask| = (batch_size * n_splits, m, n)

        c = self.attn(
            QWs, KWs, VWs,
            mask=mask,
            dk=self.hidden_size // self.n_splits,
        )
        # |c| = (batch_size * n_splits, m, hidden_size / n_splits)

        # We need to restore temporal mini-batchfied multi-head attention results.
        c = c.split(Q.size(0), dim=0)
        # |c_i| = (batch_size, m, hidden_size / n_splits)
        c = self.linear(torch.cat(c, dim=-1))
        # |c| = (batch_size, m, hidden_size)

        return c
