from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self,
                 layers,
                 hidden_size,
                 vocab_size,
                 dropout,
                 source_length,
                 length,
                 ):
        super(Encoder, self).__init__()
        self.layers = layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.dropout = dropout
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.out_proj = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

    def forward(self, x):
        x = self.embedding(x)
        x = F.dropout(x, self.dropout, training=self.training)
        residual = x
        x, hidden = self.rnn(x)
        x = self.out_proj(x)
        x = residual + x
        x = F.normalize(x, 2, dim=-1)
        encoder_outputs = x
        encoder_hidden = hidden

        return encoder_outputs, encoder_hidden
