import os
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder
from decoder import Decoder
import utils

class Predictor(nn.Module):
    def __init__(self):

    def generate_arch_emb(self, x):
        x = torch.mean(x, dim=1)
        x = F.normalize(x, 2, dim=-1)
        arch_emb = x

        return arch_emb

    def forward(self, x):
        # x : encoder_outputs
        arch_emb = self.generate_arch_emb(x)

        residual = x
        for i, mlp_layer in enumerate(self.mlp):
            x = mlp_layer(x)
            x = F.relu(x)
            if i != self.mlp_layers:
                x = F.dropout(x, self.dropout, training=self.training)
        x = (residual + x) * math.sqrt(0.5)
        x = self.regressor(x)
        predict_value = torch.sigmoid(x)
        return arch_emb, predict_value

    def infer(self, x, predict_lambda, direction='-'):
        # x : encoder_outputs
        '''
        --------------------------------------------------------------------------------------------------
        1)  Encoder     = seq -> encoder_outputs -> arch_emb
            Predictor   = arch_emb -> acc, latency
            then, grads on arch_emb
        2)  Encoder     = seq -> encoder_outputs
            Predictor   = encoder_outputs -> arch_emb -> acc, latency
            then, grads on encoder_outputs

        I chose 1 then,

        grads_on_arch_emb = torch.autograd.grad(predict_value, x, torch.ones_like(predict_value))[0]
        if direction == '+':
            new_arch_emb = x + predict_lambda * grads_on_arch_emb
        elif direction == '-':
            new_arch_emb = x - predict_lambda * grads_on_arch_emb
        else:
            raise ValueError('Direction must be + or -, got {} instead'.format(direction))

        new_arch_emb = F.normalize(new_arch_emb, 2, dim=-1)
        new_predict_value = self.forward_predictor(new_arch_emb)

        return predict_value, new_arch_emb, new_arch_emb, new_predict_value
        --------------------------------------------------------------------------------------------------
        '''
        encoder_outputs = x
        arch_emb, predict_value = self.forward(x)

        # MGDA...
        grads_on_outputs = torch.autograd.grad(predict_value, encoder_outputs, torch.ones_like(predict_value))[0]
        if direction == '+':
            new_encoder_outputs = encoder_outputs + predict_lambda * grads_on_outputs
        elif direction == '-':
            new_encoder_outputs = encoder_outputs - predict_lambda * grads_on_outputs
        else:
            raise ValueError('Direction must be + or -, got {} instead'.format(direction))

        new_encoder_outputs = F.normalize(new_encoder_outputs, 2, dim=-1)
        new_arch_emb = torch.mean(new_encoder_outputs, dim=1)
        new_arch_emb = F.normalize(new_arch_emb, 2, dim=-1)
        new_predict_value = self.forward(new_encoder_outputs)

        return predict_value, new_encoder_outputs, new_arch_emb, new_predict_value
