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


SOS_ID = 0
EOS_ID = 0

class NAO(nn.Module):
    def __init__(self,
                 encoder_layers,
                 decoder_layers,
                 mlp_layers,
                 mlp_hidden_size,
                 hidden_size,
                 vocab_size,
                 dropout,
                 source_length,
                 encoder_length,
                 decoder_length,
                 ):
        super(NAO, self).__init__()
        self.encoder = Encoder(
            encoder_layers,
            hidden_size,
            vocab_size,
            dropout,
            source_length,
            encoder_length,
        )
        # NEED TO MODIFY
        self.predictor = Predictor()

        self.decoder = Decoder(
            decoder_layers,
            hidden_size,
            vocab_size,
            dropout,
            decoder_length,
        )

        self.flatten_parameters()
    
    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    # NEED TO MODIFY
    def forward(self, input_variable, target_variable=None):
        encoder_outputs, encoder_hidden = self.encoder(input_variable)
        arch_emb, predict_value = self.predictor(encoder_outputs)
        decoder_hidden = (arch_emb.unsqueeze(0), arch_emb.unsqueeze(0))
        decoder_outputs, archs = self.decoder(target_variable, decoder_hidden, encoder_outputs)
        return predict_value, decoder_outputs, archs

    # NEED TO MODIFY
    def generate_new_arch(self, input_variable, predict_lambda=1, direction='-'):
        encoder_outputs, encoder_hidden = self.encoder(input_variable)
        new_encoder_outputs, new_arch_emb, new_predict_value = self.predictor.infer(encoder_outputs)
        new_encoder_hidden = (new_arch_emb.unsqueeze(0), new_arch_emb.unsqueeze(0))
        decoder_outputs, new_archs = self.decoder(None, new_encoder_hidden, new_encoder_outputs)
        return new_archs, new_predict_value


class SiameseNAO(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.nao = NAO(**kwargs)
        self.regressor = nn.Sequential(
            nn.Linear(kwargs['hidden_size'] * 2, kwargs['hidden_size'] * 2),
            nn.ReLU(),
            nn.Linear(kwargs['hidden_size'] * 2, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def forward_once(model, input_variable, target_variable):
        encoder_outputs, encoder_hidden, arch_emb, _ = model.encoder(input_variable)
        decoder_hidden = (arch_emb.unsqueeze(0), arch_emb.unsqueeze(0))
        decoder_outputs, archs = model.decoder(target_variable, decoder_hidden, encoder_outputs)
        return arch_emb, decoder_outputs

    def forward(self, input_1, input_2, target_1, target_2):
        embedding_1, decoder_outputs_1 = self.forward_once(self.nao, input_1, target_1)
        embedding_2, decoder_outputs_2 = self.forward_once(self.nao, input_2, target_2)

        distance_prediction = self.regressor(torch.cat((embedding_1, embedding_2), dim=1))
        return distance_prediction, decoder_outputs_1, decoder_outputs_2
