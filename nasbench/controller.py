import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder
from decoder import Decoder
from predictor import Predictor

SOS_ID = 0
EOS_ID = 0


def generate_arch_emb(x):
    x = torch.mean(x, dim=1)
    x = F.normalize(x, 2, dim=-1)
    arch_emb = x

    return arch_emb


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
                 decoder_length
                 ):
        super(NAO, self).__init__()
        '''
        MODIFY:
        Encoder was encoding + predictor.
        I separate encoder model( encoding ) + arch_emb function + predictor model
        '''
        self.encoder = Encoder(
            encoder_layers,
            hidden_size,
            vocab_size,
            dropout,
            source_length,
            encoder_length
        )
        self.predictor = Predictor(encoder_length)
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

    '''
    MODIFY:
    In train_seminas.py, I separate training step into 2 step.
    This is encoder and decoder forward for reconstruction loss.
    train_predictor.py is encoder+predictor forward for loss in acc, lat.
    '''
    def enc_dec(self, input_variable, target_variable=None):
        encoder_outputs, encoder_hidden = self.encoder(input_variable)
        arch_emb = generate_arch_emb(encoder_outputs)
        decoder_hidden = (arch_emb.unsqueeze(0), arch_emb.unsqueeze(0))
        decoder_outputs, archs = self.decoder(target_variable, decoder_hidden, encoder_outputs)
        return encoder_outputs, decoder_outputs, archs

    '''
    MODIFY:
    Encoder was encoding + predictor.
    I separate encoder model( encoding ) + arch_emb function + predictor model
    '''
    def forward(self, input_variable, target_variable=None):
        encoder_outputs, encoder_hidden = self.encoder(input_variable)
        arch_emb, predict_acc, predict_lat = self.predictor(encoder_outputs)
        decoder_hidden = (arch_emb.unsqueeze(0), arch_emb.unsqueeze(0))
        decoder_outputs, archs = self.decoder(target_variable, decoder_hidden, encoder_outputs)
        return predict_acc, predict_lat, decoder_outputs, archs

    '''
    MODIFY:
    - Find grads on acc and latency w.r.t encoder_outputs
    - Update encoder_outputs by adding or subtracting grads
    '''
    def generate_new_arch(self, input_variable, predict_lambda=1, direction='-'):
        encoder_outputs, encoder_hidden = self.encoder(input_variable)
        predict_acc, predict_lat, new_encoder_outputs, new_arch_emb, new_predict_acc, new_predict_lat = self.predictor.infer(encoder_outputs)
        new_encoder_hidden = (new_arch_emb.unsqueeze(0), new_arch_emb.unsqueeze(0))
        decoder_outputs, new_archs = self.decoder(None, new_encoder_hidden, new_encoder_outputs)
        return new_archs, new_predict_acc, new_predict_lat


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
