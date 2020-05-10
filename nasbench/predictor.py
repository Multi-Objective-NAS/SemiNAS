import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch

def generate_arch_emb(x):
    x = torch.mean(x, dim=1)
    x = F.normalize(x, 2, dim=-1)
    arch_emb = x

    return arch_emb

'''
NEED TO MODIFY:
- think about how to make model for 1-d input
'''
class MultiLeNetR(nn.Module):
    def __init__(self, input_size):
        super(MultiLeNetR, self).__init__()
        self.fc = nn.Linear(input_size, input_size)

    def forward(self, x, mask):
        residual = x
        x = F.relu(self.fc(x))
        if mask is None:
            mask = Variable(torch.bernoulli(x.data.new(x.data.size()).fill_(0.5)))
        if self.training:
            x = x * mask
        x = (residual + x) * math.sqrt(0.5)
        return x, mask



class MultiLeNetO(nn.Module):
    def __init__(self, input_size):
        super(MultiLeNetO, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, 1)
        self.prelu = nn.PReLU()

    def forward(self, x):
        residual = x
        x = F.relu(self.fc1(x))
        x = (residual + x) * math.sqrt(0.5)
        out = self.prelu(self.fc2(x))
        return out


'''
MODIFY:
Add predictor model
model['rep']: parent model for two tasks.
model['rep'] output ->  model['acc']
                  |-->  model['lat']
'''
class Predictor(nn.Module):
    def __init__(self, input_size):
        super(Predictor, self).__init__()
        model = {}
        model['rep'] = MultiLeNetR(input_size)
        self.tasks = ['acc', 'lat']
        for t in self.tasks:
            model[t] = MultiLeNetO(input_size)
        self.model = nn.ModuleDict(model)
        self.scales = {}

    def init_scale(self):
        for t in self.tasks:
            self.scales[t] = 0

    def forward(self, x):
        # x : encoder_outputs
        arch_emb = generate_arch_emb(x)
        x = arch_emb
        out = {}
        rep, _ = self.model['rep'](x, None)
        for t in self.tasks:
            out[t]= self.model[t](rep)

        return arch_emb, out['acc'], out['lat']

    '''
    MODIFY:
    - Find grads on acc and latency w.r.t encoder_outputs
    - Update encoder_outputs by adding or subtracting grads
    - Problem: scaled grads for MO needs loss:
    - train 때 average scale을 구해놓고 이를 사용하기! <- 어짜피 scale * 1/2 * (y - y_0)^2 = - scale * ( y-y_0 ) * grads on y_0
       이기 때문에 실제 loss가 있을 때 주는 영향은 grads on y_0의 크기! 따라서 이는 무시해도 되겠다는 생각으로....
    '''
    def infer(self, x, predict_lambda=1, direction='-'):
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
        arch_emb, predict_acc, predict_lat = self.forward(x)

        # -----------------------------------------------------------------------------------------
        # code used grads on accuracy function w.r.t. encoder_outputs
        # Not grads on loss function w.r.t. encoder_outputs
        acc_grads_on_outputs = torch.autograd.grad(predict_acc, encoder_outputs, torch.ones_like(predict_acc))[0]
        lat_grads_on_outputs = torch.autograd.grad(predict_lat, encoder_outputs, torch.ones_like(predict_lat))[0]
        grads_on_outputs = self.scales["acc"]*acc_grads_on_outputs + self.scales["lat"]*lat_grads_on_outputs

        if direction == '+':
            new_encoder_outputs = encoder_outputs + predict_lambda * grads_on_outputs
        elif direction == '-':
            new_encoder_outputs = encoder_outputs - predict_lambda * grads_on_outputs
        else:
            raise ValueError('Direction must be + or -, got {} instead'.format(direction))
        # -----------------------------------------------------------------------------------------

        new_encoder_outputs = F.normalize(new_encoder_outputs, 2, dim=-1)
        new_arch_emb, new_predict_acc, new_predict_lat = self.forward(new_encoder_outputs)

        return predict_acc, predict_lat, new_encoder_outputs, new_arch_emb, new_predict_acc, new_predict_lat
