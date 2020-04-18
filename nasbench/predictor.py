import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from controller import generate_arch_emb


'''
NEED TO MODIFY:
- think about how to make model for 1-d input
'''
class MultiLeNetR(nn.Module):
    def __init__(self):
        super(MultiLeNetR, self).__init__()
        self.conv1 = nn.Conv1d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv1d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc = nn.Linear(320, 50)

    def dropout1dwithmask(self, x, mask):
        channel_size = x.shape[1]
        if mask is None:
            mask = Variable(torch.bernoulli(torch.ones(1, channel_size, 1) * 0.5).cuda())
        mask = mask.expand(x.shape)
        return mask

    def forward(self, x, mask):
        x = F.relu(F.max_pool1d(self.conv1(x), 2))
        x = self.conv2(x)
        mask = self.dropout1dwithmask(x, mask)
        if self.training:
            x = x * mask
        x = F.relu(F.max_pool1d(x, 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc(x))
        return x, mask


class MultiLeNetO(nn.Module):
    def __init__(self):
        super(MultiLeNetO, self).__init__()
        self.fc1 = nn.Linear(50, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x, mask):
        x = F.relu(self.fc1(x))
        if mask is None:
            mask = Variable(torch.bernoulli(x.data.new(x.data.size()).fill_(0.5)))
        if self.training:
            x = x * mask
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), mask


'''
MODIFY:
Add predictor model
model['rep']: parent model for two tasks.
model['rep'] output ->  model['acc']
                  |-->  model['lat']
'''
class Predictor(nn.Module):
    def __init__(self):
        self.model = {}
        self.model['rep'] = MultiLeNetR()
        self.tasks = ['acc', 'lat']
        for t in self.tasks:
            self.model[t] = MultiLeNetO()
        self.scales = {}

    def init_scale(self):
        for t in self.tasks:
            self.scales[t] = 0

    def forward(self, x):
        # x : encoder_outputs
        arch_emb = generate_arch_emb(x)
        out = {}
        rep, _ = self.model['rep'](x, None)
        for t in self.tasks:
            out[t], _ = self.model[t](rep, None)

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