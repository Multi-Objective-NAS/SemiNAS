from timeit import default_timer as timer
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import losses
import datasets
import metrics
import model_selector
from min_norm_solvers import MinNormSolver, gradient_normalizers

NUM_EPOCHS = 100

def train_predictor(model, input, parameters, target_acc, target_lat, optimizer):
    # NEED TO MODIFY
    parameters += model['rep'].parameters()

    model_params = []
    for m in model:
        model_params += model[m].parameters()

    # Scaling the loss functions based on the algorithm choice
    loss_data = {}
    grads = {}
    scale = {}
    mask = None
    masks = {}
    labels = {}
    labels['acc'] = target_acc
    labels['lat'] = target_lat

    for t in model.tasks:
        # Comptue gradients of each loss function wrt parameters
        optimizer.zero_grad()
        rep, mask = model['rep'](input, mask)
        out_t, masks[t] = model[t](rep, None)
        loss = F.mse_loss[t](out_t, labels[t])
        loss_data[t] = loss.data[0]
        loss.backward()
        grads[t] = []
        for param in parameters:
            if param.grad is not None:
                grads[t].append(Variable(param.grad.data.clone(), requires_grad=False))

    # Normalize all gradients, this is optional and not included in the paper.
    gn = gradient_normalizers(grads, loss_data, "loss+")
    for t in model.tasks:
        for gr_i in range(len(grads[t])):
            grads[t][gr_i] = grads[t][gr_i] / gn[t]

    # Frank-Wolfe iteration to compute scales.
    sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in model.tasks])
    for i, t in enumerate(model.tasks):
        scale[t] = float(sol[i])

    # Scaled back-propagation
    optimizer.zero_grad()
    rep, _ = model['rep'](input, mask)
    for i, t in enumerate(model.tasks):
        out_t, _ = model[t](rep, masks[t])
        loss_t = F.mse_loss[t](out_t, labels[t])
        loss_data[t] = loss_t.data[0]
        if i > 0:
            loss = loss + scale[t] * loss_t
        else:
            loss = scale[t] * loss_t
    return loss


if __name__ == '__main__':
    train_predictor()