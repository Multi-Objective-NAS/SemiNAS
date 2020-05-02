import torch
import torch.nn.functional as F
from torch.autograd import Variable
# from tensorboardX import SummaryWriter
import numpy as np
import utils
from min_norm_solvers import MinNormSolver, gradient_normalizers


class RunningMetric(object):
    def __init__(self, metric_type, n_classes=None):
        self._metric_type = metric_type
        if metric_type == 'ACC':
            self.accuracy = 0.0
            self.num_updates = 0.0
        if metric_type == 'L1':
            self.l1 = 0.0
            self.num_updates = 0.0
        if metric_type == 'IOU':
            if n_classes is None:
                print('ERROR: n_classes is needed for IOU')
            self.num_updates = 0.0
            self._n_classes = n_classes
            self.confusion_matrix = np.zeros((n_classes, n_classes))

    def reset(self):
        if self._metric_type == 'ACC':
            self.accuracy = 0.0
            self.num_updates = 0.0
        if self._metric_type == 'L1':
            self.l1 = 0.0
            self.num_updates = 0.0
        if self._metric_type == 'IOU':
            self.num_updates = 0.0
            self.confusion_matrix = np.zeros((self._n_classes, self._n_classes))

    def _fast_hist(self, pred, gt):
        mask = (gt >= 0) & (gt < self._n_classes)
        hist = np.bincount(
            self._n_classes * gt[mask].astype(int) +
            pred[mask], minlength=self._n_classes ** 2).reshape(self._n_classes, self._n_classes)
        return hist

    def update(self, pred, gt):
        if self._metric_type == 'ACC':
            predictions = pred.data.max(1, keepdim=True)[1]
            self.accuracy += (predictions.eq(gt.data.view_as(predictions)).cpu().sum())
            self.num_updates += predictions.shape[0]

        if self._metric_type == 'L1':
            _gt = gt.data.cpu().numpy()
            _pred = pred.data.cpu().numpy()
            gti = _gt.astype(np.int32)
            mask = gti != 250
            if np.sum(mask) < 1:
                return
            self.l1 += np.sum(np.abs(gti[mask] - _pred.astype(np.int32)[mask]))
            self.num_updates += np.sum(mask)

        if self._metric_type == 'IOU':
            _pred = pred.data.max(1)[1].cpu().numpy()
            _gt = gt.data.cpu().numpy()
            for lt, lp in zip(_pred, _gt):
                self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

    def get_result(self):
        if self._metric_type == 'ACC':
            return {'acc': self.accuracy / self.num_updates}
        if self._metric_type == 'L1':
            return {'l1': self.l1 / self.num_updates}
        if self._metric_type == 'IOU':
            acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
            acc_cls = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum(axis=1)
            acc_cls = np.nanmean(acc_cls)
            iou = np.diag(self.confusion_matrix) / (
                    self.confusion_matrix.sum(axis=1) + self.confusion_matrix.sum(axis=0) - np.diag(
                self.confusion_matrix))
            mean_iou = np.nanmean(iou)
            return {'micro_acc': acc, 'macro_acc': acc_cls, 'mIOU': mean_iou}


'''
MODIFY:
In train_seminas.py, I separate training step into 2 step.
1. encoder and decoder forward for reconstruction loss.
2. train_predictor.py is encoder+predictor forward for loss in acc, lat
 '''


def train_predictor_w_encoder(predictor, input, parameters, target_acc, target_lat, optimizer):
    # parameters : encoder_outputs and model['rep'](parent model for two task)
    # Later, find grads w.r.t parameters
    parameters += predictor.model['rep'].parameters()

    model_params = []
    for m in predictor.model.keys():
        model_params += predictor.model[m].parameters()

    # Scaling the loss functions based on the algorithm choice
    loss_data = {}
    grads = {}
    scale = {}
    mask = None
    labels = {}
    labels['acc'] = target_acc
    labels['lat'] = target_lat

    for t in predictor.tasks:
        # Comptue gradients of each loss function wrt parameters
        optimizer.zero_grad()
        rep, mask = predictor.model['rep'](input, mask)
        out_t = predictor.model[t](rep)
        loss = F.mse_loss[t](out_t, labels[t])
        loss_data[t] = loss.data[0]
        loss.backward()
        grads[t] = []
        # Find grads of loss in task w.r.t. parameters
        for param in parameters:
            if param.grad is not None:
                grads[t].append(Variable(param.grad.data.clone(), requires_grad=False))

    # Normalize all gradients, this is optional and not included in the paper.
    gn = gradient_normalizers(grads, loss_data, "loss+")
    for t in predictor.tasks:
        for gr_i in range(len(grads[t])):
            grads[t][gr_i] = grads[t][gr_i] / gn[t]

    # Frank-Wolfe iteration to compute scales.
    # min_norm = w^2 = alpha * v1^2 + (1-alpha) * v2^2
    # sol = (alpha, 1-alpha)
    sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in model.tasks])
    for i, t in enumerate(predictor.tasks):
        scale[t] = float(sol[i])
        predictor.scales[t] = 0.999 * predictor.scales[t] + 0.001 * scale[t]

    # Back-propagation
    # scale task: alpha * task1 + (1-alpha) * task2
    optimizer.zero_grad()
    rep, _ = predictor.model['rep'](input, mask)
    for i, t in enumerate(predictor.tasks):
        out_t = predictor.model[t](rep)
        loss_t = F.mse_loss[t](out_t, labels[t])
        loss_data[t] = loss_t.data[0]
        if i > 0:
            loss = loss + scale[t] * loss_t
        else:
            loss = scale[t] * loss_t

    # return scaled loss
    return loss


'''
Modify:
Use when
preezeing encoder & decoder
train only predictor
'''


def train_predictor(predictor, train_queue, val_queue, epochs, batch_size, val_dst, lr, l2_reg, grad_bound):
    # model: predictor
    # loss : F.nll_loss(pred, gt)
    met = {}
    for t in predictor.tasks:
        met[t] = RunningMetric(metric_type='ACC')
    model_params = []
    for m in predictor.model.keys():
        model_params += predictor.model[m].parameters

    optimizer = torch.optim.Adam(model_params, lr=lr, weight_decay=l2_reg)
    # writer = SummaryWriter(
    #    log_dir='runs/{}_{}'.format(params['exp_id'], datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")))
    predictor.init_scale()
    obj_ls = {}
    n_iter = 0

    for epoch in range(1, epochs + 1):

        if (epoch + 1) % 10 == 0:
            # Every 50 epoch, half the LR
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.85
            print('Half the learning rate{}'.format(n_iter))
        objs = utils.AvgrageMeter()

        # batch train
        for m in predictor.model.keys():
            predictor[m].train()
        for step, sample in enumerate(train_queue):
            n_iter += 1
            input = utils.move_to_cuda(sample['encoder_output'])
            labels['acc'] = utils.move_to_cuda(sample['predictor_acc'])
            labels['lat'] = utils.move_to_cuda(sample['predictor_lat'])

            # Scaling the loss functions based on the algorithm choice
            loss_data = {}
            grads = {}
            scale = {}
            mask = None
            labels = {}

            for t in predictor.tasks:
                # Comptue gradients of each loss function wrt parameters
                optimizer.zero_grad()
                rep, mask = predictor.model['rep'](input, mask)
                out_t = predictor.model[t](rep)
                loss = F.mse_loss[t](out_t, labels[t])
                loss_data[t] = loss.data[0]
                loss.backward()
                grads[t] = []
                # Find grads of loss in task w.r.t. parameters
                for param in predictor.model['rep'].parameters():
                    if param.grad is not None:
                        grads[t].append(Variable(param.grad.data.clone(), requires_grad=False))

            # Normalize all gradients, this is optional and not included in the paper.
            gn = gradient_normalizers(grads, loss_data, "loss+")
            for t in predictor.tasks:
                for gr_i in range(len(grads[t])):
                    grads[t][gr_i] = grads[t][gr_i] / gn[t]

            # Frank-Wolfe iteration to compute scales.
            # min_norm = w^2 = alpha * v1^2 + (1-alpha) * v2^2
            # sol = (alpha, 1-alpha)
            sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in model.tasks])
            for i, t in enumerate(predictor.tasks):
                scale[t] = float(sol[i])
                predictor.scales[t] = 0.999 * predictor.scales[t] + 0.001 * scale[t]

            # Back-propagation
            # scale task: alpha * task1 + (1-alpha) * task2
            optimizer.zero_grad()
            rep, _ = predictor.model['rep'](input, mask)
            for i, t in enumerate(predictor.tasks):
                out_t = predictor.model[t](rep)
                loss_t = F.mse_loss[t](out_t, labels[t])
                loss_data[t] = loss_t.data[0]
                if i > 0:
                    loss = loss + scale[t] * loss_t
                else:
                    loss = scale[t] * loss_t

            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), grad_bound)
            optimizer.step()

            objs.update(loss.data, batch_size)
            obj_ls.append(objs.avg)

            print('training_loss', loss.data[0], n_iter)
            for t in predictor.tasks:
                print('training_loss_{}'.format(t), loss_data[t], n_iter)

        # evaluation
        for m in predictor.model.keys():
            predictor.model[m].eval()
        tot_loss = {}
        tot_loss['all'] = 0.0
        met = {}
        for t in predictor.tasks:
            tot_loss[t] = 0.0
            met[t] = 0.0

        num_val_batches = 0
        labels_val = {}
        for batch_val in val_queue:
            input = utils.move_to_cuda(batch_val['encoder_output'])
            labels_val['acc'] = utils.move_to_cuda(batch_val['predictor_acc'])
            labels_val['lat'] = utils.move_to_cuda(batch_val['predictor_lat'])

            val_rep, _ = predictor.model['rep'](input, None)
            for t in predictor.tasks:
                out_t_val, _ = predictor.model[t](val_rep, None)
                loss_t = F.mse_loss[t](out_t_val, labels_val[t])
                tot_loss['all'] += loss_t.data[0]
                tot_loss[t] += loss_t.data[0]
                met[t].update(out_t_val, labels_val[t])
            num_val_batches += 1

        for t in predictor.tasks:
            print('validation_loss_{}'.format(t), tot_loss[t] / num_val_batches, n_iter)
            metric_results = met[t].get_result()
            for metric_key in metric_results:
                print('metric_{}_{}'.format(metric_key, t), metric_results[metric_key], n_iter)
            met[t].reset()
        print('validation_loss', tot_loss['all'] / len(val_dst), n_iter)

        if epoch % 3 == 0:
            # Save after every 3 epoch
            state = {'epoch': epoch + 1,
                     'model_rep': predictor.model['rep'].state_dict(),
                     'optimizer_state': optimizer.state_dict()}
            for t in predictor.tasks:
                key_name = 'model_{}'.format(t)
                state[key_name] = predictor.model[t].state_dict()

            torch.save(state, "saved_models/{}_predictor.pkl".format(epoch + 1))


if __name__ == '__main__':
    # train_input, train_acc, train_lat
    '''
    controller_train_dataset = utils.ControllerDataset(train_input, train_acc, train_lat, True)
    controller_train_queue = torch.utils.data.DataLoader(
        controller_train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    '''
    train_predictor()
