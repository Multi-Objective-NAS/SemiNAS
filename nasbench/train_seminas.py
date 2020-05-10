import os
import sys
import glob
import time
import copy
import logging
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import utils
from controller import NAO, SiameseNAO
from nasbench import api
from train_predictor import train_predictor_w_encoder_batch, train_predictor
from multi_objective_sort import multi_objective_sort
import easydict
from livelossplot import PlotLosses

'''
MODIFY:
encoder_target => predictor_acc, predictor_lat
Separate training step into 2 step.
Get encoder_outputs and log_prob(decoder_outputs) by enc_dec func.
Enter encoder_outputs into predictor and triain it.
Then, get loss_1 for acc, lat.
'''


def controller_train(train_queue, model, optimizer, args, epoch):
    objs = utils.AvgrageMeter()
    mse = utils.AvgrageMeter()
    nll = utils.AvgrageMeter()
    model.train()
    for step, sample in enumerate(train_queue):
        encoder_input = utils.move_to_cuda(sample['encoder_input'])
        predictor_acc = utils.move_to_cuda(sample['predictor_acc'])
        predictor_lat = utils.move_to_cuda(sample['predictor_lat'])
        decoder_input = utils.move_to_cuda(sample['decoder_input'])
        decoder_target = utils.move_to_cuda(sample['decoder_target'])

        # (100, 27)
        #print("encoder_inputs: ", encoder_input.size())
        # (100, 27,64)
        #print("encoder_outputs: ",encoder_outputs.size())
        
        loss_1 = train_predictor_w_encoder_batch(model, encoder_input=encoder_input,target_acc=predictor_acc, target_lat=predictor_lat, optimizer=optimizer)
        # loss_1 = F.mse_loss(predict_value.squeeze(), predictor_target.squeeze())
        
        optimizer.zero_grad()
        encoder_outputs, log_prob, archs = model.enc_dec(encoder_input, decoder_input)
        loss_2 = F.nll_loss(log_prob.contiguous().view(-1, log_prob.size(-1)), decoder_target.view(-1))
        
        loss = args.trade_off * loss_1 + (1 - args.trade_off) * loss_2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_bound)
        optimizer.step()

        n = encoder_input.size(0)
        objs.update(loss.data, n)
        mse.update(loss_1.data, n)
        nll.update(loss_2.data, n)

    return objs.avg, mse.avg, nll.avg


'''
MODIFY:
- Find grads on acc and latency w.r.t encoder_outputs
- Update encoder_outputs by adding or subtracting grads
'''


def controller_infer(queue, model, step, direction='+'):
    new_arch_list = []
    new_predict_accs = []
    new_predict_lats = []
    model.eval()

    for i, sample in enumerate(queue):
        encoder_input = utils.move_to_cuda(sample['encoder_input'])
        model.zero_grad()
        new_arch, new_predict_acc, new_predict_lat = model.generate_new_arch(encoder_input, step, direction=direction)

        new_arch_list.extend(new_arch.data.squeeze().tolist())
        new_predict_accs.extend(new_predict_acc.data.squeeze().tolist())
        new_predict_lats.extend(new_predict_lat.data.squeeze().tolist())

    return new_arch_list, new_predict_accs, new_predict_lats


'''
Modify:
use for training only predictor
'''


def train_only_predictor(model, seq_input, acc_input, lat_input, epochs, args, val_dst=0.2):
    train_input = []
    for seq in seq_input:
        encoder_output, _ = model.encoder(seq)
        train_input.append(encoder_output)

    tot_size = len(train_input)
    val_size = int(val_dst * tot_size)
    train_size = tot_size - val_size

    val_input = train_input[-val_size:]
    val_acc = acc_input[-val_size:]
    val_lat = lat_input[-val_size:]

    train_input = train_input[:train_size]
    train_acc = acc_input[:train_size]
    train_lat = lat_input[:train_size]

    predictor_train_dataset = utils.ControllerDataset(inputs=train_input, accs=train_acc, lats=train_lat, train=True)
    predictor_train_queue = torch.utils.data.DataLoader(
        predictor_train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    predictor_val_dataset = utils.ControllerDataset(inputs=val_input, accs=val_acc, lats=val_lat, train=True)
    predictor_val_queue = torch.utils.data.DataLoader(
        predictor_val_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    train_predictor(model.predictor, predictor_train_queue, predictor_val_queue, epochs, val_dst,
                    batch_size=args.batch_size, lr=args.lr, l2_reg=args.l2_reg,
                    grad_bound=args.grad_bound)


def train_controller(model, train_input, train_acc, train_lat, epochs, args):
    # train_input, train_acc, train_lat : list
    logging.info('Train data: {}'.format(len(train_input)))
    liveloss = PlotLosses()
    
    controller_train_dataset = utils.ControllerDataset(train_input, train_acc, train_lat, True)
    controller_train_queue = torch.utils.data.DataLoader(
        controller_train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)    
    model.predictor.init_scale()
    
    logs = {}
    for epoch in range(1, epochs + 1):
        loss, mse, ce = controller_train(controller_train_queue, model, optimizer, args, epoch)
        logging.info("epoch %04d train loss %.6f mse %.6f ce %.6f", epoch, loss, mse, ce)
        
        logs['ACC & Lat Loss'] = mse
        logs['Reconstruction Loss'] = ce
        liveloss.update(logs)
        liveloss.send()


def generate_synthetic_controller_data(nasbench, model, base_arch=None, random_arch=0, direction='+'):
    random_synthetic_input = []
    random_synthetic_acc = []
    random_synthetic_lat = []
    if random_arch > 0:
        while len(random_synthetic_input) < random_arch:
            seq = utils.generate_arch(1, nasbench)[1][0]
            if seq not in random_synthetic_input and seq not in base_arch:
                random_synthetic_input.append(seq)

        controller_synthetic_dataset = utils.ControllerDataset(random_synthetic_input, None, False)
        controller_synthetic_queue = torch.utils.data.DataLoader(controller_synthetic_dataset,
                                                                 batch_size=len(controller_synthetic_dataset),
                                                                 shuffle=False, pin_memory=True)

        with torch.no_grad():
            model.eval()
            for sample in controller_synthetic_queue:
                encoder_input = sample['encoder_input'].cuda()
                encoder_outputs, encoder_hidden = model.encoder(encoder_input)
                arch_emb, predict_acc, predict_lat = model.predictor(encoder_outputs)
                random_synthetic_acc += predict_acc.data.squeeze().tolist()
                random_synthetic_lat += predict_lat.data.squeeze().tolist()

        assert len(random_synthetic_input) == len(random_synthetic_acc)
        assert len(random_synthetic_lat) == len(random_synthetic_lat)

    synthetic_input = random_synthetic_input
    synthetic_acc = random_synthetic_acc
    synthetic_lat = random_synthetic_lat
    assert len(synthetic_input) == len(synthetic_acc)
    assert len(synthetic_lat) == len(synthetic_lat)
    return synthetic_input, synthetic_acc, synthetic_lat


def main(nasbench=None):
    # if not torch.cuda.is_available():
    #    logging.info('No GPU found!')
    #    sys.exit(1)

    args = easydict.EasyDict({
        "data": "data",
        "output_dir": './Output/',
        "seed": 1,
        "seed_arch": 100,
        "random_arch": 10000,
        "nodes": 7,
        "new_arch": 100,
        "k": 100,
        "encoder_layers": 1,
        "hidden_size": 64,
        "mlp_layers": 2,
        "mlp_hidden_size": 64,
        "decoder_layers": 1,
        "source_length": 27,
        "encoder_length": 27,
        "decoder_length": 27,
        "dropout": 0.1,
        "l2_reg": 1e-4,
        "vocab_size": 7,
        "max_step_size": 100,
        "trade_off": 0.6,
        "pretrain_epochs": 10000,
        "epochs": 1000,
        "up_sample_ratio": 100,
        "batch_size": 100,
        "lr": 0.001,
        "optimizer": "adam",
        "grad_bound": 5.0,
        "iteration": 2,
        "load": False,
        "load_iteration": 0,
        "save": True,
        "save_path": './Output/Models/',
        "iteration_save" : 1
    })

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')

    device = torch.device('cuda')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = True

    logging.info("Args = %s", args)

    args.source_length = args.encoder_length = args.decoder_length = (args.nodes + 2) * (args.nodes - 1) // 2

    if nasbench is None:
        nasbench = api.NASBench(os.path.join(args.data, 'nasbench_full.tfrecord'))

    controller = SiameseNAO(
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        mlp_layers=args.mlp_layers,
        mlp_hidden_size=args.mlp_hidden_size,
        hidden_size=args.hidden_size,
        vocab_size=args.vocab_size,
        dropout=args.dropout,
        source_length=args.source_length,
        encoder_length=args.encoder_length,
        decoder_length=args.decoder_length
    )

    # load pretrained model
    pretrained_dict = torch.load(os.path.join(args.data, 'self_trained_2.pth'))
    controller_dict = controller.state_dict()
    controller_dict.update({k: v for k, v in pretrained_dict.items() if k in controller_dict})
    controller.load_state_dict(controller_dict)

    print("Pretrained Model Loaded")

    logging.info("param size = %d", utils.count_parameters(controller))
    controller = controller.cuda()

    if args.load:
        fname = args.save_path + args.fname + 'predictor_' + str(args.load_iteration) + '.dat'
        controller.nao.predictor.load_state_dict(torch.load(fname))

        print('model loaded!, lr: {}'.format(args.lr))
    else:
        args.load_iteration = 1

    print("Start training")

    # ADD LATENCY LIST
    child_arch_pool, child_seq_pool, child_arch_pool_valid_acc, child_arch_pool_lat = utils.generate_arch(
        args.seed_arch, nasbench, need_perf=True)

    arch_pool = []
    seq_pool = []
    arch_pool_valid_acc = []
    arch_pool_lat = []
    for i in range(args.load_iteration, args.iteration + 1):
        logging.info('Iteration {}'.format(i + 1))
        if not child_arch_pool_valid_acc or not child_arch_pool_lat:
            for arch in child_arch_pool:
                data = nasbench.query(arch)
                child_arch_pool_valid_acc.append(data['validation_accuracy'])
                child_arch_pool_lat.append(data['training_time'])

        arch_pool += child_arch_pool
        arch_pool_valid_acc += child_arch_pool_valid_acc
        arch_pool_lat += child_arch_pool_lat
        seq_pool += child_seq_pool

        '''
        MODIFY:
        - pareto sorting
        '''
        multi_objective_sort(arch_pool, seq_pool, arch_pool_valid_acc, arch_pool_lat)

        with open(os.path.join(args.output_dir, 'arch_pool.{}'.format(i)), 'w') as fa:
            for arch, seq, valid_acc, lat in zip(arch_pool, seq_pool, arch_pool_valid_acc, arch_pool_lat):
                fa.write('{}\t{}\t{}\t{}\t{}\n'.format(arch.matrix, arch.ops, seq, valid_acc, lat))
        for arch_index in range(10):
            print('Top 10 architectures:')
            print('Architecutre connection:{}'.format(arch_pool[arch_index].matrix))
            print('Architecture operations:{}'.format(arch_pool[arch_index].ops))
            print('Valid accuracy:{}'.format(arch_pool_valid_acc[arch_index]))
            print('Valid latency:{}'.format(arch_pool_lat[arch_index]))

        if i == args.iteration:
            print('Final architectures:')
            for arch_index in range(10):
                print('Architecutre connection:{}'.format(arch_pool[arch_index].matrix))
                print('Architecture operations:{}'.format(arch_pool[arch_index].ops))
                print('Valid accuracy:{}'.format(arch_pool_valid_acc[arch_index]))
                print('Valid latency:{}'.format(arch_pool_lat[arch_index]))
                fs, cs = nasbench.get_metrics_from_spec(arch_pool[arch_index])
                test_acc = np.mean([cs[108][j]['final_test_accuracy'] for j in range(3)])
                print('Mean test accuracy:{}'.format(test_acc))
            break

        train_encoder_input = seq_pool
        min_val = min(arch_pool_valid_acc)
        max_val = max(arch_pool_valid_acc)
        train_acc_target = [(i - min_val) / (max_val - min_val) for i in arch_pool_valid_acc]
        min_val = min(arch_pool_lat)
        max_val = max(arch_pool_lat)
        train_lat_target = [(i - min_val) / (max_val - min_val) for i in arch_pool_lat]

        # Pre-train
        logging.info('Pre-train EPD')
        train_controller(controller.nao, train_encoder_input, train_acc_target, train_lat_target, args.pretrain_epochs, args)
        #train_only_predictor(controller.nao, train_encoder_input, train_acc_target, train_lat_target, args.pretrain_epochs,args)
        logging.info('Finish pre-training EPD')
        # Generate synthetic data
        logging.info('Generate synthetic data for EPD')
        synthetic_encoder_input, synthetic_acc_target, synthetic_lat_target = generate_synthetic_controller_data(
            nasbench, controller.nao, train_encoder_input, args.random_arch)
        if args.up_sample_ratio is None:
            up_sample_ratio = np.ceil(args.random_arch / len(train_encoder_input)).astype(np.int)
        else:
            up_sample_ratio = args.up_sample_ratio
        all_encoder_input = train_encoder_input * up_sample_ratio + synthetic_encoder_input
        all_acc_target = train_acc_target * up_sample_ratio + synthetic_acc_target
        all_lat_target = train_lat_target * up_sample_ratio + synthetic_lat_target

        # Train
        logging.info('Train EPD')
        train_controller(controller.nao, all_encoder_input, all_acc_target, all_lat_target, args.epochs, args)
        #train_only_predictor(controller.nao, all_encoder_input, all_acc_target, train_lat_target, args.epochs, args)
        logging.info('Finish training EPD')

        new_archs = []
        new_seqs = []
        predict_step_size = 0
        unique_input = train_encoder_input + synthetic_encoder_input
        unique_acc = train_acc_target + synthetic_acc_target
        unique_lat = train_lat_target + synthetic_lat_target
        '''
        MODIFY:
        - pareto sorting
        '''
        multi_objective_sort(unique_input, unique_input, unique_acc, unique_lat)
        topk_archs = unique_input[:args.k]

        controller_infer_dataset = utils.ControllerDataset(topk_archs, None, False)
        controller_infer_queue = torch.utils.data.DataLoader(controller_infer_dataset,
                                                             batch_size=len(controller_infer_dataset), shuffle=False,
                                                             pin_memory=True)

        while len(new_archs) < args.new_arch:
            predict_step_size += 1
            logging.info('Generate new architectures with step size %d', predict_step_size)
            new_seq, new_accs, new_lats = controller_infer(controller_infer_queue, controller.nao, predict_step_size,
                                                           direction='+')
            for seq in new_seq:
                matrix, ops = utils.convert_seq_to_arch(seq)
                arch = api.ModelSpec(matrix=matrix, ops=ops)
                if nasbench.is_valid(arch) and len(
                        arch.ops) == 7 and seq not in train_encoder_input and seq not in new_seqs:
                    new_archs.append(arch)
                    new_seqs.append(seq)
                if len(new_seqs) >= args.new_arch:
                    break
            logging.info('%d new archs generated now', len(new_archs))
            if predict_step_size > args.max_step_size:
                break

        child_arch_pool = new_archs
        child_seq_pool = new_seqs
        child_arch_pool_valid_acc = []
        child_arch_pool_lat = []
        logging.info("Generate %d new archs", len(child_arch_pool))

        # save model checkpoint
        if args.save:
            if i % args.iteration_save == 0:
                fname = args.model_save_path + args.fname + 'controller' + str(i) + '.dat'
                torch.save(controller.state_dict(), fname)

    print(nasbench.get_budget_counters())


if __name__ == '__main__':
    main()
