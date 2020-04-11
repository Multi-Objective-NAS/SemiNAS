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
import networkx

import utils
from controller import SiameseNAO, NAO
from nasbench import api


parser = argparse.ArgumentParser()
# Basic model parameters.
parser.add_argument('--only_self_train', type=bool, default=True)
parser.add_argument('--data', type=str, default='data')
parser.add_argument('--output_dir', type=str, default='models')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--seed_arch', type=int, default=100)
parser.add_argument('--random_arch', type=int, default=10000)
parser.add_argument('--nodes', type=int, default=7)
parser.add_argument('--new_arch', type=int, default=100)
parser.add_argument('--k', type=int, default=100)
parser.add_argument('--encoder_layers', type=int, default=1)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--mlp_layers', type=int, default=2)
parser.add_argument('--mlp_hidden_size', type=int, default=64)
parser.add_argument('--decoder_layers', type=int, default=1)
parser.add_argument('--source_length', type=int, default=27)
parser.add_argument('--encoder_length', type=int, default=27)
parser.add_argument('--decoder_length', type=int, default=27)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--l2_reg', type=float, default=1e-4)
parser.add_argument('--vocab_size', type=int, default=7)
parser.add_argument('--max_step_size', type=int, default=100)
parser.add_argument('--trade_off', type=float, default=0.6)
parser.add_argument('--pretrain_epochs', type=int, default=10000)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--up_sample_ratio', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--grad_bound', type=float, default=5.0)
parser.add_argument('--iteration', type=float, default=2)
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')


def get_distances(graphs_1, graphs_2, max_distance=6):
    def _is_match(n1, n2):
        return n1['op'] == n2['op']

    def seq_to_graph(seq):
        matrix, ops = utils.convert_seq_to_arch(seq)
        g = networkx.DiGraph()
        for i, op in enumerate(ops):
            g.add_node(i, op=op)
        edge_list = [(i, j) for i in range(7) for j in range(7) if matrix[i][j]]
        g.add_edges_from(edge_list)
        return g

    distances = []
    for i, (g1, g2) in enumerate(zip(graphs_1, graphs_2)):
        distances.append(
            networkx.graph_edit_distance(
                seq_to_graph(g1),
                seq_to_graph(g2),
                node_match=_is_match,
                upper_bound=max_distance-1,
            )
            or max_distance
        )

    return torch.FloatTensor([[d / max_distance if d is not None else 1 for d in distances]])


def controller_selftrain(queue_1, queue_2, model, optimizer):
    objs = utils.AvgrageMeter()
    mse = utils.AvgrageMeter()
    nll = utils.AvgrageMeter()
    model.train()
    for step, (sample_1, sample_2) in enumerate(zip(queue_1, queue_2)):
        encoder_input_1 = utils.move_to_cuda(sample_1['encoder_input'])
        decoder_input_1 = utils.move_to_cuda(sample_1['decoder_input'])
        decoder_target_1 = utils.move_to_cuda(sample_1['decoder_target'])

        encoder_input_2 = utils.move_to_cuda(sample_2['encoder_input'])
        decoder_input_2 = utils.move_to_cuda(sample_2['decoder_input'])
        decoder_target_2 = utils.move_to_cuda(sample_2['decoder_target'])

        distances = utils.move_to_cuda(get_distances(
            sample_1['encoder_input'].numpy(), sample_2['encoder_input'].numpy()
        ))

        optimizer.zero_grad()
        distance_prediction, decoder_outputs_1, decoder_outputs_2 = model(
            encoder_input_1, encoder_input_2, decoder_input_1, decoder_input_2,
        )

        loss_nll_1 = F.nll_loss(
            decoder_outputs_1.contiguous().view(-1, decoder_outputs_1.size(-1)),
            decoder_target_1.view(-1)
        )
        loss_nll_2 = F.nll_loss(
            decoder_outputs_2.contiguous().view(-1, decoder_outputs_2.size(-1)),
            decoder_target_2.view(-1)
        )
        loss_mse = F.mse_loss(distances.squeeze(), distance_prediction.squeeze())
        loss = args.trade_off * loss_mse + (1 - args.trade_off) * (loss_nll_1 + loss_nll_2)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_bound)
        optimizer.step()

        n = encoder_input_1.size(0)
        objs.update(loss.data, n)
        mse.update(loss_mse.data, n)
        nll.update((loss_nll_1 + loss_nll_2).data, n)

        logging.info("step %04d train loss %.6f mse %.6f ce %.6f", step, objs.avg, mse.avg, nll.avg)

    return objs.avg, mse.avg, nll.avg


def selftrain_controller(model, train_input, epochs):
    logging.info('Train data: {}'.format(len(train_input)))
    controller_train_dataset = utils.ControllerDataset(train_input, train=True)
    controller_train_queue_1 = torch.utils.data.DataLoader(
        controller_train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    controller_train_queue_2 = torch.utils.data.DataLoader(
        controller_train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
    torch.save(controller.state_dict(), 'self_trained_test.pth')
    for epoch in range(1, epochs + 1):
        loss, mse, ce = controller_selftrain(controller_train_queue_1, controller_train_queue_2, model, optimizer)
        logging.info("epoch %04d train loss %.6f mse %.6f ce %.6f", epoch, loss, mse, ce)
        torch.save(controller.state_dict(), 'self_trained_%d.pth' % epoch)


def controller_infer(queue, model, step, direction='+'):
    new_arch_list = []
    new_predict_values = []
    model.eval()
    for i, sample in enumerate(queue):
        encoder_input = utils.move_to_cuda(sample['encoder_input'])
        model.zero_grad()
        new_arch, new_predict_value = model.generate_new_arch(encoder_input, step, direction=direction)
        new_arch_list.extend(new_arch.data.squeeze().tolist())
        new_predict_values.extend(new_predict_value.data.squeeze().tolist())
    return new_arch_list, new_predict_values


def train_controller(model, train_input, train_target, epochs):
    logging.info('Train data: {}'.format(len(train_input)))
    controller_train_dataset = utils.ControllerDataset(train_input, train_target, True)
    controller_train_queue = torch.utils.data.DataLoader(
        controller_train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
    for epoch in range(1, epochs + 1):
        loss, mse, ce = controller_train(controller_train_queue, model, optimizer)
        logging.info("epoch %04d train loss %.6f mse %.6f ce %.6f", epoch, loss, mse, ce)


def generate_synthetic_controller_data(nasbench, model, base_arch=None, random_arch=0, direction='+'):
    random_synthetic_input = []
    random_synthetic_target = []
    if random_arch > 0:
        while len(random_synthetic_input) < random_arch:
            seq = utils.generate_arch(1, nasbench)[1][0]
            if seq not in random_synthetic_input and seq not in base_arch:
                random_synthetic_input.append(seq)
        
        controller_synthetic_dataset = utils.ControllerDataset(random_synthetic_input, None, False)      
        controller_synthetic_queue = torch.utils.data.DataLoader(controller_synthetic_dataset, batch_size=len(controller_synthetic_dataset), shuffle=False, pin_memory=True)

        with torch.no_grad():
            model.eval()
            for sample in controller_synthetic_queue:
                encoder_input = sample['encoder_input'].cuda()
                _, _, _, predict_value = model.encoder(encoder_input)
                random_synthetic_target += predict_value.data.squeeze().tolist()
        assert len(random_synthetic_input) == len(random_synthetic_target)
    synthetic_input = random_synthetic_input
    synthetic_target = random_synthetic_target
    assert len(synthetic_input) == len(synthetic_target)
    return synthetic_input, synthetic_target


def main():
    if not torch.cuda.is_available():
        logging.info('No GPU found!')
        sys.exit(1)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = True

    logging.info("Args = %s", args)

    args.source_length = args.encoder_length = args.decoder_length = (args.nodes + 2) * (args.nodes - 1) // 2

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
        decoder_length=args.decoder_length,
    )

    logging.info("param size = %d", utils.count_parameters(controller))
    controller = controller.cuda()

    # Self-train start
    _, self_seq_pool, _ = utils.generate_arch(None, nasbench, need_perf=False)
    logging.info('Self-train ED')
    selftrain_controller(controller, self_seq_pool, epochs=10)
    logging.info('Finish self-train ED')
    if args.only_self_train:
        sys.exit(0)
    controller = controller.nao
    # Self-train end

    child_arch_pool, child_seq_pool, child_arch_pool_valid_acc = utils.generate_arch(args.seed_arch, nasbench, need_perf=True)

    arch_pool = []
    seq_pool = []
    arch_pool_valid_acc = []
    for i in range(args.iteration+1):
        logging.info('Iteration {}'.format(i+1))
        if not child_arch_pool_valid_acc:
            for arch in child_arch_pool:
                data = nasbench.query(arch)
                child_arch_pool_valid_acc.append(data['validation_accuracy'])

        arch_pool += child_arch_pool
        arch_pool_valid_acc += child_arch_pool_valid_acc
        seq_pool += child_seq_pool

        arch_pool_valid_acc_sorted_indices = np.argsort(arch_pool_valid_acc)[::-1]
        arch_pool = [arch_pool[i] for i in arch_pool_valid_acc_sorted_indices]
        seq_pool = [seq_pool[i] for i in arch_pool_valid_acc_sorted_indices]
        arch_pool_valid_acc = [arch_pool_valid_acc[i] for i in arch_pool_valid_acc_sorted_indices]
        with open(os.path.join(args.output_dir, 'arch_pool.{}'.format(i)), 'w') as fa:
            for arch, seq, valid_acc in zip(arch_pool, seq_pool, arch_pool_valid_acc):
                fa.write('{}\t{}\t{}\t{}\n'.format(arch.matrix, arch.ops, seq, valid_acc))
        for arch_index in range(10):
            print('Top 10 architectures:')
            print('Architecutre connection:{}'.format(arch_pool[arch_index].matrix))
            print('Architecture operations:{}'.format(arch_pool[arch_index].ops))
            print('Valid accuracy:{}'.format(arch_pool_valid_acc[arch_index]))

        if i == args.iteration:
            print('Final architectures:')
            for arch_index in range(10):
                print('Architecutre connection:{}'.format(arch_pool[arch_index].matrix))
                print('Architecture operations:{}'.format(arch_pool[arch_index].ops))
                print('Valid accuracy:{}'.format(arch_pool_valid_acc[arch_index]))
                fs, cs = nasbench.get_metrics_from_spec(arch_pool[arch_index])
                test_acc = np.mean([cs[108][j]['final_test_accuracy'] for j in range(3)])
                print('Mean test accuracy:{}'.format(test_acc))
            break

        train_encoder_input = seq_pool
        min_val = min(arch_pool_valid_acc)
        max_val = max(arch_pool_valid_acc)
        train_encoder_target = [(i - min_val) / (max_val - min_val) for i in arch_pool_valid_acc]

        # Pre-train
        logging.info('Pre-train EPD')
        train_controller(controller, train_encoder_input, train_encoder_target, args.pretrain_epochs)
        logging.info('Finish pre-training EPD')

        # Generate synthetic data
        logging.info('Generate synthetic data for EPD')
        synthetic_encoder_input, synthetic_encoder_target = generate_synthetic_controller_data(nasbench, controller, train_encoder_input, args.random_arch)
        if args.up_sample_ratio is None:
            up_sample_ratio = np.ceil(args.random_arch / len(train_encoder_input)).astype(np.int)
        else:
            up_sample_ratio = args.up_sample_ratio
        all_encoder_input = train_encoder_input * up_sample_ratio + synthetic_encoder_input
        all_encoder_target = train_encoder_target * up_sample_ratio + synthetic_encoder_target
        # Train
        logging.info('Train EPD')
        train_controller(controller, all_encoder_input, all_encoder_target, args.epochs)
        logging.info('Finish training EPD')
        
    
        new_archs = []
        new_seqs = []
        predict_step_size = 0
        unique_input = train_encoder_input + synthetic_encoder_input
        unique_target = train_encoder_target + synthetic_encoder_target
        unique_indices = np.argsort(unique_target)[::-1]
        unique_input = [unique_input[i] for i in unique_indices]
        topk_archs = unique_input[:args.k]
        controller_infer_dataset = utils.ControllerDataset(topk_archs, None, False)
        controller_infer_queue = torch.utils.data.DataLoader(controller_infer_dataset, batch_size=len(controller_infer_dataset), shuffle=False, pin_memory=True)
        
        while len(new_archs) < args.new_arch:
            predict_step_size += 1
            logging.info('Generate new architectures with step size %d', predict_step_size)
            new_seq, new_perfs = controller_infer(controller_infer_queue, controller, predict_step_size, direction='+')
            for seq in new_seq:
                matrix, ops = utils.convert_seq_to_arch(seq)
                arch = api.ModelSpec(matrix=matrix, ops=ops)
                if nasbench.is_valid(arch) and len(arch.ops) == 7 and seq not in train_encoder_input and seq not in new_seqs:
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
        child_arch_pool_test_acc = []
        logging.info("Generate %d new archs", len(child_arch_pool))

    print(nasbench.get_budget_counters())


if __name__ == '__main__':
    main()
